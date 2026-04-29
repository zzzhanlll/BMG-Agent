# bmg_agent.py
import os
import json
import logging
import pandas as pd
from typing import List, Dict, Any, Generator
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory
from operator import itemgetter
import re
from BMGPrompt import (
    CYPHER_SYSTEM_PROMPT,
    OUTPUT_PROMPT,
    ANSWER_SYSTEM_PROMPT,
    PROPERTIES_DESCRIPTION,
    FEWSHOT_EXAMPLES)
import streamlit as st

NEO4J_URI = st.secrets['neo4j_credentials']['NEO4J_URI']
NEO4J_USERNAME = st.secrets['neo4j_credentials']['NEO4J_USERNAME']
NEO4J_PASSWORD = st.secrets['neo4j_credentials']['NEO4J_PASSWORD']

class BMGAgent:
    """生物质气化数据库查询代理"""

    def __init__(self, session_id: str = "default", api_config: Dict[str, Any] = None, file_path: str = os.getcwd()):
        """
        初始化BMGAgent

        Args:
            session_id: 会话ID，用于历史记录
            api_config: API配置字典，包含api_key, base_url, model_name
        """
        self.session_id = session_id
        self.file_path = file_path
        self.ensure_history_dirs()
        # 默认API配置
        self.api_config = api_config or {
            "api_key": "",
            "base_url": "https://openrouter.ai/api/v1",
            "model_name": "deepseek/deepseek-v3.2"
        }

        # 初始化日志
        self.logger = self._create_logger(session_id)

        # 初始化Neo4j连接
        self.graph = self.init_neo4j()
        self.schema = self.graph.schema if self.graph else "No schema available"

        # 定义prompt
        self.define_prompts()

        # 初始化LLM
        self.init_llms()

        # 构建链
        self.build_chains()

        self._history = FileChatMessageHistory(
            f"./logs/chat_history/{self.session_id}.json"
        )
    def get_history(self, _: dict) -> FileChatMessageHistory:
        """
        LangChain 在每次调用时会把一个“configurable”字典传进来。
        我们根本不需要用它，只返回同一个实例即可。
        """
        return self._history

    def ensure_history_dirs(self):
        """确保历史记录目录存在"""
        os.makedirs("./logs/chat_log", exist_ok=True)
        os.makedirs("./logs/chat_history", exist_ok=True)
        excel_path = os.path.join(self.file_path, "keep_data/excel_data")
        graph_path = os.path.join(self.file_path, "keep_data/graph")
        os.makedirs(excel_path, exist_ok=True)
        os.makedirs(graph_path, exist_ok=True)
    def _create_logger(self, session_id: str) -> logging.Logger:
        # 清除之前的日志处理器
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    os.path.join("./logs/chat_log", f"{session_id}.txt"),
                    mode='a',
                    encoding='utf-8'
                ),
            ],
            force=True
        )

        return logging.getLogger()

    def init_neo4j(self):
        """初始化Neo4j连接"""
        try:
            graph = Neo4jGraph(
                url=NEO4J_URI,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                database=NEO4J_USERNAME,
                enhanced_schema=True
            )
            self.logger.info("Neo4j connection established successfully")
            return graph
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            return None

    def define_prompts(self):
        """定义所有prompt"""
        self.CYPHER_SYSTEM_PROMPT = CYPHER_SYSTEM_PROMPT
        self.OUTPUT_PROMPT= OUTPUT_PROMPT
        self.ANSWER_SYSTEM_PROMPT = ANSWER_SYSTEM_PROMPT
        self.PROPERTIES_DESCRIPTION = PROPERTIES_DESCRIPTION
        self.FEWSHOT_EXAMPLES = FEWSHOT_EXAMPLES

    def init_llms(self):
        """初始化LLM"""
        llm_kwargs = {
            "model": self.api_config["model_name"],
            "temperature": 0,
            "api_key": self.api_config["api_key"],
            "base_url": self.api_config["base_url"]
        }
        self.cypher_llm = ChatOpenAI(**llm_kwargs)
        self.answer_llm = ChatOpenAI(**llm_kwargs)
        self.json_parser = JsonOutputParser()

    def build_chains(self):
        """构建LangChain链"""

        # Cypher查询链
        cypher_agent_prompt = ChatPromptTemplate.from_messages([
            ("system", self.CYPHER_SYSTEM_PROMPT),
            ("system", self.FEWSHOT_EXAMPLES),
            ("system", self.OUTPUT_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("system", self.schema),
            ("system", self.PROPERTIES_DESCRIPTION),
            ("human", "{question}"),
        ])

        self.cypher_chain = (
                {"question": itemgetter("question"), "chat_history": itemgetter("chat_history")}
                | cypher_agent_prompt
                | self.cypher_llm
        )

        self.cypher_conversation = RunnableWithMessageHistory(
            self.cypher_chain,
            self.get_history,
            input_messages_key="question",
            history_messages_key="chat_history"
        )

        # 答案生成链
        answer_agent_prompt = ChatPromptTemplate.from_messages([
            ("system", self.ANSWER_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("system", self.schema),
            ("human", "{question}"),
        ])

        self.answer_chain = (
                {"chat_history": itemgetter("chat_history"), "question": itemgetter("question")}
                | answer_agent_prompt
                | self.answer_llm
        )

        self.answer_conversation = RunnableWithMessageHistory(
            self.answer_chain,
            self.get_history,
            input_messages_key="question",
            history_messages_key="chat_history"
        )

    def convert_query_result(self, result, result_type="json", for_display=False):
        """
        转换查询结果

        Args:
            result: 查询结果
            result_type: 返回类型，"json", "df"或"md"
            for_display: 是否用于 Streamlit 显示（默认为 False）

        Returns:
            转换后的结果
        """
        if result_type == "json":
            return result
        elif result_type == "df":
            df = pd.json_normalize(result)

            if for_display:
                # 用于 Streamlit 显示：解决 Arrow 兼容性问题
                for col in df.columns:
                    if df[col].dtype == 'object':
                        # 更严格的检查：检查是否有任何非字符串/数值类型
                        def check_value_type(x):
                            if pd.isnull(x):
                                return 'null'
                            elif isinstance(x, (int, float, bool)):
                                return 'numeric'
                            elif isinstance(x, str):
                                return 'str'
                            else:
                                # bytes, list, dict, tuple 等
                                return 'complex'

                        # 检查前20个非空值
                        sample_values = df[col].dropna().head(20)
                        if len(sample_values) > 0:
                            value_types = set(check_value_type(x) for x in sample_values)

                            # 如果有复杂类型或混合类型，统一转换为字符串
                            if 'complex' in value_types or len(value_types) > 1:
                                df[col] = df[col].apply(
                                    lambda x: str(x) if pd.notnull(x) else x
                                )
            else:
                # 用于保存或其他用途：保持原始类型
                # 只进行必要的安全转换
                def safe_convert(x):
                    if x is None:
                        return None
                    elif isinstance(x, bytes):
                        return str(x)  # bytes 必须转换
                    elif isinstance(x, (list, dict)):
                        return str(x)  # 复杂结构转换为字符串
                    else:
                        return x

                df = df.map(safe_convert)

            return df
        elif result_type == "md":
            df = self.convert_query_result(result, result_type="df")
            return df.to_markdown(index=False)

    def summarize_dataframe(self, result: pd.DataFrame) -> str:
        """
        总结DataFrame结果

        Args:
            result: DataFrame结果

        Returns:
            总结字符串
        """
        if isinstance(result, pd.DataFrame):
            summary = f"Size: {result.shape}\n"
            summary += f"Columns: {result.columns.tolist()}\n"
            summary += f"Data types: {result.dtypes.to_dict()}\n"
            sample_size = min(20, len(result))
            summary += f"Sample rows: {result.sample(sample_size).to_markdown() if sample_size > 0 else 'No rows available'}\n"
            return summary
        else:
            return str(result)[:20000]

    def get_data(self, cypher_query, for_display=False) -> pd.DataFrame:
        """
        获取数据

        Args:
            cypher_query: Cypher 查询语句
            for_display: 是否用于 Streamlit 显示

        Returns:
            DataFrame 数据
        """
        data = self.graph.query(cypher_query)
        convert_data = self.convert_query_result(data, result_type="df", for_display=for_display)
        return convert_data

    def task_execution(self, question: str) -> Generator:
        """
        执行任务，处理用户问题

        Args:
            question: 用户问题

        Yields:
            响应内容
        """
        try:
            # 记录用户问题
            self.logger.info(f"User: {question}")
            convert_data = None
            try:
                # 获取Cypher查询
                cypher_response = self.cypher_conversation.invoke(
                    {"question": question},
                    config={"configurable": {"session_id": self.session_id}}
                )

            except Exception as e:
                cypher_response = None
                error_msg = f"Cypher查询模型响应错误: {e}\n"
                self.logger.error(error_msg)
                yield error_msg
                # 解析响应

            if cypher_response:
                cypher_result = cypher_response.content
                self.logger.info(f"Cypher Response: {cypher_result}")
                if 'thought_process' in cypher_result or 'cypher_query' in cypher_result:
                    try:
                        cypher_result = json.loads(cypher_result)

                        if 'thought_process' in cypher_result and cypher_result.get("thought_process"):
                            if 'cypher_query' in cypher_result and cypher_result.get("cypher_query"):
                                yield cypher_result
                            else:
                                cypher_result = cypher_response.content
                                cypher_result = "响应内容中的cypher_query处出错或你的要求不合理：\n" + cypher_result + "\n请重新提问。\n"
                                yield cypher_result
                        else:
                            cypher_result = cypher_response.content
                            cypher_result = "响应内容中的thought_process处出错：\n" + cypher_result + "\n请重新提问。\n"
                            yield cypher_result

                    except:
                        cypher_result = cypher_response.content
                        cypher_result = "响应格式生成出错：\n" + cypher_result + "\n请重新提问。\n"
                        yield cypher_result

                else:
                    yield cypher_result

                # 检查是否需要Cypher查询
                if isinstance(cypher_result, dict):
                    cypher_query = cypher_result['cypher_query']

                    if cypher_query:
                        msg = "查询数据中..."
                        self.logger.info(msg)
                        self._history.add_ai_message(msg)
                        yield f"{msg}\n"
                        try:
                            convert_data = self.get_data(cypher_query, for_display=True)

                            if convert_data is not None and not convert_data.empty:
                                msg = "查询数据如下（可保存所有数据 - 可展示数据分析图）"
                                self.logger.info(msg)
                                self._history.add_ai_message(msg)
                                yield f"{msg}\n"
                                yield convert_data
                            else:
                                msg = "查询成功，但没有找到匹配的数据。"
                                self._history.add_ai_message(msg)
                                self.logger.warning(msg)
                                yield f"{msg}\n"
                        except Exception as e:
                            error_msg = f"获取数据环节出错: {e}\n"
                            self.logger.error(error_msg)
                            yield error_msg
                    else:
                        msg = "需要Cypher查询，但没有成功生成Cypher查询，请重试。"
                        self._history.add_ai_message(msg)
                        self.logger.warning(msg)
                        yield "{msg}\n"

                if convert_data is not None and not convert_data.empty:
                    # 有查询结果，生成总结
                    try:
                        summary_result = self.summarize_dataframe(convert_data)

                        third_question = "请根据'summary_result'进行总结。\n" + "summary_result：\n" + summary_result
                        # 获取答案
                        try:
                            answer_response = self.answer_conversation.invoke(
                                    {"question": third_question},
                                    config={"configurable": {"session_id": self.session_id}})

                            answer_result = answer_response.content
                            self.logger.info(f"Answer: {answer_result}")
                            yield answer_result
                        except Exception as e:
                            error_msg = f"答案模型响应错误: {e}\n"
                            self.logger.error(error_msg)
                            yield error_msg
                    except Exception as e:
                        error_msg = f"数据总结出现错误: {e}\n"
                        self.logger.error(error_msg)
                        yield error_msg

        except Exception as e:
            error_msg = f"任务执行过程中出现错误: {e}\n"
            self.logger.error(f"Task execution error: {e}")
            yield error_msg
