import random
from datetime import datetime
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
import streamlit as st
from neo4j_viz import Node, Relationship, VisualizationGraph
import streamlit.components.v1 as components
import html
from BMGPrompt import (
    CYPHER_SYSTEM_PROMPT,
    OUTPUT_PROMPT,
    ANSWER_SYSTEM_PROMPT,
    PROPERTIES_DESCRIPTION,
    FEWSHOT_EXAMPLES)
from Support_function import HistoryManager
from openai import BadRequestError
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =================================================================================================== #
### 辅助函数 ###
# =================================================================================================== #

@st.cache_resource
def get_history_manager(chat_history_dir):
    """获取历史管理器单例"""
    return HistoryManager(chat_history_dir)

def create_logger(loaded_conversation_id: str) -> logging.Logger:
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
                os.path.join(f"{st.session_state.file_path}/records/log", f"{loaded_conversation_id}.txt"),
                mode='a',
                encoding='utf-8'
            ),
        ],
        force=True
    )
    return logging.getLogger()

def sanitize_markdown(content):
    """
    安全渲染Markdown内容
    """
    if content is None:
        return ""

    content = str(content)

    # 处理代码块
    code_pattern = r'(```(?:cypher|sql|python|json|)?\s*[\s\S]*?```|~~~\s*[\s\S]*?~~~)'
    parts = re.split(code_pattern, content)

    result_parts = []
    for i, part in enumerate(parts):
        is_code_block = i % 2 == 1
        if is_code_block:
            result_parts.append(part)
        else:
            # 检查是否是Cypher查询
            if re.search(r'^\s*(MATCH|RETURN|WHERE|CREATE|DELETE|MERGE)\b', part, re.IGNORECASE | re.MULTILINE):
                wrapped_part = f"```cypher\n{part.strip()}\n```"
                result_parts.append(wrapped_part)
            else:
                # 清理非代码内容
                cleaned = re.sub(r'^:::.+', '', part, flags=re.MULTILINE)
                cleaned = cleaned.replace(':::', '')
                cleaned = re.sub(r'\{#.*?\}', '', cleaned)

                # 只转义HTML标签（防止XSS），保留Markdown语法
                import html
                cleaned = html.escape(cleaned)

                result_parts.append(cleaned)

    return ''.join(result_parts)

def render_content(content, placeholder=None):
    try:
        sanitized = sanitize_markdown(content)
        if placeholder:
            placeholder.markdown(sanitized)
        else:
            st.markdown(sanitized)
    except Exception as e:
        st.error(f"Markdown渲染错误: {e}")
        if placeholder:
            placeholder.text(content)
        else:
            st.text(content)

def safe_chat(role_type, content):
    with st.chat_message(role_type):
        render_content(content, st.empty())

# ========== 对话预览对话框 ==========
@st.dialog("对话预览")
def show_conversation_preview(conversation: Dict[str, Any]):
    """显示对话预览"""
    st.markdown(f"**标题**: {conversation['title']}")
    st.markdown(f"**ID**: {conversation['id']}")
    st.markdown(f"**时间**: {conversation['timestamp']}")
    st.markdown(f"**消息数量**: {conversation['message_count']}")

    st.markdown("---")
    st.markdown("### 消息预览")

    # 显示前几条消息
    preview_count = min(5, len(conversation.get("messages", [])))

    for i, msg in enumerate(conversation.get("messages", [])[:preview_count]):
        msg_type = msg.get("type", "")
        data = msg.get("data", {})
        content = data.get("content", "")

        # 简化显示
        if msg_type == "human" or data.get("type") == "human":
            st.markdown(f"**用户**: {content[:100]}...")
        else:
            # 尝试解析AI响应
            if isinstance(content, str) and content.strip().startswith("{"):
                try:
                    content_dict = json.loads(content)
                    thought = content_dict.get("thought_process", "")
                    if thought:
                        st.markdown(f"**助手**: {thought[:100]}...")
                    else:
                        st.markdown(f"**助手**: {content[:100]}...")
                except:
                    st.markdown(f"**助手**: {content[:100]}...")
            else:
                st.markdown(f"**助手**: {content[:100]}...")

    if len(conversation.get("messages", [])) > preview_count:
        st.info(f"... 还有 {len(conversation['messages']) - preview_count} 条消息")

def clean_bmg_data(df, selected_columns):
    """
    清洗BMG数据
    """
    # 只保留选中列
    cleaned_df = df[selected_columns].copy()
    cleaned_df = cleaned_df.dropna()

    if cleaned_df.empty:
        return cleaned_df

    # 定义列组
    cho_group = ['C', 'H', 'O']
    afv_group = ['ash', 'fc', 'volatile']
    hccc_group = ['H2', 'CO', 'CO2', 'CH4']

    if 'T' in selected_columns:
        cleaned_df['T'] = cleaned_df['T'].apply(convert_temp)

    # 首先对所有选中的列应用extract_numeric，确保它们是数值类型
    for col in selected_columns:
        cleaned_df[col] = cleaned_df[col].apply(extract_numeric)

    # 再次删除NaN（转换后可能产生新的NaN）
    cleaned_df = cleaned_df.dropna()

    if cleaned_df.empty:
        return cleaned_df

    # 比例归一化处理
    for group in [cho_group, afv_group, hccc_group]:
        if all(col in selected_columns for col in group):
            group_df = cleaned_df[group].dropna()
            if not group_df.empty:
                group_sum = group_df.sum(axis=1).replace(0, np.nan)
                for col in group:
                    cleaned_df.loc[group_df.index, col] = group_df[col] / group_sum

    # 对 ER、T 和 Agent_biomass_ratio 进行列归一化
    for col in ['ER', 'T', 'Agent_biomass_ratio']:
        if col in selected_columns and col in cleaned_df.columns:
            cleaned_df[col] = normalize_column(cleaned_df[col])

    return cleaned_df.dropna()

def standardize_bmg_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    将查询结果中的数据库属性名转换为主应用清洗函数使用的列名。
    仅转换明确列出的名称；
    不修改原始 DataFrame。
    """
    column_aliases = {
        # 元素分析
        "C(wt_%)": "C",
        "H(wt_%)": "H",
        "O(wt_%)": "O",
        # 工业分析
        "Ash(wt_%)": "ash",
        "Fixed_carbon(wt_%)": "fc",
        "Volatile_matter(wt_%)": "volatile",
        # 反应器参数
        "Temperature(K)": "T",
        "Equivalence_ratio": "ER",
        "Agent_to_biomass_ratio": "Agent_biomass_ratio",
    }
    result = df.copy()
    # 相同列名会使后续 df[col] 返回 DataFrame，而不是 Series
    if result.columns.duplicated().any():
        duplicate_names = result.columns[
            result.columns.duplicated()
        ].tolist()
        raise ValueError(
            f"查询结果包含重复列名：{duplicate_names}。"
            "请在 Cypher 中使用不同的 AS 别名。"
        )
    rename_map = {}
    for source_name, target_name in column_aliases.items():
        if source_name not in result.columns:
            continue
        # 防止同时有 C 和 C(wt_%) 时转换出两个 C 列
        if target_name in result.columns:
            raise ValueError(
                f"查询结果同时包含 '{source_name}' 和 '{target_name}'，"
                "无法确定应该使用哪一列。"
                "请在 Cypher 中通过 AS 指定唯一列名。"
            )
        rename_map[source_name] = target_name
    return result.rename(columns=rename_map)

def normalize_column(series):
    """对单列进行 min-max 归一化，常数列设为 0。"""
    series = pd.to_numeric(series, errors="coerce")
    valid_series = series.dropna()
    if valid_series.empty:
        return series
    min_val = valid_series.min()
    max_val = valid_series.max()
    # 常数列：非空值设为 0，空值仍保留 NaN
    if max_val == min_val:
        return series.where(series.isna(), 0.0)
    return (series - min_val) / (max_val - min_val)

def convert_temp(temp, default_unit="K"):
    try:
        value = extract_numeric(temp)
        if pd.isna(value):
            return np.nan
        temp_str = str(temp).upper().strip()
        if "K" in temp_str:
            return value
        if "C" in temp_str:
            return value + 273.15
        # 无单位时按约定处理
        if default_unit == "C":
            return value + 273.15
        if default_unit == "K":
            return value
        return np.nan
    except Exception:
        return np.nan

def extract_numeric(x):
    """提取数值（独立函数，供其他函数调用）"""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return np.nan if pd.isna(x) else float(x)

    x_str = str(x).strip()
    numbers = re.findall(r'[-+]?\d*\.?\d+', x_str)
    return float(numbers[0]) if numbers else np.nan

def generate_visualizations(cleaned_df, output_dir=None):
    """
    生成小提琴图和相关系数热图。

    返回：
        (violin_figure, heatmap_figure)
    """

    if cleaned_df is None or cleaned_df.empty:
        st.warning("没有可用于绘图的数据。")
        return None, None

    numeric_df = cleaned_df.select_dtypes(include=[np.number]).copy()

    if numeric_df.empty:
        st.warning("数据中没有数值列，无法生成图表。")
        return None, None

    fig1 = None
    fig2 = None

    # --------------------------------------------------
    # 1. 小提琴图
    # --------------------------------------------------
    violin_columns = []
    skipped_columns = []

    for col in numeric_df.columns:
        values = numeric_df[col].dropna()

        # 小提琴图需要至少两个有效值，且数据存在变化
        if len(values) >= 2 and values.nunique() >= 2:
            violin_columns.append(col)
        else:
            skipped_columns.append(col)

    if violin_columns:
        try:
            fig1, ax1 = plt.subplots(figsize=(12, 6))

            violin_data = [
                numeric_df[col].dropna().to_numpy()
                for col in violin_columns
            ]

            ax1.violinplot(
                violin_data,
                showmeans=True,
                showmedians=True,
            )

            ax1.set_xticks(range(1, len(violin_columns) + 1))
            ax1.set_xticklabels(
                violin_columns,
                rotation=45,
                ha="right",
            )
            ax1.set_title("Violin Plot")
            ax1.set_ylabel("Value")

            fig1.tight_layout()

        except (ValueError, np.linalg.LinAlgError) as e:
            if fig1 is not None:
                plt.close(fig1)

            fig1 = None
            st.warning(f"小提琴图生成失败：{e}")
    else:
        st.info(
            "没有足够变化的数据生成小提琴图。"
            "每个绘图列至少需要两个不同的有效数值。"
        )

    if skipped_columns:
        st.caption(
            "以下列因有效数据不足或数值相同，"
            f"未加入小提琴图：{', '.join(skipped_columns)}"
        )

    # --------------------------------------------------
    # 2. 相关系数热图
    # --------------------------------------------------
    if len(numeric_df) >= 2 and len(numeric_df.columns) >= 2:
        correlation_matrix = numeric_df.corr()

        # 常数列或数据不足时可能产生 NaN
        if not correlation_matrix.isna().all().all():
            fig2, ax2 = plt.subplots(figsize=(10, 8))

            sns.heatmap(
                correlation_matrix,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                vmin=-1,
                vmax=1,
                mask=correlation_matrix.isna(),
                ax=ax2,
            )

            ax2.set_title("Correlation Heatmap")
            fig2.tight_layout()
        else:
            st.info("没有足够的有效数据生成相关系数热图。")
    else:
        st.info("相关系数热图至少需要两行数据和两个数值列。")

    return fig1, fig2

@st.dialog("数据可视化")
def show_visualization_dialog(cleaned_data, excel_path=None, conversation_id=None, idx=None):
    """
    显示数据可视化图表对话框
    """

    st.markdown("""
    <style>
    /* 白色按钮样式 */
    .stButton button, .stDownloadButton button {
        width: 100%;
        background-color: white;
        color: #333333;
        border: 1px solid #cccccc;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-size: 1rem;
    }
    .stButton button:hover, .stDownloadButton button:hover {
        background-color: #f0f0f0;
        border-color: #999999;
    }
    </style>
    """, unsafe_allow_html=True)

    if cleaned_data.empty:
        st.error("没有可显示的数据")
        return

    # 生成可视化
    with st.spinner("生成可视化图表..."):
        fig1, fig2 = generate_visualizations(cleaned_data)

    if fig1:
        st.markdown("### 小提琴图")
        st.pyplot(fig1)

        col1, col2 = st.columns(2)
        with col1:
            if excel_path and conversation_id and idx is not None:
                if st.button("保存小提琴图", type="secondary"):
                    fig1.savefig(f"{excel_path}/{conversation_id}_{idx}_violin.png",
                                 dpi=300, bbox_inches='tight')
                    st.success(f"小提琴图已保存: {excel_path}/{conversation_id}_{idx}_violin.png")

        buf1 = BytesIO()
        fig1.savefig(buf1, format='png', dpi=300, bbox_inches='tight')
        buf1.seek(0)  # 将指针移到开头

        with col2:
            # 提供下载按钮
            st.download_button(
                label="📥 下载小提琴图",
                data=buf1,
                file_name=f"{conversation_id}_{idx}_violin.png",
                mime="image/png"
            )

    if fig2:
        st.markdown("### 相关系数热图")
        st.pyplot(fig2)

        col1, col2 = st.columns(2)
        with col1:
            if excel_path and conversation_id and idx is not None:
                if st.button("保存热图", type="secondary"):
                    fig2.savefig(f"{excel_path}/{conversation_id}_{idx}_heatmap.png",
                                 dpi=300, bbox_inches='tight')
                    st.success(f"热图已保存: {excel_path}/{conversation_id}_{idx}_heatmap.png")

        buf2 = BytesIO()
        fig2.savefig(buf2, format='png', dpi=300, bbox_inches='tight')
        buf2.seek(0)

        with col2:
            st.download_button(
                label="📥 下载热图",
                data=buf2,
                file_name=f"{conversation_id}_{idx}_heatmap.png",
                mime="image/png"
            )

@st.dialog("数据清洗和分析")
def show_data_cleaning_dialog(raw_data, conversation_id, idx):
    """
    数据清洗对话框
    """
    try:
        raw_data = standardize_bmg_columns(raw_data)
    except ValueError as e:
        st.error(f"查询结果列名不符合清洗要求：{e}")
        return

    st.markdown("---")
    st.title("清洗规则")
    st.markdown("""1. **分组选择**：用户可以按组选择要分析的列
    - C/H/O 组（元素组成）
    - ash/fc/volatile 组（工业分析）
    - H2/CO/CO2/CH4 组（产物气体）
2. **自动归一化**：对操作参数（ER、T、Agent_biomass_ratio）自动进行 **min-max 归一化**（缩放到0-1范围）
3. **删除空值**：清洗过程中会删除选中列中**任何一行存在空值**的行（完整行删除）\n
**注意**：Agent_biomass_ratio对应不同气化剂，请注意区分。\n
**有特殊需求可以保存清洗前数据，并手动清洗。**""")
    st.markdown("---")

    # 感兴趣的列 - 注意修正列名
    interest_columns = ['C', 'H', 'O', 'ash', 'fc', 'volatile',
                        'T', 'ER', 'Agent_biomass_ratio',
                        'H2', 'CO', 'CO2', 'CH4']

    available_columns = [col for col in interest_columns if col in raw_data.columns]

    # 按组检查
    cho_avail = all(col in available_columns for col in ['C', 'H', 'O'])
    afv_avail = all(col in available_columns for col in ['ash', 'fc', 'volatile'])
    hccc_avail = all(col in available_columns for col in ['H2', 'CO', 'CO2', 'CH4'])

    st.markdown("### 选择要分析的列（按组选择）")

    # 创建复选框
    if cho_avail:
        c_selected = st.checkbox("C/H/O组", value=True)
    else:
        c_selected = False
        st.warning("C/H/O组数据不完整")

    if afv_avail:
        a_selected = st.checkbox("ash/fc/volatile组", value=True)
    else:
        a_selected = False
        st.warning("ash/fc/volatile组数据不完整")

    if hccc_avail:
        h_selected = st.checkbox("H2/CO/CO2/CH4组", value=True)
    else:
        h_selected = False
        st.warning("产物气体组数据不完整")

    # 其他列
    other_cols = ['T', 'ER', 'Agent_biomass_ratio']
    other_selected = {}
    for col in other_cols:
        if col in available_columns:
            other_selected[col] = st.checkbox(col, value=True)

    # 构建选中列列表
    selected = []
    if c_selected and cho_avail:
        selected.extend(['C', 'H', 'O'])
    if a_selected and afv_avail:
        selected.extend(['ash', 'fc', 'volatile'])
    if h_selected and hccc_avail:
        selected.extend(['H2', 'CO', 'CO2', 'CH4'])

    for col in other_cols:
        if col in available_columns and other_selected.get(col, False):
            selected.append(col)

    if not selected:
        st.error("没有可分析的列")
        return

    if st.button("开始清洗和分析", type="primary"):
        with st.spinner("清洗数据中..."):
            cleaned_data = clean_bmg_data(raw_data, selected)

            if cleaned_data.empty:
                st.error("清洗后无有效数据，请确定每一行都有值")
                return

            st.success(f"清洗完成，有效数据: {len(cleaned_data)} 行")

            # 显示归一化说明
            normalized_cols = [col for col in ['ER', 'T', 'Agent_biomass_ratio']
                               if col in selected]
            if normalized_cols:
                st.info(f"已对以下列进行min-max归一化: {', '.join(normalized_cols)}")
                # 显示归一化后的统计信息
                for col in normalized_cols:
                    if col in cleaned_data.columns:
                        st.write(
                            f"**{col}**: 最小值={cleaned_data[col].min():.3f}, 最大值={cleaned_data[col].max():.3f}, 平均值={cleaned_data[col].mean():.3f}")

            # 显示清洗后数据
            st.markdown("### 清洗后数据")
            st.dataframe(cleaned_data, use_container_width=True)

            excel_path = os.path.join(st.session_state.file_path, "records/graph")
            param = {"data": cleaned_data,
                     "excel_path": excel_path,
                     "conversation_id": conversation_id,
                     "idx": idx}

            st.session_state.viz_cleaned_data_params[f"{st.session_state.loaded_conversation_id}_{idx}"] = param

            st.info("点击返回可以进行查看可视化图表和保存清洗后数据")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("返回", type="primary", use_container_width=True):
            st.rerun()


# =================================================================================================== #
### 连接对话###
# =================================================================================================== #

def get_api_config():
    return {"api_key": st.session_state.api_key,
        "base_url": st.session_state.base_url,
        "model": st.session_state.model,
        "temperature": 0,
        }

def get_new_session_id():
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    random_number = random.randint(1000, 9999)
    st.session_state.session_id = f"session_{current_time}_{random_number}"

def init_llms():
    llm_kwargs = get_api_config()
    st.session_state.llm_kwargs = llm_kwargs
    st.session_state.cypher_llm = ChatOpenAI(**llm_kwargs)
    st.session_state.answer_llm = ChatOpenAI(**llm_kwargs)

def conversation_history(session_id):
    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}
    if session_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[session_id] = FileChatMessageHistory(
            f"{st.session_state.file_path}/records/chat_history/{session_id}.json"
        )
    return st.session_state.chat_histories[session_id]

def build_chains():
    """构建并保存 LangChain 对话链"""
    # Cypher 查询链
    cypher_agent_prompt = ChatPromptTemplate.from_messages([
        ("system", CYPHER_SYSTEM_PROMPT),
        ("system", FEWSHOT_EXAMPLES),
        ("system", OUTPUT_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("system", st.session_state.neo4j_graph.schema),
        ("system", PROPERTIES_DESCRIPTION),
        ("human", "{question}"),
    ])
    cypher_chain = (
        {
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | cypher_agent_prompt
        | st.session_state.cypher_llm
    )

    st.session_state.cypher_conversation = RunnableWithMessageHistory(
        cypher_chain,
        conversation_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    # 答案生成链
    answer_agent_prompt = ChatPromptTemplate.from_messages([
        ("system", ANSWER_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("system", st.session_state.neo4j_graph.schema),
        ("human", "{question}"),
    ])
    answer_chain = (
        {
            "chat_history": itemgetter("chat_history"),
            "question": itemgetter("question"),
        }
        | answer_agent_prompt
        | st.session_state.answer_llm
    )

    st.session_state.answer_conversation = RunnableWithMessageHistory(
        answer_chain,
        conversation_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

def invoke_conversation(conversation_name, inputs, config=None, **kwargs):
    """调用对话链；temperature 不支持时重建并重试一次"""
    allowed_names = ("cypher_conversation", "answer_conversation")
    if conversation_name not in allowed_names:
        raise ValueError(f"未知的对话链名称: {conversation_name}")
    conversation = st.session_state[conversation_name]
    try:
        return conversation.invoke(inputs, config=config, **kwargs)
    except BadRequestError as exc:
        error_message = str(exc).lower()
        temperature_error = (
            "temperature" in error_message
            and any(
                keyword in error_message
                for keyword in ("unsupported", "not support", "invalid", "only")
            )
        )
        llm_kwargs = st.session_state.llm_kwargs
        # 不是 temperature 错误，或已经移除过该参数
        if not temperature_error or "temperature" not in llm_kwargs:
            raise
        # 移除 temperature，重新创建模型
        new_kwargs = dict(llm_kwargs)
        new_kwargs.pop("temperature")
        st.session_state.llm_kwargs = new_kwargs
        st.session_state.cypher_llm = ChatOpenAI(**new_kwargs)
        st.session_state.answer_llm = ChatOpenAI(**new_kwargs)
        # 旧链仍引用旧模型，因此必须重建
        build_chains()
        # 获取重建后的链，重试一次
        conversation = st.session_state[conversation_name]
        return conversation.invoke(inputs, config=config, **kwargs)

def create_new_conversation():
    """处理新建对话"""
    if not (
        st.session_state.model_config
        and st.session_state.neo4j_status
    ):
        st.error("Failed to connect to the Neo4j or AI model")
        return
    # 保存原状态，失败时恢复
    old_session_id = st.session_state.session_id
    old_loaded_id = st.session_state.loaded_conversation_id
    old_messages = st.session_state.messages
    old_logger = st.session_state.logger
    old_history = st.session_state.get("history")
    try:
        get_new_session_id()
        new_id = st.session_state.session_id
        init_llms()
        build_chains()
        new_logger = create_logger(new_id)
        new_history = conversation_history(new_id)
        st.session_state.messages = []
        st.session_state.loaded_conversation_id = new_id
        st.session_state.logger = new_logger
        st.session_state.history = new_history
    except Exception as e:
        st.session_state.session_id = old_session_id
        st.session_state.loaded_conversation_id = old_loaded_id
        st.session_state.messages = old_messages
        st.session_state.logger = old_logger
        if old_history is not None:
            st.session_state.history = old_history
        else:
            st.session_state.pop("history", None)
        st.error(f"❌ Failed to connect to the agent: {e}")
        return
    st.rerun()

def load_conversation(conversation_id: str, history_manager: HistoryManager):
    """
    加载历史对话

    Args:
        conversation_id: 对话ID
        history_manager: 历史管理器
    """
    # 加载LangChain格式的消息
    json_exists, langchain_messages = history_manager.load_conversation(conversation_id)

    if json_exists:
        try:
            init_llms()
            build_chains()

            # 更新session state
            st.session_state.messages = langchain_messages
            st.session_state.session_id = conversation_id
            st.session_state.loaded_conversation_id = conversation_id
            st.session_state.logger = create_logger(st.session_state.loaded_conversation_id)
            st.session_state.history = conversation_history(st.session_state.loaded_conversation_id)
            # st.session_state.conversation_modified = False

            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to connect to the agent: {e}")
    else:
        st.error(f"无法加载对话: {conversation_id}")

# =================================================================================================== #
### 会话响应 ###
# =================================================================================================== #

def convert_query_result(result, result_type="json", for_display=False):
    """
    转换查询结果

    Args:
        result: 查询结果
        result_type: 返回类型，"json", "df"或"md"
        for_display: 是否用于 Streamlit 显示（默认为 False）
    """
    if result_type == "json":
        return result
    elif result_type == "df":
        df = pd.json_normalize(result)

        if for_display:
            def display_safe_value(x):
                # 先处理复杂结构，避免 pd.isna 返回数组
                if isinstance(x, (list, dict, tuple, set, bytes, np.ndarray)):
                    return str(x)
                # 此时再判断标量空值
                try:
                    if pd.isna(x):
                        return None
                except (TypeError, ValueError):
                    return str(x)
                return x

            for col in df.columns:
                if df[col].dtype == "object":
                    df[col] = df[col].map(display_safe_value)
                    non_null = df[col].dropna()
                    # 混合字符串和数字时统一转成字符串，避免 Arrow 类型冲突
                    value_types = {
                        "str" if isinstance(x, str)
                        else "numeric" if isinstance(x, (int, float, bool, np.number))
                        else "other"
                        for x in non_null
                    }
                    if "other" in value_types or len(value_types) > 1:
                        df[col] = df[col].map(
                            lambda x: None if x is None else str(x)
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
        df = convert_query_result(result, result_type="df")
        return df.to_markdown(index=False)

def summarize_dataframe(result: pd.DataFrame) -> str:
    """
    总结DataFrame结果

    Args:
        result: DataFrame结果
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

def get_data(cypher_query, for_display=False) -> pd.DataFrame:
    """
    获取数据

    Args:
        cypher_query: Cypher 查询语句
        for_display: 是否用于 Streamlit 显示

    """
    data = st.session_state.neo4j_graph.query(cypher_query)
    convert_data = convert_query_result(data, result_type="df", for_display=for_display)
    return convert_data

def task_execution(question: str) -> Generator:
    """
    执行任务，处理用户问题

    Args:
        question: 用户问题
    """
    try:
        # 记录用户问题
        st.session_state.logger.info(f"User: {question}")
        convert_data = None
        try:
            # 获取Cypher查询
            cypher_response = invoke_conversation(
                "cypher_conversation",
                {"question": question},
                config={"configurable": {"session_id": st.session_state.loaded_conversation_id}}
            )

        except Exception as e:
            cypher_response = None
            error_msg = f"Cypher查询模型响应错误: {e}\n"
            st.session_state.logger.error(error_msg)
            yield error_msg
            # 解析响应

        if cypher_response:
            cypher_result = cypher_response.content
            st.session_state.logger.info(f"Cypher Response: {cypher_result}")
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
                    st.session_state.logger.info(msg)
                    st.session_state.history.add_ai_message(msg)
                    yield f"{msg}\n"
                    try:
                        convert_data = get_data(cypher_query, for_display=True)

                        if convert_data is not None and not convert_data.empty:
                            msg = "查询数据如下（可保存所有数据 - 可展示数据分析图）"
                            st.session_state.logger.info(msg)
                            st.session_state.history.add_ai_message(msg)
                            yield f"{msg}\n"
                            yield convert_data
                        else:
                            msg = "查询成功，但没有找到匹配的数据。"
                            st.session_state.history.add_ai_message(msg)
                            st.session_state.logger.warning(msg)
                            yield f"{msg}\n"
                    except Exception as e:
                        error_msg = f"获取数据环节出错: {e}\n"
                        st.session_state.logger.error(error_msg)
                        yield error_msg
                else:
                    msg = "需要Cypher查询，但没有成功生成Cypher查询，请重试。"
                    st.session_state.history.add_ai_message(msg)
                    st.session_state.logger.warning(msg)
                    yield "{msg}\n"

            if convert_data is not None and not convert_data.empty:
                # 有查询结果，生成总结
                try:
                    summary_result = summarize_dataframe(convert_data)

                    third_question = "请根据'summary_result'进行总结。\n" + "summary_result：\n" + summary_result
                    # 获取答案
                    try:
                        answer_response =invoke_conversation(
                                "answer_conversation",
                                {"question": third_question},
                                config={"configurable": {"session_id": st.session_state.loaded_conversation_id}})

                        answer_result = answer_response.content
                        st.session_state.logger.info(f"Answer: {answer_result}")
                        yield answer_result
                    except Exception as e:
                        error_msg = f"答案模型响应错误: {e}\n"
                        st.session_state.logger.error(error_msg)
                        yield error_msg
                except Exception as e:
                    error_msg = f"数据总结出现错误: {e}\n"
                    st.session_state.logger.error(error_msg)
                    yield error_msg

    except Exception as e:
        error_msg = f"任务执行过程中出现错误: {e}\n"
        st.session_state.logger.error(f"Task execution error: {e}")
        yield error_msg

def process_user_input(user_input: str, conversation_id: str, history_manager: HistoryManager):
    """处理用户输入并生成响应"""

    # st.session_state.conversation_modified = True
    cypher_records = {}

    # 获取agent响应
    try:
        agent_generator = task_execution(user_input)

        # 处理流式响应
        for chunk in agent_generator:
            if isinstance(chunk, dict):
                if "thought_process" in chunk:
                    with st.expander("🤔 推理过程", expanded=False):
                        render_content(chunk.get("thought_process", ""))

                if "cypher_query" in chunk:
                    with st.expander("🔍 Cypher查询", expanded=False):
                        st.code(chunk.get("cypher_query", ""), language="cypher")
                        json_exists, st.session_state.messages = history_manager.load_conversation(st.session_state.loaded_conversation_id)
                        if not json_exists:
                            st.error("当前状态文件不存在，请检查")
                        else:
                            idx = len(st.session_state.messages)-1
                            cypher_records[idx] = chunk.get("cypher_query", "")

            elif isinstance(chunk, pd.DataFrame):
                st.dataframe(chunk, use_container_width=True)
                excel_path = os.path.join(st.session_state.file_path, "records/table")
                keep_idx = f"{st.session_state.loaded_conversation_id}_{idx}"

                if "LIMIT 20" in cypher_records[idx] and len(chunk) >= 20:
                    if st.button("搜索并保存所有数据", key=f"cypher_present_{idx}", use_container_width=True):
                        try:
                            cypher_records_all = cypher_records[idx].replace("LIMIT 20", "")
                            data = get_data(cypher_records_all, for_display=False)

                            if data is not None and not data.empty:
                                data.to_excel(
                                    f"{excel_path}/{keep_idx}.xlsx",
                                    index=False)
                                st.success(
                                    f"数据已保存：{excel_path}/{keep_idx}.xlsx")

                                param = {"data": data,
                                         "conversation_id": st.session_state.loaded_conversation_id,
                                         "idx": idx}
                                st.session_state.cleaning_dialog_params[keep_idx] = param

                                st.rerun()
                            else:
                                st.info("没有查询到相关数据")
                        except Exception as e:
                            st.warning(f"获取数据时出错：{e}")
                else:
                    st.info("已展示全部数据")
                    if st.button("保存数据", key=f"cypher_present_{idx}", use_container_width=True):
                        chunk.to_excel(
                            f"{excel_path}/{keep_idx}.xlsx",
                            index=False)
                        st.success(
                            f"数据已保存：{excel_path}/{keep_idx}.xlsx")

                        param = {"data": chunk,
                                 "conversation_id": st.session_state.loaded_conversation_id,
                                 "idx": idx}
                        st.session_state.cleaning_dialog_params[keep_idx] = param

                        st.rerun()

            else:
                safe_chat(role_type="ai", content=chunk)

        json_exists, st.session_state.messages = history_manager.load_conversation(conversation_id)
        if not json_exists:
            st.error("当前状态文件不存在，请检查")

    except Exception as e:
        error_msg = f"处理请求时出错: {e}"
        safe_chat(role_type="ai", content=error_msg)

# =================================================================================================== #
### 主程序 ###
# =================================================================================================== #
### Neo4j database connection
# 获取neo4j数据库字段，详细设置见".streamlit/secrets.toml"
NEO4J_URI = st.secrets['neo4j_credentials']['NEO4J_URI']
NEO4J_USERNAME = st.secrets['neo4j_credentials']['NEO4J_USERNAME']
NEO4J_PASSWORD = st.secrets['neo4j_credentials']['NEO4J_PASSWORD']

# Check the Neo4j database connection status
if "neo4j_status" not in st.session_state:
    st.session_state.neo4j_status = False

# Neo4j Database Messages
if "neo4j_messages" not in st.session_state:
    st.session_state.neo4j_messages = ""

# 尝试连接Neo4j数据库
try:
    graph = Neo4jGraph(
                    url=NEO4J_URI,
                    username=NEO4J_USERNAME,
                    password=NEO4J_PASSWORD,
                    database=NEO4J_USERNAME,
                    enhanced_schema=True
        )
    st.session_state.neo4j_status = True
    st.session_state.neo4j_messages = "✅ Neo4j connection established successfully"
    if "neo4j_graph" not in st.session_state:
        st.session_state.neo4j_graph = graph
except Exception as e:
    st.session_state.neo4j_status = False
    st.session_state.neo4j_messages = f"❌ Failed to connect to Neo4j: {e}"

### Settings Parameters
# API Settings for AI Models
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# URL Settings for AI Models
if "base_url" not in st.session_state:
    st.session_state.base_url = ""

# AI Model Settings
if "model" not in st.session_state:
    st.session_state.model = ""

# Model Settings Status
if "model_config" not in st.session_state:
    st.session_state.model_config = False

# Path to save the file
if "file_path" not in st.session_state:
    st.session_state.file_path = os.getcwd()

### Agent Status Parameters
# Creating New Dialogue Title
if "session_id" not in st.session_state:
    st.session_state.session_id = None
# Data Storage
if "messages" not in st.session_state:
    st.session_state.messages = []

# Current dialogue title
if "loaded_conversation_id" not in st.session_state:
    st.session_state.loaded_conversation_id = None

# if "conversation_modified" not in st.session_state:
#     st.session_state.conversation_modified = False

if "cypher_llm" not in st.session_state:
    st.session_state.cypher_llm = None

if "answer_llm" not in st.session_state:
    st.session_state.answer_llm = None

if "llm_kwargs" not in st.session_state:
    st.session_state.llm_kwargs = None

if "logger" not in st.session_state:
    st.session_state.logger = None

if "cypher_conversation" not in st.session_state:
    st.session_state.cypher_conversation = None

if "answer_conversation" not in st.session_state:
    st.session_state.answer_conversation = None

if "graph_visibility" not in st.session_state:
    st.session_state.graph_visibility = {}

if "show_cleaning_dialog" not in st.session_state:
    st.session_state.show_cleaning_dialog = False

if "cleaning_dialog_params" not in st.session_state:
    st.session_state.cleaning_dialog_params = {}

if "dialog_triggered" not in st.session_state:
    st.session_state.dialog_triggered = None

if "cleaning_or_viz" not in st.session_state:
    st.session_state.cleaning_or_viz = None

if "viz_cleaned_data_params" not in st.session_state:
    st.session_state.viz_cleaned_data_params = {}

# 初始化历史管理器
history_manager = get_history_manager(
    os.path.join(st.session_state.file_path, "records", "chat_history")
                                      )
# ====================================================================================================== #

st.set_page_config(
    page_title="BMG Agent",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌱 Biomass Gasification Database Interactive Agent")

left_col, _, right_col = st.columns([1, 2, 1])

with left_col:
    st.markdown(f"Corrrent Conversation ID: {st.session_state.loaded_conversation_id}" if st.session_state.loaded_conversation_id else "Corrrent Conversation ID: None")

st.markdown("---")

# 提示数据库连接状态
if st.session_state.neo4j_status:
    st.sidebar.info(st.session_state.neo4j_messages)
else:
    st.warning(st.session_state.neo4j_messages)

# 验证模型是否连接成功
if st.session_state.model_config:
    st.sidebar.info("✅ AI Model Connected Successfully")
else:
    st.warning("⚠️ No AI model is connected")

os.makedirs(f"{st.session_state.file_path}/records/log", exist_ok=True)
os.makedirs(f"{st.session_state.file_path}/records/chat_history", exist_ok=True)
os.makedirs(f"{st.session_state.file_path}/records/table", exist_ok=True)
os.makedirs(f"{st.session_state.file_path}/records/graph", exist_ok=True)

# 1. 展示历史记录
with st.sidebar:
    st.markdown("---")
    st.markdown("### 💬 对话历史")
    # 获取所有历史对话
    all_conversations = history_manager.get_all_conversations()
    # 当前对话状态指示
    current_loaded = st.session_state.get("loaded_conversation_id")
    if current_loaded:
        st.info(f"📁 当前加载: {current_loaded}")
    # 显示历史对话列表
    if all_conversations:
        for conv in all_conversations:
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    is_current = (
                        conv["id"] == st.session_state.get("loaded_conversation_id")
                        or conv["id"] == st.session_state.get("session_id")
                    )
                    if st.button(
                        f"{'▶️ ' if is_current else ''}{conv['title']}",
                        key=f"conv_{conv['id']}",
                        help=f"点击加载此对话 ({conv['timestamp']})",
                        use_container_width=True,
                    ):
                        load_conversation(conv["id"], history_manager)
                with col2:
                    if st.button(
                        "👁️",
                        key=f"preview_{conv['id']}",
                        help="预览对话",
                    ):
                        show_conversation_preview(conv)
                with col3:
                    if st.button(
                        "🗑️",
                        key=f"delete_{conv['id']}",
                        help="删除此对话",
                    ):
                        deleted_id = conv["id"]
                        if history_manager.delete_conversation(deleted_id):
                            # 同时清除缓存的历史对象，见第 8 点
                            if "chat_histories" in st.session_state:
                                st.session_state.chat_histories.pop(deleted_id, None)
                            # 删除的是当前对话
                            if deleted_id == st.session_state.get("loaded_conversation_id"):
                                st.session_state.messages = []
                                st.session_state.loaded_conversation_id = None
                                st.session_state.session_id = None
                                st.session_state.logger = None
                                st.session_state.pop("history", None)
                            st.rerun()
                        else:
                            st.error(f"删除对话失败：{conv['title']}")
    else:
        st.info("暂无历史对话")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ 新建对话", use_container_width=True):
            create_new_conversation()
    with col2:
        if st.button("🔄 刷新列表", use_container_width=True):
            st.rerun()

# 2. 展示已有聊天记录
cypher_records = {}

if st.session_state.messages:
    for idx, message in enumerate(st.session_state.messages):
        msg_type = message.get("type", "")
        data = message.get("data", "")
        if msg_type == "human" or msg_type == "HumanMessageChunk":
            if "请根据'summary_result'进行总结。" in data["content"]:
                continue
            else:
                safe_chat(role_type="human", content=data["content"])

        elif msg_type == "ai" or msg_type == "AIMessageChunk":
            content = data.get("content", "")

            if 'thought_process' in content or 'cypher_query' in content:
                try:
                    content = json.loads(content)

                    if 'thought_process' in content and content.get("thought_process"):
                        if 'cypher_query' in content and content.get("cypher_query"):
                            if "thought_process" in content:
                                with st.expander("🤔 推理过程", expanded=False):
                                    render_content(content.get("thought_process", ""))

                            if "cypher_query" in content:
                                with st.expander("🔍 Cypher查询", expanded=False):
                                    st.code(content.get("cypher_query", ""), language="cypher")
                                    cypher_records[idx] = content.get("cypher_query", "")
                        else:
                            content = data.get("content", "")
                            content = "响应内容中的cypher_query处出错或你的要求不合理：\n" + content + "\n请重新提问。\n"
                            safe_chat(role_type="ai", content=content)
                    else:
                        content = data.get("content", "")
                        content = "响应内容中的thought_process处出错：\n" + content + "\n请重新提问。\n"
                        safe_chat(role_type="ai", content=content)

                except:
                    content = data.get("content", "")
                    content = "响应格式生成出错：\n" + content + "\n请重新提问。\n"
                    safe_chat(role_type="ai", content=content)
            else:
                safe_chat(role_type="ai", content=content)

            if content == "查询数据如下（可保存所有数据 - 可展示数据分析图）":
                max_idx = max(cypher_records.keys())
                cypher_query = cypher_records[max_idx]
                convert_data = get_data(cypher_query, for_display=True)
                st.dataframe(convert_data, use_container_width=True)
                excel_path = os.path.join(st.session_state.file_path, "records/table")
                keep_idx = f"{st.session_state.loaded_conversation_id}_{max_idx}"

                if "LIMIT 20" in cypher_records[max_idx] and len(convert_data)>=20:
                    if st.button("搜索并保存所有数据", key=f"cypher_present_{max_idx}", use_container_width=True):
                        try:
                            cypher_records_all = cypher_records[max_idx].replace("LIMIT 20", "")
                            data = get_data(cypher_records_all, for_display=False)

                            if data is not None and not data.empty:
                                data.to_excel(f"{excel_path}/{keep_idx}.xlsx",
                                              index=False)
                                st.success(f"数据已保存：{excel_path}/{keep_idx}.xlsx")

                                param = {"data": data,
                                         "conversation_id": st.session_state.loaded_conversation_id,
                                         "idx": max_idx}
                                st.session_state.cleaning_dialog_params[keep_idx] = param

                                st.rerun()
                            else:
                                st.info("没有查询到相关数据")
                        except Exception as e:
                            st.warning(f"获取数据时出错：{e}")
                else:
                    st.info("已展示全部数据")
                    if st.button("保存数据", key=f"cypher_present_{max_idx}", use_container_width=True):

                        convert_data.to_excel(f"{excel_path}/{keep_idx}.xlsx",
                                      index=False)
                        st.success(f"数据已保存：{excel_path}/{keep_idx}.xlsx")

                        param = {"data": convert_data,
                                 "conversation_id": st.session_state.loaded_conversation_id,
                                 "idx": max_idx}
                        st.session_state.cleaning_dialog_params[keep_idx] = param
                        st.rerun()

                if st.session_state.cleaning_dialog_params is not None and keep_idx in st.session_state.cleaning_dialog_params:
                    st.success(f"数据已保存：{excel_path}/{keep_idx}.xlsx")

                    main_right_side, main_left_side = st.columns(2)
                    with main_right_side:
                        if st.button("数据清洗和分析", key=f"clean_data_{max_idx}", use_container_width=True):
                            st.session_state.show_cleaning_dialog = True
                            st.session_state.dialog_triggered = keep_idx
                            st.session_state.cleaning_or_viz = "cleaning"
                            st.rerun()

                    with main_left_side:
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            st.session_state.cleaning_dialog_params[keep_idx]["data"].to_excel(writer, index=False)
                        excel_data = output.getvalue()

                        # 提供下载按钮
                        st.download_button(
                            label="📥 下载所有数据",
                            data=excel_data,
                            file_name=f"{keep_idx}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )


                if st.session_state.viz_cleaned_data_params is not None and keep_idx in st.session_state.viz_cleaned_data_params:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("查看可视化图表", key=f"viz_{max_idx}", type="primary", use_container_width=True):
                            st.session_state.show_cleaning_dialog = True
                            st.session_state.dialog_triggered = keep_idx
                            st.session_state.cleaning_or_viz = "viz"
                            st.rerun()

                    with col2:
                        # 保存数据按钮
                        if st.button("保存清洗后的数据", key=f"keep_cleaned_{max_idx}", type="secondary", use_container_width=True):
                            save_path = f"{excel_path}/{keep_idx}_cleaned.xlsx"
                            st.session_state.viz_cleaned_data_params[keep_idx]["data"].to_excel(save_path, index=False)
                            st.success(f"清洗数据已保存: {save_path}")

                    with col3:
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            st.session_state.viz_cleaned_data_params[keep_idx]["data"].to_excel(writer, index=False)
                        excel_data = output.getvalue()

                        # 提供下载按钮
                        st.download_button(
                            label="📥 下载清洗后的数据",
                            data=excel_data,
                            file_name=f"{keep_idx}_cleaned.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
        else:
            st.error("历史记录出现非对话内容")


if st.session_state.show_cleaning_dialog is True and st.session_state.dialog_triggered is not None:
    if st.session_state.cleaning_or_viz == "cleaning":
        keep_idx = st.session_state.dialog_triggered
        show_data_cleaning_dialog(
            raw_data=st.session_state.cleaning_dialog_params[keep_idx]["data"],
            conversation_id=st.session_state.cleaning_dialog_params[keep_idx]["conversation_id"],
            idx=st.session_state.cleaning_dialog_params[keep_idx]["idx"]
        )
    elif st.session_state.cleaning_or_viz == "viz":
        keep_idx = st.session_state.dialog_triggered
        show_visualization_dialog(
            cleaned_data=st.session_state.viz_cleaned_data_params[keep_idx]["data"],
            excel_path=st.session_state.viz_cleaned_data_params[keep_idx]["excel_path"],
            conversation_id=st.session_state.viz_cleaned_data_params[keep_idx]["conversation_id"],
            idx=st.session_state.viz_cleaned_data_params[keep_idx]["idx"]
        )
    else:
        st.error("数据清洗或可视化图表无法弹出窗口")
    st.session_state.show_cleaning_dialog = False
    st.session_state.dialog_triggered = None
    st.session_state.cleaning_or_viz = None

# 4. 创建首页提示区域
welcome_placeholder = st.empty()

# 只有没有聊天记录时才显示
if not st.session_state.loaded_conversation_id:
    with welcome_placeholder.container():
        st.info(
            """
            👋 欢迎使用知识图谱助手！

            **你可以尝试：**
            - 查询生物质气化相关数据
            - 了解节点和关系的含义
            - 根据温度、原料等条件筛选数据
            - 询问如何使用图谱可视化功能

            请点击下发创建对话，开始对话。
            """
        )

    if st.button("创建对话", use_container_width=True):
        create_new_conversation()

if st.session_state.loaded_conversation_id:

    if user_input := st.chat_input("请输入你的问题..."):

        # 用户发送第一条消息后，立即清除欢迎提示
        safe_chat(role_type="human", content=user_input)

        process_user_input(user_input=user_input, conversation_id=st.session_state.loaded_conversation_id, history_manager=history_manager)

# . 处理预定义问题
with st.sidebar:
    st.markdown("---")

    with st.expander("💡 快速开始"):
        predefined_questions = [
            "帮我找到含有生物质'beech'的实验数据",
            "温度在600-800°C之间的实验有哪些？",
            "帮我找到ER最大的实验数据",
            "生物质基本属性中，ash, volatile, fc, C, H, O这六个属性非空字符串的所有实验数据",
            "哪些实验使用了Steam-Air作为气化剂？"
        ]

        for question in predefined_questions:
            if st.button(question, key=f"predef_{question}", use_container_width=True):
                # 保存当前输入
                st.session_state.predefined_question = question
                st.session_state.should_process_predefined = True
                st.rerun()

if st.session_state.get("should_process_predefined", False) and st.session_state.get("predefined_question"):
    input_to_process = st.session_state.predefined_question
    st.session_state.predefined_question = ""
    st.session_state.should_process_predefined = False

    # 显示用户消息
    safe_chat(role_type="human", content=input_to_process)

    # 处理响应
    if st.session_state.neo4j_status and st.session_state.model_config:
        if st.session_state.loaded_conversation_id:
            process_user_input(user_input=input_to_process, conversation_id=st.session_state.loaded_conversation_id, history_manager=history_manager)
        else:
            create_new_conversation()
            process_user_input(user_input=input_to_process, conversation_id=st.session_state.loaded_conversation_id,
                               history_manager=history_manager)
    else:
        st.error("Failed to connect to the Neo4j or AI model")