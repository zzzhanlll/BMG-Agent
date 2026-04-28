# history_manager.py
import os
import json
import glob
from typing import List, Dict, Any, Optional
from datetime import datetime
import re


class HistoryManager:
    """历史对话管理器，负责从chat_history文件夹读取和管理历史记录"""

    def __init__(self, chat_history_dir: str = "./logs/chat_history"):
        """
        初始化历史管理器

        Args:
            chat_history_dir: 聊天历史文件夹路径
        """
        self.chat_history_dir = chat_history_dir
        os.makedirs(chat_history_dir, exist_ok=True)

    def get_all_conversations(self) -> List[Dict[str, Any]]:
        """
        获取所有历史对话

        Returns:
            历史对话列表，每个对话包含id, title, timestamp, file_path等信息
        """
        conversations = []

        # 查找所有json文件
        json_files = glob.glob(os.path.join(self.chat_history_dir, "*.json"))

        for file_path in json_files:
            try:
                conversation_id = os.path.basename(file_path).replace(".json", "")

                # 读取文件获取对话内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    messages = json.load(f)

                # 生成对话标题
                title = self._generate_conversation_title(messages, conversation_id)

                # 获取文件修改时间作为时间戳
                timestamp = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S")

                conversations.append({
                    "id": conversation_id,
                    "title": title,
                    "timestamp": timestamp,
                    "file_path": file_path,
                    "message_count": len(messages),
                    "messages": messages  # 包含完整消息，便于快速预览
                })

            except Exception as e:
                print(f"Error loading conversation from {file_path}: {e}")
                continue

        # 按时间倒序排序（最新的在前面）
        conversations.sort(key=lambda x: x["timestamp"], reverse=True)

        return conversations

    def _generate_conversation_title(self, messages: List[Dict[str, Any]], conversation_id: str) -> str:
        """
        根据消息生成对话标题

        Args:
            messages: 消息列表
            conversation_id: 对话ID

        Returns:
            对话标题
        """
        if not messages:
            return f"对话 {conversation_id}"

        # 查找第一个人类消息作为标题
        for msg in messages:
            if msg.get("type") == "human" or (msg.get("data") and msg["data"].get("type") == "human"):
                content = msg.get("content") or msg.get("data", {}).get("content", "")
                if isinstance(content, str):
                    # 提取前50个字符作为标题
                    title = content[:50].strip()
                    if len(content) > 50:
                        title += "..."
                    return title
                elif isinstance(content, dict):
                    # 如果是字典，尝试提取thought_process
                    thought = content.get("thought_process", "")
                    if thought:
                        title = thought[:50].strip()
                        if len(thought) > 50:
                            title += "..."
                        return title

        # 如果没有找到合适的内容，返回对话ID
        return f"对话 {conversation_id}"

    def load_conversation(self, conversation_id: str) -> tuple[bool, Optional[List[Dict[str, Any]]]]:
        """
        加载特定对话

        Args:
            conversation_id: 对话ID

        Returns:
            (是否成功, 消息列表)
        """
        file_path = os.path.join(self.chat_history_dir, f"{conversation_id}.json")

        if not os.path.exists(file_path):
            return False, []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                messages = json.load(f)
                messages = messages if messages else []
            return True, messages
        except Exception as e:
            print(f"Error loading conversation {conversation_id}: {e}")
            return False, []

    def delete_conversation(self, conversation_id: str) -> bool:
        """
        删除对话

        Args:
            conversation_id: 对话ID

        Returns:
            是否删除成功
        """
        file_path = os.path.join(self.chat_history_dir, f"{conversation_id}.json")

        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                return True
            except Exception as e:
                print(f"Error deleting conversation {conversation_id}: {e}")
                return False
        return False

    def convert_to_streamlit_format(self, langchain_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将LangChain格式的消息转换为Streamlit消息格式

        Args:
            langchain_messages: LangChain格式的消息列表

        Returns:
            Streamlit格式的消息列表
        """
        streamlit_messages = []

        for msg in langchain_messages:
            msg_type = msg.get("type", "")
            data = msg.get("data", {})

            if msg_type == "human" or data.get("type") == "human":
                # 人类消息
                content = data.get("content", "")
                streamlit_messages.append({
                    "role": "user",
                    "content": content
                })

            elif msg_type == "ai" or data.get("type") == "ai" or msg_type == "AIMessageChunk":
                # AI消息
                content = data.get("content", "")

                # 处理可能的JSON格式内容
                if isinstance(content, str) and content.strip().startswith("{"):
                    try:
                        content_dict = json.loads(content)
                        thought_process = content_dict.get("thought_process", "")
                        cypher_query = content_dict.get("cypher_query", "")

                        streamlit_msg = {
                            "role": "assistant",
                            "content": content
                        }

                        if thought_process:
                            streamlit_msg["thought_process"] = thought_process
                        if cypher_query:
                            streamlit_msg["cypher_query"] = cypher_query

                        streamlit_messages.append(streamlit_msg)
                        continue
                    except json.JSONDecodeError:
                        pass  # 不是JSON，继续普通处理

                # 普通AI消息
                streamlit_messages.append({
                    "role": "assistant",
                    "content": content
                })

        return streamlit_messages

    def convert_to_langchain_format(self, streamlit_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将Streamlit格式的消息转换为LangChain格式

        Args:
            streamlit_messages: Streamlit格式的消息列表

        Returns:
            LangChain格式的消息列表
        """
        langchain_messages = []

        for msg in streamlit_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                langchain_messages.append({
                    "type": "human",
                    "data": {
                        "content": content,
                        "additional_kwargs": {},
                        "response_metadata": {},
                        "type": "human",
                        "name": None,
                        "id": None,
                        "example": False
                    }
                })
            elif role == "assistant":
                # 检查是否是包含thought_process的特殊格式
                if isinstance(content, str) and content.strip().startswith("{"):
                    try:
                        content_dict = json.loads(content)
                        # 已经是JSON格式，保持原样
                        langchain_messages.append({
                            "type": "ai",
                            "data": {
                                "content": content,
                                "additional_kwargs": {"refusal": None},
                                "response_metadata": {
                                    "model_name": "deepseek/deepseek-v3.2",
                                    "finish_reason": "stop"
                                },
                                "type": "ai",
                                "name": None,
                                "id": f"run-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                                "example": False,
                                "tool_calls": [],
                                "invalid_tool_calls": [],
                                "usage_metadata": {
                                    "input_tokens": 0,
                                    "output_tokens": 0,
                                    "total_tokens": 0
                                }
                            }
                        })
                        continue
                    except json.JSONDecodeError:
                        pass  # 不是JSON，继续普通处理

                # 普通AI消息
                langchain_messages.append({
                    "type": "ai",
                    "data": {
                        "content": content,
                        "additional_kwargs": {"refusal": None},
                        "response_metadata": {
                            "model_name": "deepseek/deepseek-v3.2",
                            "finish_reason": "stop"
                        },
                        "type": "ai",
                        "name": None,
                        "id": f"run-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        "example": False,
                        "tool_calls": [],
                        "invalid_tool_calls": [],
                        "usage_metadata": {
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0
                        }
                    }
                })

        return langchain_messages

    def save_current_conversation(self, conversation_id: str, streamlit_messages: List[Dict[str, Any]]):
        """
        保存当前对话到文件

        Args:
            conversation_id: 对话ID
            streamlit_messages: Streamlit格式的消息列表
        """
        # 转换为LangChain格式
        langchain_messages = self.convert_to_langchain_format(streamlit_messages)

        # 保存到文件
        file_path = os.path.join(self.chat_history_dir, f"{conversation_id}.json")

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(langchain_messages, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving conversation {conversation_id}: {e}")