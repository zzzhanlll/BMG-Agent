# BMGStreamlit.py
import streamlit as st
import os
import json
import time
import random
from datetime import datetime
import pandas as pd
from typing import List, Dict, Any
from BMGAgent import BMGAgent  # 导入上面创建的BMGAgent类
from BMGHistory_manager import HistoryManager
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ========== 辅助函数 ==========
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

def normalize_column(series):
    """对单列进行min-max归一化"""
    # 确保是数值类型
    if not pd.api.types.is_numeric_dtype(series):
        # 尝试转换为数值类型
        series = pd.to_numeric(series, errors='coerce')

    # 计算有效值的最小值和最大值
    valid_series = series.dropna()
    if valid_series.empty or valid_series.nunique() <= 1:
        # 如果全为相同值或空，返回原值（NaN保持不变）
        return series
    min_val = valid_series.min()
    max_val = valid_series.max()
    if max_val == min_val:
        # 所有值相同，归一化为1
        return series.where(series.isna(), 1.0)
    # 归一化
    return (series - min_val) / (max_val - min_val)

def convert_temp(temp):
    try:
        value = extract_numeric(temp)
        if pd.isna(value):
            return np.nan

        temp_str = str(temp).upper()
        if '°C' in temp_str or ('C' in temp_str and 'K' not in temp_str):
            return value + 273.15
        elif 'K' in temp_str:
            return value
        else:
            return value if value > 500 else value + 273.15
    except:
        return np.nan

def extract_numeric(x):
    """提取数值（独立函数，供其他函数调用）"""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)

    x_str = str(x).strip()
    numbers = re.findall(r'[-+]?\d*\.?\d+', x_str)
    return float(numbers[0]) if numbers else np.nan


def generate_visualizations(cleaned_df, output_dir=None):
    """
    生成小提琴图和热图
    """
    if cleaned_df.empty or len(cleaned_df.columns)<2:
        return None, None

    # 小提琴图
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
    violin_data = [cleaned_df[col].dropna() for col in cleaned_df]
    parts = ax1.violinplot(violin_data, showmeans=True, showmedians=True)
    ax1.set_xticks(range(1, len(violin_data) + 1))
    ax1.set_xticklabels(numeric_cols, rotation=45)
    ax1.set_title('Violin Plot')
    ax1.set_ylabel('value')
    plt.tight_layout()

    # 热图
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    correlation_matrix = cleaned_df.select_dtypes(include=[np.number]).corr()
    if not correlation_matrix.empty:
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f',
                    cmap='coolwarm', center=0, ax=ax2)
        ax2.set_title('HeatMap')
        plt.tight_layout()

    return fig1, fig2

# ========== 加载对话函数 ==========
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
        # 更新session state
        st.session_state.messages = langchain_messages
        st.session_state.session_id = conversation_id
        st.session_state.loaded_conversation_id = conversation_id
        st.session_state.conversation_modified = False

        # 重新初始化Agent（如果需要）
        if "agent" in st.session_state:
            # 保留原有配置，只更新session_id
            api_config = {
                "api_key": st.session_state.get("api_key", ""),
                "base_url": st.session_state.get("base_url", "https://openrouter.ai/api/v1"),
                "model_name": st.session_state.get("model_name", "deepseek/deepseek-v3.2")
            }

            try:
                st.session_state.agent = BMGAgent(
                    session_id=conversation_id,
                    api_config=api_config,
                    file_path=st.session_state.file_path
                )
                st.success(f"✅ 已加载对话: {conversation_id}")
            except Exception as e:
                st.error(f"❌ 重新初始化Agent失败: {e}")

        st.rerun()
    else:
        st.error(f"无法加载对话: {conversation_id}")

def handle_new_conversation():
    """处理新建对话"""
    # 生成新的session_id
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    random_number = random.randint(1000, 9999)
    st.session_state.session_id = f"session_{current_time}_{random_number}"

    # 重置消息和状态
    st.session_state.messages = []
    st.session_state.loaded_conversation_id = None
    st.session_state.conversation_modified = False

    # 重新初始化Agent
    if st.session_state.get("api_key"):
        api_config = {
            "api_key": st.session_state.api_key,
            "base_url": st.session_state.get("base_url", "https://openrouter.ai/api/v1"),
            "model_name": st.session_state.get("model_name", "deepseek/deepseek-v3.2")
        }

        try:
            st.session_state.agent = BMGAgent(
                session_id=st.session_state.session_id,
                api_config=api_config,
                file_path=st.session_state.file_path
            )
            st.success("✅ 新对话已创建")
            st.session_state.loaded_conversation_id = st.session_state.session_id
        except Exception as e:
            st.error(f"❌ 初始化Agent失败: {e}")

    st.rerun()

# ========== 处理用户输入的函数 ==========
def process_user_input(user_input: str, conversation_id: str, history_manager: HistoryManager):
    """处理用户输入并生成响应"""

    st.session_state.conversation_modified = True
    cypher_records = {}

    # 获取agent响应
    try:
        agent_generator = st.session_state.agent.task_execution(user_input)

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
                excel_path = os.path.join(st.session_state.file_path, "keep_data/excel_data")
                keep_idx = f"{st.session_state.loaded_conversation_id}_{idx}"

                if "LIMIT 20" in cypher_records[idx] and len(chunk) >= 20:
                    if st.button("搜索并保存所有数据", key=f"cypher_present_{idx}", use_container_width=True):
                        try:
                            cypher_records_all = cypher_records[idx].replace("LIMIT 20", "")
                            data = st.session_state.agent.get_data(cypher_records_all, for_display=False)

                            if data is not None and not data.empty:
                                data.to_excel(
                                    f"{excel_path}/{keep_idx}.xlsx",
                                    index=False)
                                st.success(
                                    f"数据已保存：{excel_path}/{keep_idx}.xlsx")

                                param = {"data": data,
                                         "conversation_id": st.session_state.loaded_conversation_id,
                                         "idx": idx}
                                st.session_state.cleaning_dialog_params = {keep_idx: param}

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
                        st.session_state.cleaning_dialog_params = {keep_idx: param}

                        st.rerun()

            else:
                safe_chat(role_type="ai", content=chunk)

        json_exists, st.session_state.messages = history_manager.load_conversation(conversation_id)
        if not json_exists:
            st.error("当前状态文件不存在，请检查")

    except Exception as e:
        error_msg = f"处理请求时出错: {e}"
        safe_chat(role_type="ai", content=error_msg)

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



# ========== 对话框函数 ==========
@st.cache_resource
def get_history_manager():
    """获取历史管理器单例"""
    return HistoryManager()

@st.dialog("📖 用户手册")
def show_user_manual_popup():
    """显示用户手册"""
    st.markdown("""
    <style>
    .stDialog > div {
        width: 95vw !important;
        max-width: 1200px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    manual_content = """
    # BMG Agent 用户手册

    ## 简介
    BMG Agent是一个交互式生物质气化问答聊天机器人，基于Neo4j数据库和大型语言模型构建。

    ## 功能特性

    ### 1. 智能问答
    - 支持自然语言查询生物质气化实验数据
    - 自动生成Cypher查询语句
    - 提供数据分析和总结

    ### 2. 数据可视化
    - 支持图形化展示数据关系
    - 显示查询路径结构

    ### 3. 历史记录
    - 自动保存对话历史
    - 支持历史对话查看和恢复

    ### 4. 预定义问题
    - 提供常见问题示例
    - 快速开始查询

    ## 使用指南

    ### 基本查询
    1. 在底部输入框中输入您的问题
    2. 系统将自动分析问题并生成Cypher查询
    3. 查看查询结果和总结分析

    ### 示例问题
    - "帮我找到含有生物质'beech'的实验数据"
    - "温度在600-800°C之间的实验有哪些？"
    - "哪种生物质产生的氢气最多？"

    ## 数据说明
    数据库包含以下节点类型：
    - **Id**: 文章ID和数据条目索引
    - **Basic_properties**: 生物质基本属性
    - **Reactor**: 反应器条件
    - **Production_properties**: 气化产物
    - **Metadata**: 文章元数据

    ## 注意事项
    - 确保网络连接正常
    - 查询结果最多返回20条记录
    - 如果记录超过20条，可以保存所有数据
    - 复杂的查询可能需要较长时间
    """

    with st.container(height=700):
        st.markdown(manual_content, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("关闭手册", type="primary", use_container_width=True):
            st.rerun()

@st.dialog("⚙️ 设置")
def show_settings_dialog():
    """显示设置对话框"""
    st.markdown("""
    <style>
    .stDialog > div {
        width: 80vw !important;
        max-width: 800px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.subheader("API配置")

    # API配置
    api_key = st.text_input(
        "API Key:",
        type="password",
        value=st.session_state.get("api_key", ""),
        help="OpenRouter API Key"
    )

    base_url = st.text_input(
        "Base URL:",
        value=st.session_state.get("base_url", "https://openrouter.ai/api/v1"),
        help="API基础URL"
    )

    model_name = st.text_input(
        "Model Name:",
        value=st.session_state.get("model_name", "deepseek/deepseek-v3.2"),
        help="模型名称"
    )

    file_path = st.text_input(
        "File Path:",
        value=st.session_state.get("file_path", os.getcwd()),
        help="保存数据的路径",
        placeholder=f"目前路径：{st.session_state.file_path}",
    )

    # 保存按钮
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("保存", type="primary"):
            st.session_state.api_key = api_key
            st.session_state.base_url = base_url
            st.session_state.model_name = model_name
            st.session_state.file_path = file_path
            st.session_state.api_config_changed = True
            st.toast("设置已保存", icon="✅")

    with col3:
        if st.button("关闭", type="secondary"):
            st.rerun()

    # 显示当前配置状态
    if st.session_state.get("agent"):
        st.info("✅ Agent已初始化")
    else:
        st.warning("⚠️ Agent未初始化，请配置API Key")

    st.markdown("---")
    st.markdown("### 通过OpenRouter获取模型 (推荐)：")
    st.info("🔗 **[OpenRouter 官网](https://openrouter.ai)**")


@st.dialog("数据可视化")
def show_visualization_dialog(cleaned_data, excel_path=None, conversation_id=None, idx=None):
    """
    显示数据可视化图表对话框
    """
    if cleaned_data.empty:
        st.error("没有可显示的数据")
        return

    # 生成可视化
    with st.spinner("生成可视化图表..."):
        fig1, fig2 = generate_visualizations(cleaned_data)

    if fig1:
        st.markdown("### 小提琴图")
        st.pyplot(fig1)

    if fig2:
        st.markdown("### 相关系数热图")
        st.pyplot(fig2)

    # 如果提供了保存路径，提供保存图像选项
    if excel_path and conversation_id and idx is not None:
        if st.button("保存可视化图像", type="secondary"):
            if fig1:
                fig1.savefig(f"{excel_path}/{conversation_id}_{idx}_violin.png",
                             dpi=300, bbox_inches='tight')
                st.success(f"小提琴图已保存: {excel_path}/{conversation_id}_{idx}_violin.png")
            if fig2:
                fig2.savefig(f"{excel_path}/{conversation_id}_{idx}_heatmap.png",
                             dpi=300, bbox_inches='tight')
                st.success(f"热图已保存: {excel_path}/{conversation_id}_{idx}_heatmap.png")

@st.dialog("数据清洗和分析")
def show_data_cleaning_dialog(raw_data, conversation_id, idx):
    """
    数据清洗对话框
    """

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

            excel_path = os.path.join(st.session_state.file_path, "keep_data/graph")
            param = {"data": cleaned_data,
                     "excel_path": excel_path,
                     "conversation_id": conversation_id,
                     "idx": idx}

            st.session_state.viz_cleaned_data_params = {f"{st.session_state.loaded_conversation_id}_{idx}": param}

            st.info("点击返回可以进行查看可视化图表和保存清洗后数据")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("返回", type="primary", use_container_width=True):
            st.rerun()

# ========== 主应用 ==========
def main():
    # 页面配置
    st.set_page_config(
        page_title="BMG Agent - 生物质气化问答系统",
        page_icon="🌱",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 自定义CSS
    st.markdown("""
    <style>
    /* 主容器样式 */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* 侧边栏样式 */
    .stSidebar {
        background-color: #f8f9fa;
        padding: 20px;
    }

    /* 消息样式 */
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }

    .stChatMessage[data-testid="user"] {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }

    .stChatMessage[data-testid="assistant"] {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }

    /* 扩展器样式 */
    .streamlit-expanderHeader {
        background-color: #e8f5e8;
        border-radius: 5px;
        padding: 10px;
        font-weight: bold;
    }

    /* 按钮样式 */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        transition: all 0.3s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    /* 输入框样式 */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 12px;
    }

    /* 右侧边栏样式 */
    .right-sidebar {
        background-color: #f0f7ff;
        padding: 20px;
        border-left: 1px solid #e0e0e0;
        height: 100vh;
        overflow-y: auto;
    }

    /* 历史记录项样式 */
    .history-item {
        padding: 10px;
        margin-bottom: 8px;
        border-radius: 8px;
        background-color: white;
        border: 1px solid #e0e0e0;
        cursor: pointer;
        transition: all 0.2s;
    }

    .history-item:hover {
        background-color: #e3f2fd;
        border-color: #2196f3;
        transform: translateX(5px);
    }

    .history-item.active {
        background-color: #bbdefb;
        border-color: #1976d2;
        font-weight: bold;
    }
    
    /* 全局对话框居中样式 */
    .stDialog, 
    div[data-testid="stDialog"],
    div[role="dialog"] {
        align-items: center !important;
        justify-content: center !important;
    }
    
    .stDialog > div,
    div[data-testid="stDialog"] > div,
    div[role="dialog"] > div {
        margin: 0 auto !important;
        left: 0 !important;
        right: 0 !important;
        transform: none !important;
        position: relative !important;
    }
    
    /* 定位整个 chat_input 容器 - 只作为定位容器，不添加白色背景 */
    div[data-testid="stElementContainer"][class*="st-key-chat_input"] {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        z-index: 9999;
        display: flex;
        justify-content: center;  /* 水平居中 */
        pointer-events: none;      /* 让容器不拦截点击事件 */
        background: transparent;   /* 透明背景 */
    }
    
    /* 只对输入框本身应用样式，保持居中且白色背景只包裹输入区域 */
    div[data-testid="stElementContainer"][class*="st-key-chat_input"] div[data-testid="stChatInput"] {
        right: 50px;
        width: 100%;
        max-width: 800px;
        pointer-events: auto;      /* 恢复输入框的点击事件 */
        margin-bottom: 60px;       /* 距离底部间距 */
        }
    
    /* 确保主内容区域不会被固定输入框遮挡 */
    .main .block-container {
        padding-bottom: 120px;  /* 为更高的输入框留出更多空间 */
    }

    </style>
    """, unsafe_allow_html=True)

    # 初始化历史管理器
    history_manager = get_history_manager()

    # ========== 初始化Session State ==========
    if "session_id" not in st.session_state:
        # 生成唯一的session_id
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        random_number = random.randint(1000, 9999)
        st.session_state.session_id = f"session_{current_time}_{random_number}"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "loaded_conversation_id" not in st.session_state:
        st.session_state.loaded_conversation_id = None

    if "conversation_modified" not in st.session_state:
        st.session_state.conversation_modified = False

    if "graph_visibility" not in st.session_state:
        st.session_state.graph_visibility = {}

    if "api_key" not in st.session_state:
        st.session_state.api_key = ""

    if "base_url" not in st.session_state:
        st.session_state.base_url = "https://openrouter.ai/api/v1"

    if "model_name" not in st.session_state:
        st.session_state.model_name = "deepseek/deepseek-v3.2"

    if "api_config_changed" not in st.session_state:
        st.session_state.api_config_changed = False

    if "agent" not in st.session_state:
        st.session_state.agent = None

    if "file_path" not in st.session_state:
        st.session_state.file_path = os.getcwd()

    if "show_cleaning_dialog" not in st.session_state:
        st.session_state.show_cleaning_dialog = False

    if "cleaning_dialog_params" not in st.session_state:
        st.session_state.cleaning_dialog_params = None

    if "dialog_triggered" not in st.session_state:
        st.session_state.dialog_triggered = None

    if "cleaning_or_viz" not in st.session_state:
        st.session_state.cleaning_or_viz = None

    if "viz_cleaned_data_params" not in st.session_state:
        st.session_state.viz_cleaned_data_params = None

    # ========== 页面标题和按钮 ==========
    col1, col2, col3, col4 = st.columns([6, 1, 1, 1])

    with col1:
        st.title("🌱 BMG Agent")
        st.subheader("生物质气化数据库智能问答系统")

    with col2:
        if st.button("🆕 新建对话", help="开始新的对话", use_container_width=True):
            handle_new_conversation()  # 使用新的处理函数

    with col3:
        if st.button("📖 手册", help="查看用户手册", use_container_width=True):
            show_user_manual_popup()

    with col4:
        if st.button("⚙️ 设置", help="系统设置", use_container_width=True):
            show_settings_dialog()

    st.markdown("---")

    # ========== 初始化Agent ==========
    if st.session_state.api_config_changed or st.session_state.agent is None:
        if st.session_state.api_key:
            try:
                with st.spinner("🔄 初始化Agent..."):
                    api_config = {
                        "api_key": st.session_state.api_key,
                        "base_url": st.session_state.base_url,
                        "model_name": st.session_state.model_name
                    }

                    st.session_state.agent = BMGAgent(
                        session_id=st.session_state.session_id,
                        api_config=api_config,
                        file_path=st.session_state.file_path
                    )

                    st.session_state.api_config_changed = False
                    st.success("✅ Agent初始化成功")
                    st.session_state.loaded_conversation_id = st.session_state.session_id
            except Exception as e:
                st.error(f"❌ Agent初始化失败: {e}")
                st.session_state.agent = None
        else:
            st.warning("⚠️ 请先在设置中配置API Key")

    # ========== 创建布局 ==========
    # 三列布局：左侧历史记录，中间聊天区域，右侧信息面板
    left_col, main_col, right_col = st.columns([2, 6, 3])

    # ========== 左侧：历史记录 ==========
    with left_col:
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
                        # 判断是否是当前加载的对话
                        is_current = (conv["id"] == st.session_state.get("loaded_conversation_id") or
                                      conv["id"] == st.session_state.session_id)

                        # 创建对话项目
                        if st.button(
                                f"{'▶️ ' if is_current else ''}{conv['title']}",
                                key=f"conv_{conv['id']}",
                                help=f"点击加载此对话 ({conv['timestamp']})",
                                use_container_width=True
                        ):
                            # 加载对话
                            load_conversation(conv["id"], history_manager)

                    with col2:
                        # 预览按钮
                        if st.button("👁️", key=f"preview_{conv['id']}", help="预览对话"):
                            show_conversation_preview(conv)

                    with col3:
                        # 删除按钮
                        if st.button("🗑️", key=f"delete_{conv['id']}", help="删除此对话"):
                            if history_manager.delete_conversation(conv["id"]):
                                st.success(f"已删除对话: {conv['title']}")
                                if conv['id'] == st.session_state.loaded_conversation_id:
                                    handle_new_conversation()
                                st.rerun()
        else:
            st.info("暂无历史对话")

        # 对话管理按钮
        col1, col2 = st.columns(2)
        with col2:
            if st.button("🔄 刷新列表", use_container_width=True):
                st.rerun()

        st.markdown("---")

        # 预定义问题
        with st.expander("### 💡 快速开始"):

            predefined_questions = [
                "帮我找到含有生物质'beech'的实验数据",
                "温度在600-800°C之间的实验有哪些？",
                "帮我找到ER最大的实验数据",
                "生物质基本属性中，ash, volatile, fc, C, H, O这六个属性非空字符串的所有实验数据",
                "农业残留物在不同温度下的气化产物分布",
                "哪些实验使用了Steam-Air作为气化剂？"
            ]

            for question in predefined_questions:
                if st.button(question, key=f"predef_{question}", use_container_width=True):
                    # 保存当前输入
                    st.session_state.predefined_question = question
                    st.session_state.should_process_predefined = True
                    st.rerun()

    # ========== 中间：主聊天区域 ==========
    with main_col:
        # 显示现有消息
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
                        convert_data = st.session_state.agent.get_data(cypher_query, for_display=True)
                        st.dataframe(convert_data, use_container_width=True)
                        excel_path = os.path.join(st.session_state.file_path, "keep_data/excel_data")
                        keep_idx = f"{st.session_state.loaded_conversation_id}_{max_idx}"

                        if "LIMIT 20" in cypher_records[max_idx] and len(convert_data)>=20:
                            if st.button("搜索并保存所有数据", key=f"cypher_present_{max_idx}", use_container_width=True):
                                try:
                                    cypher_records_all = cypher_records[max_idx].replace("LIMIT 20", "")
                                    data = st.session_state.agent.get_data(cypher_records_all, for_display=False)

                                    if data is not None and not data.empty:
                                        data.to_excel(f"{excel_path}/{keep_idx}.xlsx",
                                                      index=False)
                                        st.success(f"数据已保存：{excel_path}/{keep_idx}.xlsx")

                                        param = {"data": data,
                                                 "conversation_id": st.session_state.loaded_conversation_id,
                                                 "idx": max_idx}
                                        st.session_state.cleaning_dialog_params = {keep_idx: param}

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
                                st.session_state.cleaning_dialog_params = {keep_idx: param}

                                st.rerun()

                        if st.session_state.cleaning_dialog_params is not None and keep_idx in st.session_state.cleaning_dialog_params:
                            st.success(f"数据已保存：{excel_path}/{keep_idx}.xlsx")
                            if st.button("数据清洗和分析", key=f"clean_data_{max_idx}", use_container_width=True):
                                st.session_state.show_cleaning_dialog = True
                                st.session_state.dialog_triggered = keep_idx
                                st.session_state.cleaning_or_viz = "cleaning"
                                st.rerun()

                        if st.session_state.viz_cleaned_data_params is not None and keep_idx in st.session_state.viz_cleaned_data_params:
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("查看可视化图表", type="primary", use_container_width=True):
                                    st.session_state.show_cleaning_dialog = True
                                    st.session_state.dialog_triggered = keep_idx
                                    st.session_state.cleaning_or_viz = "viz"
                                    st.rerun()

                            with col2:
                                # 保存数据按钮
                                if st.button("保存清洗后数据", type="secondary", use_container_width=True):
                                    save_path = f"{excel_path}/{keep_idx}_cleaned.xlsx"
                                    st.session_state.viz_cleaned_data_params[keep_idx]["data"].to_excel(save_path, index=False)
                                    st.success(f"清洗数据已保存: {save_path}")
                else:
                    st.error("历史记录出现非对话内容")

        # 处理预定义问题
        if st.session_state.get("should_process_predefined", False) and st.session_state.get("predefined_question"):
            input_to_process = st.session_state.predefined_question
            st.session_state.predefined_question = ""
            st.session_state.should_process_predefined = False

            # 显示用户消息
            safe_chat(role_type="human", content=input_to_process)

            # 处理响应
            if st.session_state.agent:
                process_user_input(user_input=input_to_process, conversation_id=st.session_state.loaded_conversation_id, history_manager=history_manager)
            else:
                st.error("Agent未初始化，请检查设置")

    # ========== 右侧：信息面板 ==========
    with right_col:
        # 可收放的侧边栏
        json_exists, st.session_state.messages = history_manager.load_conversation(st.session_state.loaded_conversation_id)
        if not json_exists:
            st.error("当前状态文件不存在，请检查")
        with st.expander("💾 对话管理", expanded=False):
            # 当前对话信息
            if st.session_state.messages:
                st.markdown(f"**当前对话ID**: `{st.session_state.loaded_conversation_id}`")
                st.markdown(f"**消息数量**: {len(st.session_state.messages)}")
            else:
                st.info("没有正在进行的对话")

            # 历史统计
            all_conversations = history_manager.get_all_conversations()
            if all_conversations:
                st.markdown(f"**历史对话总数**: {len(all_conversations)}")

                # 最近对话
                st.markdown("**最近对话**:")
                for conv in all_conversations[:3]:
                    st.markdown(f"- {conv['title']} ({conv['timestamp']})")
        with st.expander("📊 数据库信息", expanded=True):
            if st.session_state.agent and st.session_state.agent.graph:
                st.success("✅ 数据库连接正常")
                st.info(f"Session ID: {st.session_state.loaded_conversation_id}")

                # 显示数据库统计信息
                try:
                    # 获取数据统计
                    count_query = "MATCH (n) RETURN count(n) as total_nodes"
                    result = st.session_state.agent.graph.query(count_query)
                    if result:
                        st.metric("总节点数", result[0]['total_nodes'])

                    # 显示节点类型
                    node_types_query = """
                    MATCH (n) 
                    RETURN labels(n)[0] as node_type, count(*) as count 
                    ORDER BY count DESC
                    """
                    node_result = st.session_state.agent.graph.query(node_types_query)
                    if node_result:
                        st.markdown("#### 节点类型分布")
                        for item in node_result[:5]:
                            st.text(f"{item['node_type']}: {item['count']}")

                except Exception as e:
                    st.warning(f"无法获取数据库统计: {e}")
            else:
                st.warning("⚠️ 数据库未连接")

        with st.expander("🔧 查询示例", expanded=False):
            st.markdown("""
            **简单查询：**
            - `农业残留物的实验数据`
            - `温度大于700°C的实验`

            **复杂查询：**
            - `Macroalgae在400-600°C温度范围内的气化产物`
            - `使用Steam-Air气化剂且压力为1bar的实验`

            **分析查询：**
            - `哪种生物质产生的氢气最多？`
            - `温度对氢气产量的影响`
            """)

        with st.expander("📈 使用统计", expanded=False):
            if st.session_state.messages:
                user_count = len([m for m in st.session_state.messages if m["type"] == "human"])
                assistant_count = len([m for m in st.session_state.messages if m["type"] == "ai"])

                st.metric("用户消息", user_count)
                st.metric("助手回复", assistant_count)

                if user_count > 0:
                    st.progress(min(assistant_count / user_count, 1.0), text="回复率")
            else:
                st.info("暂无统计数据")

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

    # ========== 底部：用户输入 ==========
    # 在右侧列下方添加用户输入
    with main_col:
        # 用户输入框
        user_input = st.chat_input(
            "请输入您的问题...",
            key="chat_input"
        )

        # 处理用户输入
        if user_input:
            # 显示用户消息
            safe_chat(role_type="human", content=user_input)

            # 处理响应
            if st.session_state.agent:
                process_user_input(user_input=user_input, conversation_id=st.session_state.session_id, history_manager=history_manager)
            else:
                st.error("Agent未初始化，请检查设置")
                st.info("请在设置中配置API Key")
# ========== 运行主应用 ==========
if __name__ == "__main__":
    # 确保必要的目录存在
    os.makedirs("./logs/chat_log", exist_ok=True)
    os.makedirs("./logs/chat_history", exist_ok=True)

    main()