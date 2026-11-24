import pandas as pd
import streamlit as st
from neo4j_viz import Node, Relationship, VisualizationGraph
import streamlit.components.v1 as components
import html
import plotly.express as px

st.set_page_config(
    page_title="📊 BMG Knowledge Graph",
    layout="wide",
)

st.title("📊 BMG Knowledge Graph")

neo4j_ready = (
    st.session_state.get("neo4j_status", False)
    and st.session_state.get("neo4j_graph") is not None
)
if neo4j_ready:
    st.success(
        st.session_state.get(
            "neo4j_messages",
            "✅ Neo4j 已连接",
        )
    )
else:
    st.warning(
        st.session_state.get(
            "neo4j_messages",
            "Neo4j 尚未初始化，请先返回主应用建立连接。",
        )
    )

st.markdown(
    """
    <style>
        div[data-testid="stColumn"]:has(.schema-toc) {
            position: sticky;
            top: 5rem;
            align-self: flex-start;
            height: fit-content;
            z-index: 10;
        }
        /* 目录卡片本身的样式 */
        .schema-toc {
            padding: 1rem;
            border: 1px solid rgba(128, 128, 128, 0.35);
            border-radius: 0.6rem;
            background-color: rgba(128, 128, 128, 0.05);
        }
        .schema-toc-title {
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            color: #808495;
            margin-bottom: 0.8rem;
        }
        .schema-toc a {
            display: block;
            padding: 0.32rem 0;
            color: inherit;
            text-decoration: none;
            font-size: 0.9rem;
            line-height: 1.4;
        }
        .schema-toc a:hover {
            color: #4F8BF9;
            text-decoration: underline;
        }
        .schema-toc hr {
            border: none;
            border-top: 1px solid rgba(128, 128, 128, 0.25);
            margin: 0.75rem 0;
        }
        /* 锚点跳转后，避免标题被顶端区域遮住 */
        h2, h3 {
            scroll-margin-top: 5rem;
        }
        /* 小屏幕时不固定，以免影响阅读 */
        @media (max-width: 768px) {
            div[data-testid="stColumn"]:has(.schema-toc) {
                position: static;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

content_col, toc_col = st.columns([4, 1], gap="large")

with toc_col:
    st.markdown(
        """
        <div class="schema-toc">
            <div class="schema-toc-title">CONTENTS</div>
            <a href="#schema-overview">Schema Overview</a>
            <a href="#graph-structure">Graph Structure</a>
            <a href="#data-entry-identifier">1. Data Entry Identifier</a>
            <a href="#basic-properties">2. Basic Properties</a>
            <a href="#reactor-conditions">3. Reactor Conditions</a>
            <a href="#production-properties">4. Production Properties</a>
            <a href="#publication-metadata">5. Publication Metadata</a>
            <hr>
            <a href="#graph-visualization">Graph Visualization</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# 左侧：实际内容
# ---------------------------------------------------------
st.html("""
<style>
    /* 小屏幕下内容过宽时允许横向滚动 */
    .schema-table-wrapper {
        width: 100%;
        overflow-x: auto;
    }
    .schema-table-wrapper table {
        display: table;
        width: 100%;
        margin: 0;
        border-collapse: collapse;
        table-layout: auto;
    }
    .schema-table-wrapper th,
    .schema-table-wrapper td {
        padding: 0.55rem 0.75rem;
        text-align: left;
        vertical-align: center;
        border: 1px solid rgba(128, 128, 128, 0.25);
    }
    .schema-table-wrapper th {
        background-color: rgba(128, 128, 128, 0.1);
        font-weight: 600;
    }
    /* 第一列按字段内容确定宽度 */
    .schema-table-wrapper th:first-child,
    .schema-table-wrapper td:first-child {
        width: 1%;
        white-space: nowrap;
    }
</style>
""")

def render_schema_table(rows: list[tuple[str, str]]):
    """rows 格式: [(字段名, 描述), ...]"""
    body_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(str(name))}</code></td>"
        f"<td>{html.escape(str(description))}</td>"
        "</tr>"
        for name, description in rows
    )
    table_html = (
        '<div class="schema-table-wrapper">'
        "<table>"
        "<thead>"
        "<tr><th>Property</th><th>Description</th></tr>"
        "</thead>"
        f"<tbody>{body_rows}</tbody>"
        "</table>"
        "</div>"
    )
    st.html(table_html)


with content_col:
    # ===== Schema Overview =====
    st.header("Schema Overview", anchor="schema-overview")
    st.markdown(
        """
        This schema describes the relationships among biomass properties,
        gasification operating conditions, product gas composition,
        and publication metadata.
        """
    )
    if neo4j_ready:
        # 获取数据统计
        count_query = "MATCH (n) RETURN count(n) as total_nodes"
        result = st.session_state.neo4j_graph.query(count_query)
        if result:
            st.metric("Total Number of Nodes:", f"{result[0]['total_nodes']:,}")

        # 显示节点类型
        node_types_query = """
        MATCH (n) 
        RETURN labels(n)[0] as node_type, count(*) as count 
        ORDER BY count DESC
        """
        node_result = st.session_state.neo4j_graph.query(node_types_query)
        if node_result:
            st.markdown("##### Distribution of Node Types")
            df = pd.DataFrame(node_result)
            # 未设置标签的节点会得到 None
            df["node_type"] = df["node_type"].fillna("Unlabeled")
            # 横向图从下往上排列，因此升序后最大值显示在顶部
            df = df.sort_values("count", ascending=True)
            df["count_text"] = df["count"].map(lambda value: f"{value:,}")
            fig = px.bar(
                df,
                x="count",
                y="node_type",
                orientation="h",
                text="count_text",
                color_discrete_sequence=["#4C9AFF"],
            )
            fig.update_traces(
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}: %{x:,}<extra></extra>",
            )
            order = [
                "Id",
                "Basic_properties",
                "Reactor",
                "Production_properties",
                "Metadata",
            ]
            fig.update_layout(
                height=260,
                margin=dict(l=0, r=40, t=10, b=0),
                xaxis_title=None,
                yaxis_title=None,
                yaxis=dict(
                    categoryorder="array",
                    categoryarray=order,
                    autorange="reversed",
                ),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No node type data available at this time")

    # ===== Graph Structure =====
    st.subheader("Graph Structure", anchor="graph-structure")
    with st.container(border=True):
        st.code(
            """
        (:Id)-[:HAS_BASIC_PROPERTIES]->(:Basic_properties)
        (:Id)-[:USES_REACTOR]->(:Reactor)
        (:Id)-[:PRODUCES]->(:Production_properties)
        (:Id)-[:HAS_METADATA]->(:Metadata)
                    """.strip(),
                    language="cypher",
        )
        st.info(
            "`Id` is the central data-entry node. Each data entry links "
            "biomass/feedstock properties, reactor conditions, product gas "
            "properties, and publication metadata."
        )
    st.markdown("---")

    # ===== Id =====
    st.subheader(
        "1. Data Entry Identifier",
        anchor="data-entry-identifier",
    )
    st.markdown(
        "Dry basis: ash + vm + fc = 100 wt%; "
        "C + H + O + N + S = 100 wt%"
    )
    with st.container(border=True):
        st.caption("Central node label: `Id`")
        render_schema_table([
            ("Paper_id", "Paper / article identifier"),
            ("Data_entry_index", "ndex of a data entry within a paper")
        ])

    # ===== Basic Properties =====
    st.subheader("2. Basic Properties", anchor="basic-properties")
    with st.container(border=True):
        st.caption("Node label: `Basic_properties`")
        render_schema_table([
            ("Feedstock", "Biomass / feedstock name, e.g. Macroalgae"),
            ("Ash(wt_%)", "Ash content obtained from biomass proximate analysis (dry basis)"),
            ("Volatile_matter(wt_%)", "Volatile matter content obtained from biomass proximate analysis (dry basis)"),
            ("Fixed_carbon(wt_%)", "Fixed carbon content obtained from biomass proximate analysis (dry basis)"),
            ("Moisture(wt_%)", "Moisture content obtained from biomass proximate analysis"),
            ("C(wt_%)", "Carbon content obtained from biomass ultimate analysis (dry basis)"),
            ("H(wt_%)", "Hydrogen content obtained from biomass ultimate analysis (dry basis)"),
            ("O(wt_%)", "Oxygen content obtained from biomass ultimate analysis (dry basis)"),
            ("N(wt_%)", "Nitrogen content obtained from biomass ultimate analysis (dry basis)"),
            ("S(wt_%)", "Sulfur content obtained from biomass ultimate analysis (dry basis)"),
            ("HHV", "High heating value"),
            ("LHV", "Low heating value"),
        ])

    # ===== Reactor Conditions =====
    st.subheader("3. Reactor Conditions", anchor="reactor-conditions")
    with st.container(border=True):
        st.caption("Node label: `Reactor`")
        render_schema_table([
            ("Temperature(K)", "Gasification temperature, unit: K"),
            ("Pressure", "Gasification pressure"),
            ("Equivalence_ratio", "Equivalence ratio (ER)"),
            ("Agent_type", "Gasification agent, e.g. Steam, Air, CO₂, O₂"),
            ("Agent_to_biomass_ratio", "Agent-to-biomass ratio"),
            ("Steam_to_biomass_ratio", "Steam-to-biomass ratio"),
            ("Reactor_type", "Reactor type, simulation model, or industrial equipment"),
            ("Catalyst", "Catalyst used in the process"),
            ("Particle_size", "Biomass particle size"),
            ("Bed_material_with_size", "Bed material and particle size"),
            (
                "Supplementary_information",
                "Other operating conditions or supplementary information",
            ),
        ])

    # ===== Production Properties =====
    st.subheader(
        "4. Production Properties",
        anchor="production-properties",
    )
    st.markdown(
        "When all four gas production rates are present, "
        "the unit will be converted to volume fraction."
    )
    with st.container(border=True):
        st.caption("Node label: `Production_properties`")
        render_schema_table([
            ("H2", "Hydrogen concentration, typically vol%"),
            ("CO", "Carbon monoxide concentration, typically vol%"),
            ("CO2", "Carbon dioxide concentration, typically vol%"),
            ("CH4", "Methane concentration, typically vol%"),
        ])

    # ===== Publication Metadata =====
    st.subheader(
        "5. Publication Metadata",
        anchor="publication-metadata",
    )
    with st.container(border=True):
        st.caption("Node label: `Metadata`")
        render_schema_table([
            ("UID", "Web of Science unique identifier"),
            ("Title", "Article title"),
            ("Type", "Document type, e.g. Article or Meeting"),
            ("Journal", "Journal or conference name"),
            ("Year", "Publication year"),
            ("Volume", "Volume"),
            ("Issue", "Issue number"),
            ("Pages", "Page range"),
            ("First_Author", "First author"),
            ("Author_Count", "Number of authors"),
            ("All_Authors", "Full author list"),
            ("Citations", "Citation count"),
            ("DOI", "Digital Object Identifier"),
            ("PMID", "	PubMed ID, if available"),
            ("Keywords", "Article keywords"),
            ("WOS_Link", "Web of Science record link"),
        ])

    st.markdown("---")
    # ===== Graph Visualization =====
    st.header(
        "Knowledge Graph Visualization",
        anchor="graph-visualization",
    )
    with st.container(border=True):
        graph_visualization_placeholder = st.empty()
        # 例如：
        # records = graph.query(cypher_query)
        # render_graph(records)

# 1. Streamlit 页面设置
# ============================================================
# 2. 辅助函数
# ============================================================
def safe_value(value, default=""):
    """把 None 转为默认值，并统一转成字符串。"""
    if value is None or value == "":
        return default
    return str(value)
def short_text(value, max_length=45):
    """截断太长的文本，避免节点 caption 过大。"""
    text = safe_value(value)
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text
def get_labels_text(labels):
    """
    labels 是 Cypher labels(node) 返回的列表，例如：
    ["Reactor"]
    """
    if not labels:
        return "Unknown"
    return ", ".join(labels)
# ============================================================
# 3. 根据节点类型生成 neo4j-viz 节点标题
# ============================================================
def build_caption(labels, props):
    """
    labels: Neo4j 节点的 Label 列表，例如 ["Id"]、["Reactor"]
    props:  Neo4j 节点属性字典
    注意：
    这里只把最重要的字段放进 caption。
    如果把所有属性都放上去，图会非常难读。
    """
    labels = set(labels or [])
    props = props or {}
    # -------------------------
    # 中心节点：论文 / 数据记录
    # -------------------------
    if "Id" in labels:
        return (
            "Paper / Data Entry\n"
            f"Paper ID: {safe_value(props.get('Paper_id'), 'Unknown')}\n"
            f"Entry: {safe_value(props.get('Data_entry_index'), 'N/A')}"
        )
    # -------------------------
    # 基础性质节点
    # -------------------------
    if "Basic_properties" in labels:
        return (
            "Basic Properties\n"
            f"Feedstock: {short_text(props.get('Feedstock'), 30)}\n"
            f"C: {safe_value(props.get('C(wt_%)'), 'N/A')} wt%\n"
            f"H: {safe_value(props.get('H(wt_%)'), 'N/A')} wt%\n"
            f"Ash: {safe_value(props.get('Ash(wt_%)'), 'N/A')} wt%\n"
            f"Moisture: {safe_value(props.get('Moisture(wt_%)'), 'N/A')} wt%"
        )
    # -------------------------
    # 反应器节点
    # -------------------------
    if "Reactor" in labels:
        return (
            "Reactor\n"
            f"Type: {short_text(props.get('Reactor_type'), 40)}\n"
            f"Agent: {short_text(props.get('Agent_type'), 30)}\n"
            f"T: {safe_value(props.get('Temperature(K)'), 'N/A')} K"
        )
    # -------------------------
    # 产物组成节点
    # -------------------------
    if "Production_properties" in labels:
        return (
            "Production Properties\n"
            f"H₂: {safe_value(props.get('H2'), 'N/A')}\n"
            f"CO: {safe_value(props.get('CO'), 'N/A')}\n"
            f"CH₄: {safe_value(props.get('CH4'), 'N/A')}\n"
            f"CO₂: {safe_value(props.get('CO2'), 'N/A')}"
        )
    # -------------------------
    # 文献元数据节点
    # -------------------------
    if "Metadata" in labels:
        return (
            "Metadata\n"
            f"Title: {short_text(props.get('Title'), 55)}\n"
            f"Author: {short_text(props.get('First_Author'), 35)}\n"
            f"Year: {safe_value(props.get('Year'), 'N/A')}\n"
            f"DOI: {short_text(props.get('DOI'), 35)}"
        )
    # 兜底：未知节点类型
    return f"{get_labels_text(labels)}\n{short_text(str(props), 50)}"
def build_node_size(labels):
    """按照节点 Label 设定可视化大小。"""
    labels = set(labels or [])
    if "Id" in labels:
        return 30
    if "Reactor" in labels:
        return 23
    if "Production_properties" in labels:
        return 23
    if "Basic_properties" in labels:
        return 23
    if "Metadata" in labels:
        return 20
    return 15

# ============================================================
# 5. 默认 Cypher 查询
# ============================================================

DEFAULT_GRAPH_QUERY = """
MATCH (source:Id)
WITH source
LIMIT 5

OPTIONAL MATCH (source)-[rel]->(target)
WHERE type(rel) IN [
    'HAS_BASIC_PROPERTIES',
    'USES_REACTOR',
    'PRODUCES',
    'HAS_METADATA'
]

RETURN
    elementId(source) AS source_id,
    labels(source) AS source_labels,
    properties(source) AS source_props,

    elementId(target) AS target_id,
    labels(target) AS target_labels,
    properties(target) AS target_props,

    type(rel) AS relationship_type,
    elementId(rel) AS relationship_id
LIMIT 100
""".strip()


# ============================================================
# 6. 图谱查询结果检查
# ============================================================

def validate_graph_records(records):
    """
    检查查询结果是否具有绘图所需字段。

    注意：
    空结果无法检查返回列名，由调用方单独提示。
    """

    if not isinstance(records, list):
        raise ValueError("Neo4j 查询结果不是预期的列表格式。")

    if not records:
        return

    required_fields = {
        "source_id",
        "source_labels",
        "source_props",
        "target_id",
        "target_labels",
        "target_props",
        "relationship_type",
        "relationship_id",
    }

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(
                f"第 {index + 1} 条结果不是字典格式。"
            )

        missing_fields = required_fields - record.keys()

        if missing_fields:
            raise ValueError(
                "Cypher 返回字段不完整，缺少："
                + ", ".join(sorted(missing_fields))
            )

        source_labels = record.get("source_labels")
        source_props = record.get("source_props")
        target_labels = record.get("target_labels")
        target_props = record.get("target_props")

        if source_labels is not None and not isinstance(
            source_labels, (list, tuple)
        ):
            raise ValueError(
                "source_labels 应通过 labels(source) 返回列表。"
            )

        if source_props is not None and not isinstance(source_props, dict):
            raise ValueError(
                "source_props 应通过 properties(source) 返回字典。"
            )

        if target_labels is not None and not isinstance(
            target_labels, (list, tuple)
        ):
            raise ValueError(
                "target_labels 应通过 labels(target) 返回列表。"
            )

        if target_props is not None and not isinstance(target_props, dict):
            raise ValueError(
                "target_props 应通过 properties(target) 返回字典。"
            )


# ============================================================
# 7. 查询结果转换为 neo4j-viz 数据
# ============================================================

def convert_records_to_viz_data(records):
    """
    将查询结果转换成：

        nodes: List[Node]
        relationships: List[Relationship]

    约定：
        source 是关系起点，target 是关系终点。

    用户自定义查询时，应使用有方向的模式：
        (source)-[rel]->(target)
    """

    validate_graph_records(records)

    node_map = {}
    relationships = []
    relationship_ids = set()

    for record in records:
        source_id = record.get("source_id")
        source_labels = record.get("source_labels") or []
        source_props = record.get("source_props") or {}

        # 没有起点无法画这一行
        if source_id is None:
            continue

        source_id = str(source_id)

        if source_id not in node_map:
            node_map[source_id] = Node(
                id=source_id,
                caption=build_caption(
                    source_labels,
                    source_props,
                ),
                size=build_node_size(source_labels),
            )

        target_id = record.get("target_id")
        relationship_type = record.get("relationship_type")
        relationship_id = record.get("relationship_id")

        # OPTIONAL MATCH 允许没有关系，保留单独的 source 节点
        if target_id is None or relationship_type is None:
            continue

        target_id = str(target_id)
        target_labels = record.get("target_labels") or []
        target_props = record.get("target_props") or {}

        if target_id not in node_map:
            node_map[target_id] = Node(
                id=target_id,
                caption=build_caption(
                    target_labels,
                    target_props,
                ),
                size=build_node_size(target_labels),
            )

        # 使用 Neo4j 关系 ID 去重，避免重复记录重复画边
        if relationship_id is None:
            raise ValueError(
                "存在关系，但 relationship_id 为空。"
                "请使用 elementId(rel) AS relationship_id。"
            )

        relationship_id = str(relationship_id)

        if relationship_id not in relationship_ids:
            relationships.append(
                Relationship(
                    source=source_id,
                    target=target_id,
                    caption=str(relationship_type),
                )
            )

            relationship_ids.add(relationship_id)

    return list(node_map.values()), relationships


# ============================================================
# 8. 图谱页面状态
# ============================================================

if "kg_query_editor" not in st.session_state:
    st.session_state.kg_query_editor = DEFAULT_GRAPH_QUERY

if "kg_records" not in st.session_state:
    st.session_state.kg_records = None

if "kg_executed_query" not in st.session_state:
    st.session_state.kg_executed_query = None


def reset_graph_query():
    """恢复默认查询；不自动执行，避免隐藏的数据库请求。"""

    st.session_state.kg_query_editor = DEFAULT_GRAPH_QUERY
    st.session_state.kg_records = None
    st.session_state.kg_executed_query = None


# ============================================================
# 9. 在前面创建的占位区域中绘制 UI
# ============================================================

with graph_visualization_placeholder.container():

    with st.container(border=True):

        st.markdown("### 输入 Cypher 查询")

        st.caption(
            "修改查询条件后，点击执行即可更新图谱。"
            "请保留示例中的 RETURN 字段别名。"
        )

        # 表单保证编辑过程中不会每次输入都执行查询
        with st.form("kg_cypher_query_form"):
            query_text = st.text_area(
                "Cypher",
                key="kg_query_editor",
                height=320,
                help=(
                    "查询需要返回 source、target、"
                    "relationship 对应的固定别名。"
                ),
            )

            submitted = st.form_submit_button(
                "▶ 执行查询并更新图谱",
                type="primary",
                use_container_width=True,
            )

        st.button(
            "恢复默认查询",
            on_click=reset_graph_query,
            use_container_width=True,
        )

        if submitted:
            if not neo4j_ready:
                st.error("Neo4j 尚未连接，请先返回主应用建立连接。")

            elif not query_text.strip():
                st.warning("请输入 Cypher 查询。")

            else:
                try:
                    with st.spinner("正在从 Neo4j 查询图谱数据..."):
                        records = st.session_state.neo4j_graph.query(
                            query_text.strip()
                        )

                        validate_graph_records(records)

                    # 查询成功后再更新状态
                    st.session_state.kg_records = records
                    st.session_state.kg_executed_query = query_text.strip()

                    if not records:
                        st.info("查询成功，但没有返回任何记录。")

                except Exception as e:
                    # 不删除上一次成功的结果
                    st.error(
                        "Cypher 执行失败，或返回字段不符合图谱格式。"
                    )
                    st.exception(e)

        records = st.session_state.kg_records

        if records is not None:
            if st.session_state.kg_executed_query:
                with st.expander("查看当前图谱对应的 Cypher"):
                    st.code(
                        st.session_state.kg_executed_query,
                        language="cypher",
                    )

            if records:
                try:
                    nodes, relationships = convert_records_to_viz_data(
                        records
                    )

                    if not nodes:
                        st.warning(
                            "查询有结果，但没有可绘制的 source 节点。"
                        )
                    else:
                        col1, col2 = st.columns(2)

                        col1.metric("节点数", len(nodes))
                        col2.metric("关系数", len(relationships))

                        visualization_graph = VisualizationGraph(
                            nodes=nodes,
                            relationships=relationships,
                        )

                        rendered_html = visualization_graph.render()

                        components.html(
                            rendered_html.data,
                            height=800,
                            scrolling=True,
                        )

                except Exception as e:
                    st.error("图谱数据转换或渲染失败。")
                    st.exception(e)

