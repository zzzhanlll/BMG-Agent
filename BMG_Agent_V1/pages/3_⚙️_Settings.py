import streamlit as st
import os
import ipaddress
from urllib.parse import urlparse
import httpx


### 辅助函数
def validate_base_url(base_url: str) -> tuple[bool, str]:
    """仅进行 URL 和 SSRF 基础防护校验。"""
    base_url = base_url.strip().rstrip("/")
    try:
        parsed = urlparse(base_url)
    except Exception:
        return False, "API URL 格式不正确"
    if parsed.scheme not in ("https", "http"):
        return False, "API URL 必须以 http:// 或 https:// 开头"
    if not parsed.hostname:
        return False, "API URL 缺少主机名"
    hostname = parsed.hostname.lower()
    # 生产环境通常建议仅允许 HTTPS
    # if parsed.scheme != "https":
    #     return False, "生产环境仅允许 HTTPS API 地址"
    blocked_hosts = {"localhost", "0.0.0.0", "metadata.google.internal"}
    if hostname in blocked_hosts or hostname.endswith(".localhost"):
        return False, "不允许使用 localhost 或本地地址"
    # 阻止直接填写的内网/IP 地址
    try:
        ip = ipaddress.ip_address(hostname)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
        ):
            return False, "不允许使用内网、本机或保留 IP 地址"
    except ValueError:
        # hostname 是域名，不是 IP；生产环境还应在服务端做 DNS 解析后二次校验
        pass
    return True, base_url

def test_openai_compatible_config(
    base_url: str,
    api_key: str,
    model: str,
) -> tuple[bool, str]:
    """测试 OpenAI-compatible 的 API 配置。"""
    ok, result = validate_base_url(base_url)
    if not ok:
        return False, result
    base_url = result
    api_key = api_key.strip()
    model = model.strip()
    if len(api_key) < 10:
        return False, "API Key 不能为空，且长度似乎不正确"
    if not model:
        return False, "Model 名称不能为空"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}]
    }
    timeout = httpx.Timeout(
        timeout=15.0,
        connect=5.0,
    )
    try:
        with httpx.Client(timeout=timeout, follow_redirects=False) as client:
            response = client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
        if 200 <= response.status_code < 300:
            return True, "连接成功，API Key 和模型均可用"
        # 不要把 Key 或完整底层报错直接回显
        if response.status_code == 401:
            return False, "认证失败：API Key 无效或已过期"
        if response.status_code == 403:
            return False, "无访问权限：请检查 API Key 权限或模型授权"
        if response.status_code == 404:
            return False, "接口地址或模型不存在：请检查 Base URL 和 Model 名称"
        if response.status_code == 429:
            return False, "请求受限：可能是限流、余额不足或配额耗尽"
        if 500 <= response.status_code < 600:
            return False, "模型服务暂时异常，请稍后重试"
        return False, f"测试失败，服务返回 HTTP {response.status_code}"
    except httpx.ConnectTimeout:
        return False, "连接超时：请检查网络、域名、端口或代理"
    except httpx.ReadTimeout:
        return False, "响应超时：模型服务响应过慢或暂不可用"
    except httpx.ConnectError:
        return False, "无法连接 API 服务：请检查 URL 和网络"
    except httpx.HTTPError:
        return False, "请求 API 时发生网络错误"

# Settings
st.title("⚙️ Settings")

model_settings, path_settings = st.tabs(
    ["Model Settings", "Path Settings"]
)
with model_settings:

    st.subheader("Model Settings")
    st.caption("- Applies only to OpenAI-compatible methods")

    base_url = st.text_input(
        "🚋 API Base URL:",
        value=st.session_state.get("base_url", ""),
        placeholder="https://openrouter.ai/api/v1"
    )

    if base_url:
        try:
            url_valid, url_message = validate_base_url(base_url)
            if not url_valid:
                st.warning(url_message)
        except Exception:
            pass

    api_key = st.text_input(
        "🔑 API Key:",
        type="password",
        value=st.session_state.get("api_key", ""),
        placeholder="enter API Key"
    )

    model = st.text_input(
        "🌎 Model:",
        value=st.session_state.get("model", ""),
        placeholder="deepseek/deepseek-v3.2"
    )

    left_col, _, right_col = st.columns([1, 2, 1])
    with left_col:
        if st.button("Test Connection", type="secondary", use_container_width=True):
            with st.spinner("Verifying configuration..."):
                success, message = test_openai_compatible_config(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                )
            if success:
                st.success(message)
                st.session_state.api_key = api_key
                st.session_state.base_url = base_url
                st.session_state.model = model
                st.session_state.model_config = True
            else:
                st.error(message)

    with right_col:
        if st.button("Clear settings", type="primary", use_container_width=True):
            st.session_state.api_key = ""
            st.session_state.base_url = ""
            st.session_state.model = ""
            st.session_state.model_config = False


    # 显示当前配置状态
    if st.session_state.get("model_config"):
        st.info("✅ AI Model Connected Successfully")
    else:
        st.warning("⚠️ No AI model is connected")

    pass

with path_settings:
    st.subheader("Path Settings")

    file_path = st.text_input(
        "File Path:",
        value=st.session_state.get("file_path"),
        help="Path where data is saved",
        placeholder=f"Current path：{st.session_state.file_path}",
    )

    if os.path.exists(file_path):
        st.session_state.file_path = file_path
    else:
        st.warning("⚠️ Path error, please input a valid file path!")

st.markdown("---")
st.markdown("### 通过OpenRouter获取模型 (推荐)：")
st.info("🔗 **[OpenRouter 官网](https://openrouter.ai)**")
