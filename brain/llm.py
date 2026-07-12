"""
LLM 调用封装 — 复用 Hermes 的 custom provider 配置。

Phase 0 直接用智谱 GLM API（OpenAI 兼容格式）。
"""

from __future__ import annotations
import json
import urllib.request
from typing import Optional


# ─── 默认配置（从 Hermes config.yaml 读取） ──────────────────

DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/coding/paas/v4"
DEFAULT_MODEL = "glm-5.2"
DEFAULT_API_KEY = "18fee0e349ef48269fa45ded6c61b7e2.OE5nWnSh93ySUnfM"


def llm_call(prompt: str,
             base_url: str = DEFAULT_BASE_URL,
             api_key: str = DEFAULT_API_KEY,
             model: str = DEFAULT_MODEL,
             temperature: float = 0.3,
             timeout: int = 60) -> str:
    """
    调用 LLM，返回纯文本响应。

    Phase 0 简化版：只支持单轮 prompt → response。
    不支持 system prompt、不支持 streaming、不支持 function calling。
    """
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[LLM_ERROR] {e}"


def llm_call_json(prompt: str, **kwargs) -> dict:
    """
    调用 LLM 并解析 JSON 响应。

    如果 LLM 返回的文本包含 ```json 代码块，会自动提取。
    """
    raw = llm_call(prompt, **kwargs)

    # 尝试直接解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 尝试从 ```json ... ``` 提取
    if "```json" in raw:
        start = raw.index("```json") + 7
        end = raw.index("```", start)
        try:
            return json.loads(raw[start:end].strip())
        except json.JSONDecodeError:
            pass

    # 尝试从 { ... } 提取
    if "{" in raw and "}" in raw:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass

    return {"_parse_error": True, "raw": raw[:500]}
