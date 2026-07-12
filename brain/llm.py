"""
LLM 调用封装 — 复用 Hermes 的 custom provider 配置。

Phase 0 直接用智谱 GLM API（OpenAI 兼容格式）。
"""

from __future__ import annotations
import json
import time
import urllib.request
from typing import Optional


# ─── 默认配置 ──────────────────────────────

# 优先使用 xiaomimimo（GLM 配额耗尽时的备用）
DEFAULT_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
DEFAULT_MODEL = "mimo-v2.5-pro"
DEFAULT_API_KEY = "tp-c4xcionjc5b9v80v06yw5wb07qf6myl1tf961e1r9jkrwm1s"

# 速率控制
_REQUEST_INTERVAL = 2.0  # 每次请求至少间隔 2 秒
_last_request_time = 0.0


def _rate_limit():
    """简单速率限制：两次请求之间至少间隔 _REQUEST_INTERVAL 秒"""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < _REQUEST_INTERVAL:
        time.sleep(_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.time()


def llm_call(prompt: str,
             base_url: str = DEFAULT_BASE_URL,
             api_key: str = DEFAULT_API_KEY,
             model: str = DEFAULT_MODEL,
             temperature: float = 0.3,
             timeout: int = 60,
             max_retries: int = 3) -> str:
    """
    调用 LLM，返回纯文本响应。

    Phase 0 简化版：只支持单轮 prompt → response。
    内置速率限制和 429 重试。
    """
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    data = json.dumps(payload).encode("utf-8")

    for attempt in range(max_retries):
        _rate_limit()

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
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = min(2 ** attempt * 3, 30)  # 3s, 6s, 12s
                print(f"  [429] 限流，等待 {wait}s 后重试 ({attempt+1}/{max_retries})...")
                time.sleep(wait)
                continue
            return f"[LLM_ERROR] HTTP {e.code}: {e.reason}"
        except Exception as e:
            return f"[LLM_ERROR] {e}"

    return "[LLM_ERROR] 最大重试次数已达"


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
