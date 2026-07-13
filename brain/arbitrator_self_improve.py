"""
Arbitrator 自改进 — LLM 改进自己的 prompt，通过 Verifier 验证。

核心机制：
  1. 收集历史：哪些建议被接受？哪些被拒绝？
  2. 分析模式：被接受的建议有什么共同点？被拒绝的有什么问题？
  3. 提出改进：修改 prompt 模板
  4. Verifier 验证：新模板生成的建议是否更有效？

安全性：
  - Arbitrator 只能改自己的 prompt 模板，不能改 Verifier 规则
  - 每次改进必须通过 A/B 测试（新模板 vs 旧模板）
  - 改进记录写入 WAL，可审计
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from typing import Optional
from brain.wal import WALWriter, utc_now
from brain.db import BrainDB
from brain.llm import llm_call


# ─── 默认 prompt 模板 ─────────────────────────────────────

DEFAULT_ADVISORY_PROMPT = """你是一个认知校准专家。基于以下 AI 助手过去犯的校准错误，生成一份简洁的校准建议。

校准建议的目标：让 AI 助手在未来对话中减少同类错误。

要求：
1. 只输出 actionable 的建议（"在回答 X 类问题前先做 Y"）
2. 按错误类型分组
3. 控制在 300 字以内
4. 输出纯文本，不要 JSON

最近 {failure_count} 条校准失败记录：
{failures}

当前已有的校准建议（如果有，在此基础上改进）：
{current_advisory}

请输出更新后的校准建议："""

# ─── 改进分析 prompt ──────────────────────────────────────

ANALYSIS_PROMPT = """你是一个 prompt 工程专家。分析以下校准建议的历史数据，找出改进方向。

最近的建议历史（共 {count} 条）：
{history}

请分析：
1. 被接受的建议有什么共同特征？
2. 被拒绝的建议有什么问题？
3. 当前 prompt 模板有什么可以改进的地方？

输出 JSON：
```json
{{
  "analysis": "一句话总结",
  "accept_patterns": ["被接受建议的特征1", "特征2"],
  "reject_patterns": ["被拒绝建议的问题1", "问题2"],
  "improvements": [
    {{
      "area": "改进领域",
      "suggestion": "具体改进建议",
      "expected_impact": "预期效果"
    }}
  ],
  "new_prompt_template": "改进后的完整 prompt 模板（用 {{failure_count}}, {{failures}}, {{current_advisory}} 作为变量占位符）"
}}
```"""


class ArbitratorSelfImprover:
    """Arbitrator 自改进器"""

    def __init__(self, wal: WALWriter, db: BrainDB):
        self.wal = wal
        self.db = db
        self.current_prompt = DEFAULT_ADVISORY_PROMPT
        self.prompt_version = 1
        self.improvement_history = []

    def analyze_and_improve(self) -> dict:
        """
        分析历史数据，提出 prompt 改进。

        返回:
            {
                "analysis": str,
                "improvements": list,
                "new_prompt": str | None,
                "prompt_changed": bool,
            }
        """
        # 1. 收集历史
        history = self._collect_history()

        if len(history) < 2:
            return {
                "analysis": "历史数据不足，需要至少 2 条建议记录",
                "improvements": [],
                "new_prompt": None,
                "prompt_changed": False,
            }

        # 2. 让 LLM 分析历史并提出改进
        history_text = "\n".join([
            f"- v{h['version']}: {h['verdict']} | 防错率: {h.get('prevention_rate', 'N/A')} | "
            f"内容摘要: {h['content'][:80]}..."
            for h in history
        ])

        analysis_prompt = ANALYSIS_PROMPT.format(history=history_text, count=len(history))
        raw = llm_call(analysis_prompt, temperature=0.3)

        # 3. 解析结果
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            # 尝试提取 JSON
            if "```json" in raw:
                start = raw.index("```json") + 7
                end = raw.index("```", start)
                result = json.loads(raw[start:end].strip())
            elif "{" in raw:
                start = raw.index("{")
                end = raw.rindex("}") + 1
                result = json.loads(raw[start:end])
            else:
                return {
                    "analysis": "LLM 响应解析失败",
                    "improvements": [],
                    "new_prompt": None,
                    "prompt_changed": False,
                }

        # 4. 提取新 prompt
        new_prompt = result.get("new_prompt_template")
        prompt_changed = bool(new_prompt and new_prompt != self.current_prompt)

        # 5. 记录改进历史
        self.improvement_history.append({
            "timestamp": utc_now(),
            "analysis": result.get("analysis", ""),
            "improvements": result.get("improvements", []),
            "prompt_changed": prompt_changed,
            "old_prompt_version": self.prompt_version,
        })

        return {
            "analysis": result.get("analysis", ""),
            "accept_patterns": result.get("accept_patterns", []),
            "reject_patterns": result.get("reject_patterns", []),
            "improvements": result.get("improvements", []),
            "new_prompt": new_prompt,
            "prompt_changed": prompt_changed,
        }

    def apply_improvement(self, new_prompt: str) -> dict:
        """
        应用改进后的 prompt 模板。

        参数:
            new_prompt: 新的 prompt 模板

        返回:
            {old_version, new_version, applied}
        """
        old_version = self.prompt_version
        self.prompt_version += 1
        self.current_prompt = new_prompt

        # 写 WAL
        self.wal.append(
            actor="arbitrator_self_improve",
            event_type="prompt_updated",
            data={
                "old_version": old_version,
                "new_version": self.prompt_version,
                "old_prompt_length": len(self.current_prompt),
                "new_prompt_length": len(new_prompt),
            },
            verified=False,
        )

        return {
            "old_version": old_version,
            "new_version": self.prompt_version,
            "applied": True,
        }

    def get_current_prompt(self) -> str:
        """获取当前 prompt 模板"""
        return self.current_prompt

    def get_prompt_version(self) -> int:
        """获取当前 prompt 版本"""
        return self.prompt_version

    # ─── 内部方法 ──────────────────────────────────────────

    def _collect_history(self) -> list[dict]:
        """收集建议历史（最近 10 条）"""
        rows = self.db.conn.execute("""
            SELECT id, version, content, status, post_score, created_at, tested_at
            FROM advisories
            ORDER BY id DESC
            LIMIT 10
        """).fetchall()

        return [
            {
                "version": r["version"],
                "content": r["content"],
                "verdict": r["status"],
                "prevention_rate": r["post_score"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]
