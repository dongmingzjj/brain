"""
校准建议生成器 — Brain 的 Arbitrator Phase 0 版。

读取最近的校准失败记录，生成一份校准建议（advisory）。
目标是让 AI 助手在未来对话中减少同类错误。
"""

from __future__ import annotations
import json
from .llm import llm_call
from .wal import WALWriter, utc_now
from .db import BrainDB


ARBITRATOR_PROMPT = """你是一个认知校准专家。基于以下 AI 助手过去犯的校准错误，生成一份简洁的校准建议。

校准建议的目标：让 AI 助手在未来对话中减少同类错误。

要求：
1. 只输出 actionable 的建议（"在回答 X 类问题前先做 Y"）
2. 按错误类型分组
3. 控制在 300 字以内
4. 输出纯文本，不要 JSON，不要 Markdown 标题

最近 {failure_count} 条校准失败记录：
{failures}

当前已有的校准建议（如果有，在此基础上改进）：
{current_advisory}

请输出更新后的校准建议："""


class Arbitrator:
    """校准建议生成器"""

    def __init__(self, wal: WALWriter, db: BrainDB):
        self.wal = wal
        self.db = db

    def propose_advisory(self) -> dict:
        """
        读取最近失败，生成新的校准建议。

        返回:
            {seq, content, version, failure_count}
        """
        failures = self.db.get_training_failures(limit=30)

        if not failures:
            return {"status": "skipped", "reason": "no failures to learn from"}

        current = self.db.get_current_advisory()
        current_version = current["version"] if current else 0

        # 格式化失败记录给 LLM
        failures_text = "\n".join([
            f"- [{f['error_type']}] Q: {f.get('question_summary', '?')}"
            f" → AI错误地: {f.get('wrong_answer_summary', '?')}"
            f" → 应该: {f.get('correction_summary', '?')}"
            for f in failures
        ])

        current_text = current["content"] if current else "（无）"

        prompt = ARBITRATOR_PROMPT.format(
            failure_count=len(failures),
            failures=failures_text,
            current_advisory=current_text,
        )

        new_content = llm_call(prompt)

        # 写 WAL
        ts = utc_now()
        seq = self.wal.append(
            actor="arbitrator",
            event_type="advisory_proposed",
            data={"version": current_version + 1, "content": new_content.strip()},
            verified=False,
            timestamp=ts,
        )

        # 同步 events 表（满足外键约束）
        self.db.index_event({
            "seq": seq,
            "timestamp": ts,
            "actor": "arbitrator",
            "event_type": "advisory_proposed",
            "data": {"version": current_version + 1, "content": new_content.strip()},
            "evidence": None,
            "verified": False,
        })

        # 同步 advisories 表
        self.db.add_advisory(
            seq=seq,
            version=current_version + 1,
            content=new_content.strip(),
            created_at=ts,
        )

        return {
            "seq": seq,
            "content": new_content.strip(),
            "version": current_version + 1,
            "failure_count": len(failures),
        }
