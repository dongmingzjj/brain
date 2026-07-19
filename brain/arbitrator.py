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

    def __init__(self, wal: WALWriter, db: BrainDB, mem_executor=None, prompt_template=None):
        """
        mem_executor: Memory Region 执行器（可选，用于增强检索）
        prompt_template: 自定义 prompt 模板（可选，用于自改进）
        """
        self.wal = wal
        self.db = db
        self.mem_executor = mem_executor
        self._prompt_template = prompt_template

    def propose_advisory(self) -> dict:
        """
        读取最近失败，生成新的校准建议。

        如果 Memory Region 可用，会同时从 Memory Region 检索相关失败，
        补充 flat 查询可能遗漏的历史模式。

        返回:
            {seq, content, version, failure_count}
        """
        # 从 flat DB 取训练集
        failures = self.db.get_training_failures(limit=30)

        if not failures:
            return {"status": "skipped", "reason": "no failures to learn from"}

        # 如果 Memory Region 可用，检索相关失败补充
        memory_failures = []
        if self.mem_executor:
            # 用所有失败的错误类型作为查询
            error_types = list(set(f.get("error_type", "") for f in failures if f.get("error_type")))
            for etype in error_types[:3]:  # 最多查 3 种类型
                results = self.mem_executor.retrieve(etype, top_k=5, mem_type="calibration_failure")
                for r in results:
                    try:
                        import json
                        data = json.loads(r["value"])
                        # 去重（和 flat DB 不重复）
                        if not any(f.get("question_summary") == data.get("question_summary") for f in failures):
                            memory_failures.append(data)
                    except (json.JSONDecodeError, KeyError):
                        continue

        # 合并
        all_failures = failures + memory_failures
        failure_count = len(all_failures)

        current = self.db.get_current_advisory()
        current_version = current["version"] if current else 0

        # 格式化失败记录给 LLM
        failures_text = "\n".join([
            f"- [{f.get('error_type', '?')}] Q: {f.get('question_summary', '?')}"
            f" → AI错误地: {f.get('wrong_answer_summary', '?')}"
            f" → 应该: {f.get('correction_summary', '?')}"
            for f in all_failures
        ])

        current_text = current["content"] if current else "（无）"

        # 使用自定义 prompt 或默认模板
        template = self._prompt_template or ARBITRATOR_PROMPT
        prompt = template.format(
            failure_count=failure_count,
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
