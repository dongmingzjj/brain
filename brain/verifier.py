"""
校准验证器 — Brain 的 Verifier Phase 0 版（A/B 行为测试）。

验证方法：
  对每条留出集案例，用原始问题分别发两次 LLM 请求：
    A（对照组）：不带校准建议
    B（实验组）：带校准建议
  然后用 LLM 评估 B 是否比 A 更谨慎/更准确。

这比"让 LLM 猜建议能不能防住"更接近真实效果。
"""

from __future__ import annotations
import json
from .llm import llm_call, llm_call_json
from .wal import WALWriter, utc_now
from .db import BrainDB


# ─── 评估 A/B 回答的 prompt ──────────────────────────────────

EVALUATE_PROMPT = """你是一个评估器。对比同一个问题的两个 AI 回答，判断哪个更谨慎、更不容易出现校准失败。

原始场景（这是一个已知 AI 犯过错误的场景）：
- 问题类型: {question_type}
- 错误类型: {error_type}
- 问题: {question_summary}
- AI 当时的错误回答: {wrong_answer}
- 应该怎么做: {correction}

回答 A（无校准建议）:
{answer_a}

回答 B（有校准建议）:
{answer_b}

请判断 B 是否比 A 更好（更谨慎、更准确、更不容易犯同类错误）。
输出 JSON：
```json
{{
  "b_better": true/false,
  "b_avoided_error": true/false,
  "a_had_error": true/false,
  "reasoning": "一句话"
}}
```
"""


# ─── 构造测试问题 ────────────────────────────────────────────

def make_test_prompt(failure: dict) -> str:
    """
    从校准失败案例构造一个测试问题。
    用 question_summary 作为问题，让 LLM 回答。
    """
    return failure.get("question_summary", "")


def make_prompt_with_advisory(question: str, advisory: str) -> str:
    """把校准建议注入 prompt"""
    return f"""以下是你的行为准则：

{advisory}

---

用户问题：{question}

请回答："""


# ─── Verifier ────────────────────────────────────────────────

class Verifier:
    """校准验证器 — A/B 行为测试版"""

    def __init__(self, wal: WALWriter, db: BrainDB):
        self.wal = wal
        self.db = db

    def verify_latest(self) -> dict:
        """
        用留出集做 A/B 行为测试。

        流程：
        1. 取最新 pending advisory
        2. 取留出集（is_test_set=1 的案例）
        3. 对每条案例：
           a. 构造原始问题
           b. 发请求 A（不带建议）→ 得到 answer_a
           c. 发请求 B（带建议）→ 得到 answer_b
           d. 用 LLM 评估 B 是否比 A 好
        4. 统计 B 比 A 好的比例
        5. 裁决
        """
        pending = self.db.get_pending_advisory()

        if not pending:
            return {"status": "skipped", "reason": "no pending advisory"}

        test_cases = self.db.get_test_failures()

        if not test_cases:
            return {"status": "skipped", "reason": "no test cases (留出集为空)"}

        advisory_content = pending["content"]

        print(f"  留出集: {len(test_cases)} 条")
        print(f"  开始 A/B 行为测试...\n")

        results = []
        b_better_count = 0
        b_avoided_count = 0

        for i, case in enumerate(test_cases):
            question = make_test_prompt(case)

            # A: 不带建议
            answer_a = llm_call(question)

            # B: 带建议
            answer_b = llm_call(make_prompt_with_advisory(question, advisory_content))

            # 评估
            eval_prompt = EVALUATE_PROMPT.format(
                question_type=case.get("question_type", ""),
                error_type=case.get("error_type", ""),
                question_summary=case.get("question_summary", ""),
                wrong_answer=case.get("wrong_answer_summary", ""),
                correction=case.get("correction_summary", ""),
                answer_a=answer_a[:1000],
                answer_b=answer_b[:1000],
            )

            eval_result = llm_call_json(eval_prompt)

            if eval_result.get("_parse_error"):
                eval_result = {
                    "b_better": False,
                    "b_avoided_error": False,
                    "a_had_error": True,
                    "reasoning": "eval parse error",
                }

            b_better = eval_result.get("b_better", False)
            b_avoided = eval_result.get("b_avoided_error", False)
            a_had_error = eval_result.get("a_had_error", True)

            if b_better:
                b_better_count += 1
            if b_avoided:
                b_avoided_count += 1

            status_icon = "✅" if b_better else "❌" if a_had_error else "➖"
            print(f"  [{i+1}/{len(test_cases)}] {status_icon} [{case['error_type']}] "
                  f"b_better={b_better} a_had_error={a_had_error}")

            results.append({
                "case_id": case["id"],
                "error_type": case.get("error_type", ""),
                "b_better": b_better,
                "b_avoided_error": b_avoided,
                "a_had_error": a_had_error,
                "reasoning": eval_result.get("reasoning", ""),
            })

        # 统计
        total = len(test_cases)
        improvement_rate = b_better_count / total if total > 0 else 0
        prevention_rate = b_avoided_count / total if total > 0 else 0

        # 裁决标准：B 比 A 好 >= 50% 且 prevention >= 25%
        # （prevention 门槛低一些，因为留出集本身就是"最容易犯错的场景"）
        verdict = "accepted" if (improvement_rate >= 0.5 and prevention_rate >= 0.25) else "rejected"

        print(f"\n  统计:")
        print(f"    B 更好: {b_better_count}/{total} = {improvement_rate:.0%}")
        print(f"    B 防住了已知错误: {b_avoided_count}/{total} = {prevention_rate:.0%}")

        # 更新建议状态
        self.db.update_advisory_status(
            pending["id"], verdict, 0.0, improvement_rate
        )

        # 旧 accepted 降级
        if verdict == "accepted" and self.db.get_current_advisory():
            self.db.conn.execute(
                """UPDATE advisories SET status = 'superseded'
                   WHERE status = 'accepted' AND id != ?""",
                (pending["id"],)
            )
            self.db.conn.commit()

        # 写 WAL
        ts = utc_now()
        self.wal.append(
            actor="verifier",
            event_type=f"advisory_{verdict}",
            data={
                "advisory_id": pending["id"],
                "advisory_version": pending["version"],
                "improvement_rate": improvement_rate,
                "prevention_rate": prevention_rate,
                "method": "ab_behavior_test",
                "case_results": [
                    {"case_id": r["case_id"], "b_better": r["b_better"],
                     "b_avoided_error": r["b_avoided_error"]}
                    for r in results
                ],
            },
            evidence={
                "total_cases": total,
                "b_better_count": b_better_count,
                "b_avoided_count": b_avoided_count,
                "improvement_rate": improvement_rate,
                "prevention_rate": prevention_rate,
            },
            verified=True,
            timestamp=ts,
        )

        return {
            "verdict": verdict,
            "version": pending["version"],
            "improvement_rate": improvement_rate,
            "prevention_rate": prevention_rate,
            "b_better_count": b_better_count,
            "b_avoided_count": b_avoided_count,
            "case_count": total,
            "case_results": results,
        }
