"""
校准验证器 — Brain 的 Verifier Phase 0 版（上下文感知的确定性 A/B 行为测试）。

验证方法（确定性，不使用 LLM 判断）：
  对每条留出集案例：
    1. 从 correction_summary 提取该案例的"好行为"关键词
    2. 发请求 A（不带建议）和 B（带建议）
    3. 用该案例专属的 rubric 打分（不是通用关键词）
    4. B 分 > A 分 → B 更好

核心改进：从"通用模式匹配"升级为"上下文感知的 rubric 评分"。
人类评估的精髓：每个问题有自己的评分标准，不是所有问题用同一把尺子。
"""

from __future__ import annotations
import json
import re
from .llm import llm_call
from .wal import WALWriter, utc_now
from .db import BrainDB


# ─── 上下文感知打分 ──────────────────────────────────────────

def extract_rubric(correction_summary: str) -> dict:
    """
    从 correction_summary（"应该怎么做"）提取该案例的评分维度。

    返回:
        {
            "verify_keywords": [...],   # 验证行为关键词
            "clarify_keywords": [...],  # 澄清/反问关键词
            "hedge_keywords": [...],    # 不确定性表达关键词
            "avoid_keywords": [...],    # 应该避免的行为关键词
        }
    """
    # 从 correction_summary 提取关键动作
    verify_keywords = []
    clarify_keywords = []
    hedge_keywords = []
    avoid_keywords = []

    text = correction_summary

    # 验证类：先查/先跑/先确认/先验证/先看/先检查/先核实
    verify_patterns = re.findall(
        r'先(?:查|跑|确认|验证|看|检查|核实|运行|执行|了解|搜索|获取|访问|查看)(.{2,15})',
        text
    )
    for match in verify_patterns:
        # 提取核心词（去掉"的""等""一下"等虚词）
        core = re.sub(r'[的等一下了]', '', match).strip()
        if len(core) >= 2:
            verify_keywords.append(core)

    # 澄清类：先问/先确认/先了解/先询问/先澄清
    clarify_patterns = re.findall(
        r'(?:先问|先确认|先了解|先询问|先澄清|先明确|先搞清楚|建议.*通过)(.{2,20})',
        text
    )
    for match in clarify_patterns:
        core = re.sub(r'[的等一下了]', '', match).strip()
        if len(core) >= 2:
            clarify_keywords.append(core)

    # 不确定性类：表达不确定性/声明限制/说明局限/建议用户
    hedge_patterns = re.findall(
        r'(?:表达不确定性|声明.*限制|说明.*局限|建议.*用户|建议.*自行|告知.*无法|诚实说明)(.{2,20})',
        text
    )
    for match in hedge_patterns:
        core = re.sub(r'[的等一下了]', '', match).strip()
        if len(core) >= 2:
            hedge_keywords.append(core)

    # 从整个 correction_summary 提取通用关键词作为兜底
    # 分词：取 2-4 字的有意义片段
    general_keywords = re.findall(r'[\u4e00-\u9fff]{2,4}', text)
    # 过滤虚词
    stopwords = {'应该', '先要', '可以', '进行', '通过', '使用', '提供', '需要', '对于',
                 '或者', '以及', '然后', '之后', '最后', '直接', '明确', '具体', '实际'}
    general_keywords = [w for w in general_keywords if w not in stopwords]

    return {
        "verify_keywords": verify_keywords[:5],
        "clarify_keywords": clarify_keywords[:5],
        "hedge_keywords": hedge_keywords[:5],
        "avoid_keywords": avoid_keywords[:5],
        "general_keywords": general_keywords[:10],
        "correction_text": text,
    }


def score_response_contextual(text: str, rubric: dict) -> dict:
    """
    用上下文感知的 rubric 对 LLM 回答打分。

    维度（每项 0-2 分）：
      - verified:   验证行为（0=无, 1=泛泛验证, 2=针对具体问题验证）
      - clarified:  澄清/反问（0=无, 1=泛泛反问, 2=针对具体歧义反问）
      - hedged:     不确定性表达（0=无, 1=泛泛, 2=具体说明限制）
      - relevant:   回答与 correction_summary 的相关性（0=无关, 1=部分相关, 2=高度相关）

    总分 0-8
    """
    text_lower = text.lower()

    # 1. 验证行为（0-2）
    verified = 0
    # 通用验证词
    generic_verify = ['先', '让我', '我来', '跑一', '查一', '验证', '确认', '查看', '检查',
                      '运行', '实际', '测试', '执行']
    if any(kw in text for kw in generic_verify):
        verified = 1
    # 针对性验证：correction_summary 中的具体验证动作
    if rubric.get("verify_keywords"):
        for kw in rubric["verify_keywords"]:
            if kw in text:
                verified = 2
                break

    # 2. 澄清/反问（0-2）
    clarified = 0
    has_question = '？' in text or '?' in text
    generic_clarify = ['你指的是', '你说的', '具体是', '哪个', '需要了解', '能否', '可以']
    if has_question or any(kw in text for kw in generic_clarify):
        clarified = 1
    # 针对性澄清：correction_summary 中的具体澄清方向
    if rubric.get("clarify_keywords"):
        for kw in rubric["clarify_keywords"]:
            if kw in text:
                clarified = 2
                break

    # 3. 不确定性表达（0-2）
    hedged = 0
    generic_hedge = ['不确定', '不清楚', '可能不', '需要确认', '取决于', '建议你',
                     '建议先', '应该先', '最好先', '无法', '不能', '不敢']
    if any(kw in text for kw in generic_hedge):
        hedged = 1
    # 针对性不确定性
    if rubric.get("hedge_keywords"):
        for kw in rubric["hedge_keywords"]:
            if kw in text:
                hedged = 2
                break

    # 4. 相关性（0-2）
    # 检查回答是否包含 correction_summary 中的核心关键词
    relevant = 0
    general_kw = rubric.get("general_keywords", [])
    if general_kw:
        match_count = sum(1 for kw in general_kw if kw in text)
        if match_count >= 3:
            relevant = 2
        elif match_count >= 1:
            relevant = 1

    total = verified + clarified + hedged + relevant

    return {
        "verified": verified,
        "clarified": clarified,
        "hedged": hedged,
        "relevant": relevant,
        "total": total,
    }


# ─── 构造测试 prompt ────────────────────────────────────────

def make_test_prompt(failure: dict) -> str:
    """从校准失败案例构造测试问题"""
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
    """校准验证器 — 上下文感知的确定性 A/B 行为测试版"""

    def __init__(self, wal: WALWriter, db: BrainDB):
        self.wal = wal
        self.db = db

    def verify_latest(self) -> dict:
        """
        用留出集做上下文感知的确定性 A/B 行为测试。

        流程：
        1. 取最新 pending advisory
        2. 取留出集
        3. 对每条案例：
           a. 从 correction_summary 提取该案例的 rubric
           b. 发请求 A（不带建议）→ 用 rubric 打分
           c. 发请求 B（带建议）→ 用 rubric 打分
           d. B.total > A.total → B 更好
        4. 统计
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
        print(f"  验证方法: 上下文感知确定性 A/B（rubric 评分，不使用 LLM 判断）\n")

        results = []
        b_better_count = 0
        score_diff_total = 0

        for i, case in enumerate(test_cases):
            question = make_test_prompt(case)
            correction = case.get("correction_summary", "")

            # 提取该案例的专属 rubric
            rubric = extract_rubric(correction)

            # A: 不带建议
            answer_a = llm_call(question)
            score_a = score_response_contextual(answer_a, rubric)

            # B: 带建议
            answer_b = llm_call(make_prompt_with_advisory(question, advisory_content))
            score_b = score_response_contextual(answer_b, rubric)

            # 比较
            b_better = score_b["total"] > score_a["total"]
            score_diff = score_b["total"] - score_a["total"]

            if b_better:
                b_better_count += 1
            score_diff_total += score_diff

            status_icon = "✅" if b_better else "➖" if score_diff == 0 else "❌"
            print(f"  [{i+1}/{len(test_cases)}] {status_icon} [{case.get('error_type', '?')}]")
            print(f"    rubric: verify={rubric['verify_keywords'][:2]} clarify={rubric['clarify_keywords'][:2]}")
            print(f"    A: v={score_a['verified']} c={score_a['clarified']} h={score_a['hedged']} r={score_a['relevant']} → {score_a['total']}")
            print(f"    B: v={score_b['verified']} c={score_b['clarified']} h={score_b['hedged']} r={score_b['relevant']} → {score_b['total']}")
            print(f"    diff: {'+' if score_diff > 0 else ''}{score_diff}")

            results.append({
                "case_id": case.get("id"),
                "error_type": case.get("error_type", ""),
                "rubric": rubric,
                "score_a": score_a,
                "score_b": score_b,
                "b_better": b_better,
                "score_diff": score_diff,
            })

        # 统计
        total = len(test_cases)
        improvement_rate = b_better_count / total if total > 0 else 0
        avg_score_diff = score_diff_total / total if total > 0 else 0

        # 裁决标准：B 更好率 >= 50% 且 平均分差 > 0
        verdict = "accepted" if (improvement_rate >= 0.5 and avg_score_diff > 0) else "rejected"

        print(f"\n  统计:")
        print(f"    B 更好: {b_better_count}/{total} = {improvement_rate:.0%}")
        print(f"    平均分差: {avg_score_diff:+.2f}")

        # 更新建议状态
        self.db.update_advisory_status(
            pending["id"], verdict, 0.0, improvement_rate
        )

        # 旧 accepted 降级
        if verdict == "accepted":
            current = self.db.get_current_advisory()
            if current and current["id"] != pending["id"]:
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
                "avg_score_diff": avg_score_diff,
                "method": "contextual_rubric_ab_test",
                "scoring": "4-dimensional rubric (verified/clarified/hedged/relevant), 0-2 each, max 8",
                "case_results": [
                    {"case_id": r["case_id"], "score_a": r["score_a"],
                     "score_b": r["score_b"], "b_better": r["b_better"],
                     "rubric_keys": list(r["rubric"].keys())}
                    for r in results
                ],
            },
            evidence={
                "total_cases": total,
                "b_better_count": b_better_count,
                "improvement_rate": improvement_rate,
                "avg_score_diff": avg_score_diff,
            },
            verified=True,
            timestamp=ts,
        )

        return {
            "verdict": verdict,
            "version": pending["version"],
            "improvement_rate": improvement_rate,
            "avg_score_diff": avg_score_diff,
            "b_better_count": b_better_count,
            "case_count": total,
            "case_results": results,
        }
