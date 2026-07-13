"""
跨区评估 — Region 之间互相打标签。

核心原则：
  - Region B 不改 Region A 的输出，只给 A 的输出打标签
  - 标签 = {confidence, recommendation, reason}
  - 标签一致 → 直接决策
  - 标签冲突 → 升级到 Arbitrator

Phase 2.0 实现两种评估方向：
  1. Memory → Action：基于历史经验评估 Action 的执行结果
  2. Action → Memory：基于执行结果验证 Memory 的检索建议

不实现的：
  - 多数投票（只有 2 个 Region，没有投票的必要）
  - Arbitrator 自动仲裁（留给 Phase 3）
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from brain.event_bus import EventBus, Signal


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Tag:
    """跨区评估标签"""
    source_region: str              # 打标签的 Region
    target_region: str              # 被评估的 Region
    confidence: float               # 0-1（1=高度认可，0=强烈质疑）
    recommendation: str             # "approve" | "warn" | "reject"
    reason: str                     # 一句话理由
    timestamp: str = field(default_factory=utc_now)
    task_id: str = ""

    def to_dict(self) -> dict:
        return {
            "source": self.source_region,
            "target": self.target_region,
            "confidence": self.confidence,
            "recommendation": self.recommendation,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "task_id": self.task_id,
        }


class CrossRegionEvaluator:
    """跨区评估器"""

    def __init__(self, bus: EventBus,
                 mem_executor=None,
                 action_executor=None):
        """
        bus: Event Bus 实例
        mem_executor: Memory Region 的 executor（可选）
        action_executor: Action Region 的 executor（可选）
        """
        self.bus = bus
        self.mem = mem_executor
        self.action = action_executor
        self._tag_log: list[Tag] = []

    # ─── Memory → Action 评估 ──────────────────────────────

    def memory_evaluates_action(self, action_result: dict,
                                task_id: str = "") -> Tag:
        """
        Memory Region 评估 Action 的执行结果。

        策略：
        1. 用 Action 的 tool + command 检索 Memory 中的历史
        2. 如果有失败历史 → 降低 confidence
        3. 如果只有成功历史 → 高 confidence
        4. 如果没有历史 → 中性 confidence

        返回:
            Tag 标签
        """
        tool = action_result.get("tool", "")
        command = action_result.get("command", "")
        success = action_result.get("success", False)

        confidence = 0.5  # 默认中性
        recommendation = "approve"
        reason = "无历史参考"

        if self.mem:
            # 从 Memory 检索类似工具的历史
            results = self.mem.retrieve(
                f"action {tool} execute",
                top_k=5,
                mem_type="action_history",
            )

            if results:
                # 有历史记录
                # 分析历史中的成功/失败模式
                import json
                failures = 0
                successes = 0
                for r in results:
                    try:
                        data = json.loads(r["value"])
                        if data.get("success"):
                            successes += 1
                        else:
                            failures += 1
                    except (json.JSONDecodeError, KeyError):
                        continue

                total = successes + failures
                if total > 0:
                    fail_rate = failures / total

                    if fail_rate > 0.5:
                        confidence = 0.3
                        recommendation = "warn"
                        reason = f"历史失败率 {fail_rate:.0%}（{failures}/{total}）"
                    elif fail_rate > 0.2:
                        confidence = 0.6
                        recommendation = "approve"
                        reason = f"历史失败率较低 {fail_rate:.0%}（{failures}/{total}）"
                    else:
                        confidence = 0.9
                        recommendation = "approve"
                        reason = f"历史成功率高（{successes}/{total}）"
            else:
                reason = "无此类工具的历史记录"

        tag = Tag(
            source_region="memory",
            target_region="action",
            confidence=confidence,
            recommendation=recommendation,
            reason=reason,
            task_id=task_id,
        )

        # 记录标签
        self._tag_log.append(tag)

        # 通过 Event Bus 发布标签信号
        self.bus.publish(Signal(
            source="region:memory",
            type="action_evaluation",
            task_id=task_id,
            content=tag.to_dict(),
        ))

        return tag

    # ─── Action → Memory 评估 ──────────────────────────────

    def action_evaluates_memory(self, memory_result: dict,
                                task_id: str = "") -> Tag:
        """
        Action Region 评估 Memory 的检索结果。

        策略：
        1. Memory 返回了 N 条结果
        2. Action 检查：这些结果中的建议是否可执行？
        3. 如果 Memory 的建议包含可执行的命令 → 高 confidence
        4. 如果 Memory 的建议太模糊 → 低 confidence

        返回:
            Tag 标签
        """
        relevance = memory_result.get("score", 0)
        result_count = len(memory_result.get("results", []))

        confidence = 0.5
        recommendation = "approve"
        reason = ""

        if result_count == 0:
            confidence = 0.2
            recommendation = "warn"
            reason = "Memory 无匹配结果"
        elif relevance > 0.5:
            confidence = 0.8
            reason = f"高相关性（{relevance:.2f}）"
        elif relevance > 0.2:
            confidence = 0.5
            reason = f"中等相关性（{relevance:.2f}）"
        else:
            confidence = 0.3
            recommendation = "warn"
            reason = f"低相关性（{relevance:.2f}）"

        tag = Tag(
            source_region="action",
            target_region="memory",
            confidence=confidence,
            recommendation=recommendation,
            reason=reason,
            task_id=task_id,
        )

        self._tag_log.append(tag)

        self.bus.publish(Signal(
            source="region:action",
            type="memory_evaluation",
            task_id=task_id,
            content=tag.to_dict(),
        ))

        return tag

    # ─── 决策聚合 ──────────────────────────────────────────

    def make_decision(self, tags: list[Tag]) -> dict:
        """
        汇总多个标签做决策。

        Phase 2.0 简单规则：
        - 所有标签都 approve → 直接执行
        - 有任何 warn → 执行但记录警告
        - 有任何 reject → 不执行，升级到 Arbitrator

        返回:
            {decision: "execute"/"warn"/"escalate", tags: [...], summary}
        """
        if not tags:
            return {"decision": "execute", "reason": "无标签，默认通过"}

        recommendations = [t.recommendation for t in tags]
        avg_confidence = sum(t.confidence for t in tags) / len(tags)

        if "reject" in recommendations:
            return {
                "decision": "escalate",
                "avg_confidence": round(avg_confidence, 3),
                "reason": "存在 reject 标签，需 Arbitrator 仲裁",
            }
        elif "warn" in recommendations:
            return {
                "decision": "warn",
                "avg_confidence": round(avg_confidence, 3),
                "reason": "存在 warn 标签，执行但需注意",
            }
        else:
            return {
                "decision": "execute",
                "avg_confidence": round(avg_confidence, 3),
                "reason": "所有标签通过",
            }

    # ─── 查询 ──────────────────────────────────────────────

    def get_tag_log(self, task_id: str = None) -> list[Tag]:
        """获取标签历史"""
        if task_id:
            return [t for t in self._tag_log if t.task_id == task_id]
        return self._tag_log

    def clear_tag_log(self):
        self._tag_log.clear()
