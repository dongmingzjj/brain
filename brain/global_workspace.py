"""
Global Workspace — 信号竞争广播机制。

设计基于 Bernard Baars 的 Global Workspace Theory (GWT)：
  - 多个 Region 并行处理，大部分不进入意识
  - 只有当局部处理不够时（冲突/不确定/新情况），信号才进入全局广播
  - 广播的信息被所有 Region 共享，触发协作

实现：
  - 竞争规则：置信度 + 紧急度 + 新颖度 → 综合分数
  - 阈值门控：只有综合分数 > 阈值的信号才广播
  - 冲突检测：同一任务的多个信号互相矛盾 → 强制广播
  - 集体决策：多个 Region 的信号一致 → 直接决策，不升级
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from brain.event_bus import EventBus, Signal


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CompetitiveSignal:
    """参与竞争的信号"""
    signal: Signal
    urgency: float = 0.0       # 紧急度 (0-1)
    novelty: float = 0.0       # 新颖度 (0-1，新情况得分高)
    confidence: float = 0.0    # 来自信号本身的置信度
    composite_score: float = 0.0

    def compute_score(self) -> float:
        """计算综合竞争分数"""
        self.composite_score = (
            self.confidence * 0.4 +
            self.urgency * 0.4 +
            self.novelty * 0.2
        )
        return self.composite_score


class GlobalWorkspace:
    """全局工作区 — 信号竞争广播"""

    def __init__(self, bus: EventBus, broadcast_threshold: float = 0.5):
        """
        bus: Event Bus 实例
        broadcast_threshold: 广播阈值（综合分数 > 此值才广播）
        """
        self.bus = bus
        self.broadcast_threshold = broadcast_threshold
        self._broadcast_log: list[dict] = []
        self._conflict_log: list[dict] = []

    def evaluate_and_broadcast(self, signals: list[Signal],
                               task_id: str = "") -> dict:
        """
        评估一组信号，决定哪些广播。

        流程：
        1. 竞争打分
        2. 冲突检测
        3. 集体一致性检测
        4. 广播决策

        返回:
            {
                broadcast: list[Signal],      — 广播的信号
                conflicts: list[dict],         — 检测到的冲突
                consensus: bool,               — 是否集体一致
                escalated: bool,               — 是否升级到 Arbitrator
            }
        """
        if not signals:
            return {"broadcast": [], "conflicts": [], "consensus": False, "escalated": False}

        # 1. 竞争打分
        competitive = []
        for sig in signals:
            cs = CompetitiveSignal(
                signal=sig,
                confidence=sig.confidence,
                urgency=self._compute_urgency(sig),
                novelty=self._compute_novelty(sig),
            )
            cs.compute_score()
            competitive.append(cs)

        # 2. 冲突检测
        conflicts = self._detect_conflicts(competitive)
        self._conflict_log.extend(conflicts)

        # 3. 集体一致性检测
        consensus = self._check_consensus(competitive)

        # 4. 广播决策
        broadcast = []
        escalated = False

        if conflicts:
            # 有冲突 → 强制广播所有冲突信号 + 升级
            broadcast = [cs.signal for cs in competitive if cs.composite_score > 0]
            escalated = True
        elif consensus:
            # 集体一致 → 广播最强信号，不升级
            top = max(competitive, key=lambda c: c.composite_score)
            broadcast = [top.signal]
        else:
            # 正常竞争：只广播超过阈值的
            for cs in competitive:
                if cs.composite_score >= self.broadcast_threshold:
                    broadcast.append(cs.signal)

            # 如果有信号但都低于阈值 → 升级到 Arbitrator
            if not broadcast and competitive:
                escalated = True

        # 记录
        self._broadcast_log.append({
            "timestamp": utc_now(),
            "task_id": task_id,
            "input_count": len(signals),
            "broadcast_count": len(broadcast),
            "conflict_count": len(conflicts),
            "consensus": consensus,
            "escalated": escalated,
        })

        # 发布广播信号
        for sig in broadcast:
            self.bus.publish(Signal(
                source="global_workspace",
                type="broadcast",
                task_id=task_id,
                content={
                    "original_type": sig.type,
                    "original_source": sig.source,
                    "content": sig.content,
                },
            ))

        # 如果升级，发布升级信号
        if escalated:
            self.bus.publish(Signal(
                source="global_workspace",
                type="escalation",
                task_id=task_id,
                content={
                    "reason": "conflict" if conflicts else "low_confidence",
                    "signal_count": len(signals),
                    "conflict_details": [c for c in conflicts],
                },
            ))

        return {
            "broadcast": broadcast,
            "conflicts": conflicts,
            "consensus": consensus,
            "escalated": escalated,
        }

    # ─── 竞争打分 ──────────────────────────────────────────

    def _compute_urgency(self, signal: Signal) -> float:
        """计算紧急度"""
        content = signal.content or {}

        # 紧急关键词
        text = str(content).lower()
        urgent_markers = ["紧急", "急", "马上", "urgent", "error", "error", "失败", "错误"]
        if any(m in text for m in urgent_markers):
            return 0.9
        return 0.1

    def _compute_novelty(self, signal: Signal) -> float:
        """计算新颖度（基于历史中是否见过类似信号）"""
        # 简单实现：同类型的信号越少，新颖度越高
        history_types = [s.get("original_type", "") for s in self._broadcast_log[-20:]]
        if signal.type not in history_types:
            return 0.8  # 新类型信号
        return 0.2

    # ─── 冲突检测 ──────────────────────────────────────────

    def _detect_conflicts(self, competitive: list[CompetitiveSignal]) -> list[dict]:
        """
        检测信号之间的冲突。

        冲突定义：
        - 同一 task_id 下，两个信号的 confidence 差距 > 0.5
        - 或者两个信号的 recommendation 相反
        """
        conflicts = []
        n = len(competitive)

        for i in range(n):
            for j in range(i + 1, n):
                cs_a = competitive[i]
                cs_b = competitive[j]

                # 置信度差距大
                conf_diff = abs(cs_a.confidence - cs_b.confidence)
                if conf_diff > 0.5:
                    conflicts.append({
                        "type": "confidence_mismatch",
                        "signal_a": cs_a.signal.type,
                        "signal_b": cs_b.signal.type,
                        "confidence_diff": round(conf_diff, 3),
                        "source_a": cs_a.signal.source,
                        "source_b": cs_b.signal.source,
                    })

        return conflicts

    # ─── 集体一致性 ────────────────────────────────────────

    def _check_consensus(self, competitive: list[CompetitiveSignal]) -> bool:
        """
        检测集体一致性。

        一致性定义：所有信号的置信度都在 0.6 以上，且标准差 < 0.15。
        """
        if len(competitive) < 2:
            return False

        confidences = [cs.confidence for cs in competitive]
        avg = sum(confidences) / len(confidences)
        variance = sum((c - avg) ** 2 for c in confidences) / len(confidences)
        std = variance ** 0.5

        return avg >= 0.6 and std < 0.15

    # ─── 查询 ──────────────────────────────────────────────

    def get_broadcast_log(self, limit: int = 20) -> list[dict]:
        return self._broadcast_log[-limit:]

    def get_conflict_log(self, limit: int = 20) -> list[dict]:
        return self._conflict_log[-limit:]

    def clear_logs(self):
        self._broadcast_log.clear()
        self._conflict_log.clear()
