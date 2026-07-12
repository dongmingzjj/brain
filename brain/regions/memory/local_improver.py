"""
Memory Region Local Improver — 非 LLM 的局部改进循环。

改进机制（全部确定性，零 LLM 调用）：
  1. 检索质量监控：跟踪每次检索的相关性分数
  2. 阈值自适应：如果检索质量下降，调整检索阈值
  3. 重要性衰减：定期降低未访问记忆的重要性
  4. 遗忘触发：删除重要性过低且从未被访问的记忆

核心原则：
  - 只依赖局部指标（检索准确率、访问频率）
  - 不知道全局任务是什么
  - 每次改进都可量化、可回滚
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from .executor import MemoryExecutor


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LocalImprover:
    """Memory Region 的局部改进器"""

    def __init__(self, executor: MemoryExecutor):
        self.executor = executor
        self.improvement_log = []  # 改进记录

    def run_cycle(self) -> dict:
        """
        执行一轮改进循环。

        返回:
            {
                "actions_taken": [...],
                "metrics_before": {...},
                "metrics_after": {...},
                "improvement": float  # 正=改善, 负=退化, 0=无变化
            }
        """
        metrics_before = self.executor.get_stats()
        actions = []

        # 1. 重要性衰减
        decayed = self._decay_importance()
        if decayed > 0:
            actions.append({"action": "decay", "count": decayed})

        # 2. 遗忘低重要性记忆
        forgotten = self._forget_low_importance()
        if forgotten > 0:
            actions.append({"action": "forget", "count": forgotten})

        # 3. 检索质量检查 + 阈值调整
        threshold_adjusted = self._check_retrieval_quality()
        if threshold_adjusted:
            actions.append({"action": "adjust_threshold",
                           "new_threshold": threshold_adjusted})

        # 4. 提升高频访问记忆的重要性
        boosted = self._boost_popular_memories()
        if boosted > 0:
            actions.append({"action": "boost", "count": boosted})

        metrics_after = self.executor.get_stats()

        # 计算改善度
        improvement = self._compute_improvement(metrics_before, metrics_after)

        # 记录改进日志
        log_entry = {
            "timestamp": utc_now(),
            "actions": actions,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "improvement": improvement,
        }
        self.improvement_log.append(log_entry)

        return {
            "actions_taken": actions,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "improvement": improvement,
        }

    # ─── 改进步骤 ──────────────────────────────────────────

    def _decay_importance(self) -> int:
        """重要性衰减：7天未访问的记忆降低重要性"""
        before = self.executor.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE importance > 0.1"
        ).fetchone()[0]

        self.executor.decay_importance(decay_rate=0.02)

        after = self.executor.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE importance > 0.1"
        ).fetchone()[0]

        return before - after

    def _forget_low_importance(self) -> int:
        """遗忘：删除重要性 < 0.1 且从未被访问的记忆"""
        return self.executor.forget_by_importance(threshold=0.1)

    def _check_retrieval_quality(self) -> float | None:
        """
        检查检索质量，如果下降则调整阈值。

        策略：
        - 最近 20 次检索的平均相关性 < 0.3 → 降低阈值（放宽检索）
        - 最近 20 次检索的平均相关性 > 0.7 → 提高阈值（收紧检索）
        """
        stats = self.executor.get_access_log_stats()

        if stats["total_queries"] < 10:
            return None  # 数据不足，不调整

        avg_relevance = stats["avg_relevance"]

        if avg_relevance < 0.3:
            # 检索质量差，放宽阈值
            new_threshold = max(0.1, avg_relevance - 0.1)
            return new_threshold
        elif avg_relevance > 0.7:
            # 检索质量好，可以收紧
            new_threshold = min(0.9, avg_relevance + 0.1)
            return new_threshold

        return None

    def _boost_popular_memories(self) -> int:
        """提升高频访问记忆的重要性"""
        # 找出访问次数 > 5 但重要性 < 0.7 的记忆
        rows = self.executor.conn.execute("""
            SELECT id, access_count, importance
            FROM memories
            WHERE access_count > 5 AND importance < 0.7
            ORDER BY access_count DESC
            LIMIT 10
        """).fetchall()

        boosted = 0
        for row in rows:
            # 提升公式：new = min(0.9, old + 0.05 * log(access_count))
            import math
            new_importance = min(0.9, row["importance"] + 0.05 * math.log(row["access_count"]))
            if new_importance > row["importance"]:
                self.executor.update_importance(row["id"], new_importance)
                boosted += 1

        return boosted

    def _compute_improvement(self, before: dict, after: dict) -> float:
        """
        计算改善度（-1 到 +1）。

        指标：
        - 平均重要性变化
        - 记忆总数变化（适度减少是好的，说明在遗忘）
        """
        importance_delta = after["avg_importance"] - before["avg_importance"]

        # 记忆数变化：适度减少是好的（遗忘无用记忆）
        count_before = before["total_memories"]
        count_after = after["total_memories"]
        if count_before > 0:
            count_change_ratio = (count_after - count_before) / count_before
            # 适度减少（-10% 到 0%）得正分，大幅减少或增加得负分
            if -0.1 <= count_change_ratio <= 0:
                count_score = 0.1
            elif count_change_ratio > 0.1:
                count_score = -0.05  # 增长太快不好
            else:
                count_score = -0.1  # 减少太多不好
        else:
            count_score = 0

        return round(importance_delta + count_score, 3)

    def get_improvement_history(self) -> list[dict]:
        """获取改进历史"""
        return self.improvement_log

    def get_latest_improvement(self) -> dict | None:
        """获取最近一次改进"""
        return self.improvement_log[-1] if self.improvement_log else None
