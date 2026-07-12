"""
Memory Region Metrics — 局部指标定义。

这些指标供 Local Improver 使用，也供 Verifier 验证。
全部确定性计算，不依赖 LLM。
"""

from __future__ import annotations
from .executor import MemoryExecutor


class MemoryMetrics:
    """Memory Region 的局部指标"""

    def __init__(self, executor: MemoryExecutor):
        self.executor = executor

    def compute_all(self) -> dict:
        """计算所有局部指标"""
        stats = self.executor.get_stats()
        access_stats = self.executor.get_access_log_stats()

        return {
            # 规模指标
            "total_memories": stats["total_memories"],
            "avg_importance": stats["avg_importance"],

            # 使用指标
            "total_accesses": stats["total_accesses"],
            "avg_relevance": access_stats["avg_relevance"],

            # 效率指标
            "access_per_memory": (
                stats["total_accesses"] / stats["total_memories"]
                if stats["total_memories"] > 0 else 0
            ),

            # 健康指标
            "high_importance_ratio": self._high_importance_ratio(),
            "stale_memory_ratio": self._stale_memory_ratio(),
        }

    def _high_importance_ratio(self) -> float:
        """重要性 > 0.7 的记忆占比"""
        total = self.executor.conn.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0]
        if total == 0:
            return 0.0
        high = self.executor.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE importance > 0.7"
        ).fetchone()[0]
        return round(high / total, 3)

    def _stale_memory_ratio(self) -> float:
        """7天未访问的记忆占比"""
        total = self.executor.conn.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0]
        if total == 0:
            return 0.0
        stale = self.executor.conn.execute("""
            SELECT COUNT(*) FROM memories
            WHERE last_accessed IS NULL
               OR last_accessed < datetime('now', '-7 days')
        """).fetchone()[0]
        return round(stale / total, 3)
