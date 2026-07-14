"""Perception Region Metrics"""
from .executor import PerceptionExecutor


class PerceptionMetrics:
    """Perception Region 的局部指标"""

    def __init__(self, executor: PerceptionExecutor):
        self.executor = executor

    def compute_all(self) -> dict:
        stats = self.executor.get_stats()

        if not self.executor.conn:
            return {
                "total_perceptions": stats.get("total_perceptions", 0),
                "avg_confidence": 0,
                "intent_diversity": 0,
            }

        # 平均置信度
        avg_conf = self.executor.conn.execute(
            "SELECT AVG(confidence) FROM perceptions"
        ).fetchone()[0] or 0

        # 意图多样性（多少种不同意图）
        intent_count = self.executor.conn.execute(
            "SELECT COUNT(DISTINCT intent) FROM perceptions"
        ).fetchone()[0]

        # 低置信度占比
        total = stats.get("total_perceptions", 0)
        low_conf = self.executor.conn.execute(
            "SELECT COUNT(*) FROM perceptions WHERE confidence < 0.4"
        ).fetchone()[0]

        return {
            "total_perceptions": total,
            "avg_confidence": round(avg_conf, 3),
            "intent_diversity": intent_count,
            "low_confidence_ratio": round(low_conf / total, 3) if total > 0 else 0,
            "by_intent": stats.get("by_intent", {}),
        }
