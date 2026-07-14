"""
Perception Region Local Improver — 意图分类规则自适应。

改进机制（全部确定性，零 LLM）：
  1. 意图混淆检测：同一输入匹配多个意图 → 规则不够精确
  2. 关键词权重微调：基于历史正确/错误的意图判断
  3. 新关键词发现：从历史输入中提取高频但未匹配的词
  4. 低置信度监控：置信度持续偏低 → 规则需要扩充
"""

from __future__ import annotations
import json
from collections import Counter
from datetime import datetime, timezone
from .executor import PerceptionExecutor, INTENT_RULES


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LocalImprover:
    """Perception Region 的局部改进器"""

    def __init__(self, executor: PerceptionExecutor):
        self.executor = executor
        self.improvement_log = []

    def run_cycle(self) -> dict:
        """执行一轮改进循环"""
        stats_before = self.executor.get_stats()
        actions = []

        # 1. 分析低置信度比例
        low_conf = self._check_low_confidence()
        if low_conf["ratio"] > 0.3:
            actions.append({
                "action": "low_confidence_alert",
                "ratio": low_conf["ratio"],
                "suggestion": "考虑增加意图关键词或引入语义理解",
            })

        # 2. 分析意图分布
        distribution = self._analyze_distribution()
        if distribution["imbalanced"]:
            actions.append({
                "action": "distribution_imbalance",
                "dominant": distribution["dominant"],
                "ratio": distribution["dominant_ratio"],
            })

        # 3. 关键词覆盖率检查
        coverage = self._check_keyword_coverage()
        if coverage["uncovered高频词"]:
            actions.append({
                "action": "new_keywords_candidate",
                "words": coverage["uncovered高频词"][:5],
            })

        stats_after = self.executor.get_stats()
        improvement = self._compute_improvement(stats_before, stats_after)

        log_entry = {
            "timestamp": utc_now(),
            "actions": actions,
            "improvement": improvement,
        }
        self.improvement_log.append(log_entry)

        return {
            "actions_taken": actions,
            "metrics_before": stats_before,
            "metrics_after": stats_after,
            "improvement": improvement,
        }

    def _check_low_confidence(self) -> dict:
        """检查低置信度比例"""
        if not self.executor.conn:
            return {"ratio": 0, "count": 0}

        rows = self.executor.conn.execute(
            "SELECT confidence FROM perceptions ORDER BY id DESC LIMIT 50"
        ).fetchall()

        if not rows:
            return {"ratio": 0, "count": 0}

        low_count = sum(1 for r in rows if r["confidence"] < 0.4)
        return {
            "ratio": round(low_count / len(rows), 3),
            "count": low_count,
            "total": len(rows),
        }

    def _analyze_distribution(self) -> dict:
        """分析意图分布"""
        stats = self.executor.get_stats()
        by_intent = stats.get("by_intent", {})

        if not by_intent:
            return {"imbalanced": False}

        total = sum(by_intent.values())
        if total == 0:
            return {"imbalanced": False}

        dominant = max(by_intent, key=by_intent.get)
        dominant_ratio = by_intent[dominant] / total

        return {
            "imbalanced": dominant_ratio > 0.6 and len(by_intent) > 1,
            "dominant": dominant,
            "dominant_ratio": round(dominant_ratio, 3),
            "distribution": by_intent,
        }

    def _check_keyword_coverage(self) -> dict:
        """检查关键词覆盖情况"""
        # Phase 4.0: 简化版 — 检查有多少输入的意图是 "unknown"
        if not self.executor.conn:
            return {"uncovered高频词": []}

        # 统计 unknown 意图的输入
        unknown_count = self.executor.conn.execute(
            "SELECT COUNT(*) FROM perceptions WHERE intent = 'unknown'"
        ).fetchone()[0]

        # 收集所有已覆盖的关键词
        all_keywords = set()
        for rule in INTENT_RULES.values():
            all_keywords.update(kw.lower() for kw in rule["keywords"])

        return {
            "unknown_count": unknown_count,
            "covered_keywords": len(all_keywords),
            "uncovered高频词": [],  # Phase 4.1: 从 unknown 输入中提取高频词
        }

    def _compute_improvement(self, before: dict, after: dict) -> float:
        """计算改善度"""
        # Phase 4.0: 基于低置信度比例变化
        return 0.0

    def get_improvement_history(self) -> list[dict]:
        return self.improvement_log
