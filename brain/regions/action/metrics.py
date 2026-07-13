"""
Action Region Metrics — 局部指标定义。
"""

from __future__ import annotations
from .executor import ActionExecutor


class ActionMetrics:
    """Action Region 的局部指标"""

    def __init__(self, executor: ActionExecutor):
        self.executor = executor

    def compute_all(self) -> dict:
        """计算所有局部指标"""
        stats = self.executor.get_stats()
        tool_stats = self.executor.get_tool_stats()

        # 工具成功率分布
        if tool_stats:
            rates = [t["success_rate"] for t in tool_stats.values()]
            avg_success_rate = sum(rates) / len(rates)
            best_tool = max(tool_stats.items(), key=lambda x: x[1]["success_rate"])
            worst_tool = min(tool_stats.items(), key=lambda x: x[1]["success_rate"])
        else:
            avg_success_rate = 0
            best_tool = None
            worst_tool = None

        return {
            # 规模指标
            "total_actions": stats["total_actions"],
            "tool_count": stats["tool_count"],

            # 效率指标
            "overall_success_rate": round(stats["success_rate"], 3),
            "avg_tool_success_rate": round(avg_success_rate, 3),

            # 健康指标
            "best_tool": best_tool[0] if best_tool else None,
            "best_tool_rate": round(best_tool[1]["success_rate"], 3) if best_tool else 0,
            "worst_tool": worst_tool[0] if worst_tool else None,
            "worst_tool_rate": round(worst_tool[1]["success_rate"], 3) if worst_tool else 0,
        }
