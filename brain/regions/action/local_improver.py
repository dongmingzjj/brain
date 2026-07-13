"""
Action Region Local Improver — 多臂 bandit 工具选择优化。

改进机制（全部确定性，零 LLM 调用）：
  1. 成功率追踪：统计每个工具的历史成功率
  2. bandit 选择：优先选成功率高的工具（exploit），偶尔探索新工具（explore）
  3. 失败模式检测：连续失败的工具降级
  4. 超时优化：记录平均执行时间，对慢工具加预警

核心原则：
  - 只依赖局部指标（成功率、延迟）
  - 不知道全局任务是什么
  - 每次改进都可量化
"""

from __future__ import annotations
import math
import random
from datetime import datetime, timezone
from .executor import ActionExecutor


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LocalImprover:
    """Action Region 的局部改进器（多臂 bandit）"""

    def __init__(self, executor: ActionExecutor):
        self.executor = executor
        self.improvement_log = []

    def run_cycle(self) -> dict:
        """
        执行一轮改进循环。

        返回:
            {actions_taken, metrics_before, metrics_after, improvement}
        """
        stats_before = self.executor.get_stats()
        tool_stats_before = self.executor.get_tool_stats()
        actions = []

        # 1. 检测连续失败的工具
        degraded = self._degrade_failing_tools(tool_stats_before)
        if degraded:
            actions.append({"action": "degrade_failing", "tools": degraded})

        # 2. 检测异常慢的工具
        slow = self._detect_slow_tools(tool_stats_before)
        if slow:
            actions.append({"action": "mark_slow", "tools": slow})

        # 3. 更新推荐工具排序
        ranking = self._update_tool_ranking(tool_stats_before)
        if ranking:
            actions.append({"action": "update_ranking", "top_tools": ranking[:5]})

        stats_after = self.executor.get_stats()
        improvement = self._compute_improvement(stats_before, stats_after, tool_stats_before)

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

    # ─── Bandit 选择 ──────────────────────────────────────

    def recommend_tool(self, task_description: str = "",
                       explore_rate: float = 0.1) -> str | None:
        """
        用 epsilon-greedy 策略推荐工具。

        参数:
            task_description: 任务描述（Phase 2.0 不使用，Phase 2.1 按任务类型选工具）
            explore_rate: 探索概率（0.1 = 10% 概率随机选）

        返回:
            推荐的工具名
        """
        tool_stats = self.executor.get_tool_stats()

        if not tool_stats:
            return None

        # epsilon-greedy
        if random.random() < explore_rate:
            # 探索：随机选一个
            return random.choice(list(tool_stats.keys()))

        # 利用：选成功率最高的
        ranked = sorted(
            tool_stats.items(),
            key=lambda x: x[1]["success_rate"],
            reverse=True
        )
        return ranked[0][0] if ranked else None

    def get_tool_confidence(self, tool: str) -> float:
        """
        计算工具的置信度（Wilson 区间下界）。

        比 raw success_rate 更可靠——执行 1 次成功 1 次（100%）的置信度
        不如执行 100 次成功 90 次（90%）。
        """
        stats = self.executor.get_tool_stats()
        if tool not in stats:
            return 0.0

        s = stats[tool]
        total = s["total"]
        success = s["success"]

        if total == 0:
            return 0.0

        # Wilson score interval 下界
        z = 1.96  # 95% 置信
        p = success / total
        denominator = 1 + z * z / total
        center = (p + z * z / (2 * total)) / denominator
        spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator

        return max(0, center - spread)

    # ─── 改进步骤 ──────────────────────────────────────────

    def _degrade_failing_tools(self, tool_stats: dict) -> list[str]:
        """检测连续失败的工具（成功率 < 30% 且执行 >= 3 次）"""
        degraded = []
        for tool, stats in tool_stats.items():
            if stats["total"] >= 3 and stats["success_rate"] < 0.3:
                degraded.append(tool)
        return degraded

    def _detect_slow_tools(self, tool_stats: dict) -> list[str]:
        """检测异常慢的工具（平均 > 10 秒）"""
        slow = []
        for tool, stats in tool_stats.items():
            if stats["avg_duration_ms"] > 10000:
                slow.append(f"{tool}({stats['avg_duration_ms']:.0f}ms)")
        return slow

    def _update_tool_ranking(self, tool_stats: dict) -> list[str]:
        """按 Wilson 置信度排序工具"""
        ranked = []
        for tool in tool_stats:
            confidence = self.get_tool_confidence(tool)
            ranked.append((tool, confidence))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return [tool for tool, _ in ranked]

    def _compute_improvement(self, before: dict, after: dict,
                             tool_stats: dict) -> float:
        """计算改善度"""
        # Phase 2.0: 改善度 = 工具覆盖率变化 + 平均成功率变化
        # 这里简化为 0（Local Improver 的改效果需要长期观察）
        return 0.0

    def get_improvement_history(self) -> list[dict]:
        return self.improvement_log
