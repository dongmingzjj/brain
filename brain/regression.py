"""
端到端回归测试 — 验证 R2 风险（局部改进不导致全局退化）。

策略：
  1. 定义一组端到端基准任务（经过 Memory + Action + 跨区评估完整链路）
  2. 记录改进前的全局指标
  3. 跑一轮 Memory Local Improver + Action Local Improver
  4. 重跑基准任务
  5. 如果全局指标下降 → 回归失败，拒绝改进
  6. 如果全局指标不降 → 通过

基准任务设计：
  - 每个任务经过 Memory 检索 → Action 执行 → 跨区评估 完整链路
  - 任务覆盖不同错误类型和工具类型
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BenchmarkTask:
    """一个端到端基准任务"""
    id: str
    description: str                          # 任务描述
    memory_query: str                         # Memory 检索查询
    action_command: str                       # Action 执行命令
    expected_action_success: bool = True      # 期望 Action 执行成功
    timeout: int = 10                         # 超时秒数


# ─── 默认基准任务集 ────────────────────────────────────────

DEFAULT_TASKS = [
    BenchmarkTask(
        id="e2e_001",
        description="检索 Python 记忆并执行 Python 命令",
        memory_query="Python 基础",
        action_command='python -c "print(42)"',
        expected_action_success=True,
    ),
    BenchmarkTask(
        id="e2e_002",
        description="检索 echo 记忆并执行 echo 命令",
        memory_query="echo 输出",
        action_command="echo brain_test",
        expected_action_success=True,
    ),
    BenchmarkTask(
        id="e2e_003",
        description="检索不存在的记忆，Action 仍然执行",
        memory_query="zzz_nonexistent_xyz",
        action_command="echo fallback",
        expected_action_success=True,
    ),
    BenchmarkTask(
        id="e2e_004",
        description="检索记忆后执行失败命令",
        memory_query="Python 错误",
        action_command='python -c "import sys; sys.exit(1)"',
        expected_action_success=False,
    ),
    BenchmarkTask(
        id="e2e_005",
        description="正常 git 命令",
        memory_query="git 版本控制",
        action_command="git --version",
        expected_action_success=True,
    ),
]


class RegressionTester:
    """端到端回归测试器"""

    def __init__(self, mem_executor, action_executor,
                 cross_evaluator=None,
                 tasks: list[BenchmarkTask] = None):
        self.mem = mem_executor
        self.action = action_executor
        self.cross = cross_evaluator
        self.tasks = tasks or DEFAULT_TASKS

    def run_baseline(self) -> dict:
        """
        跑一轮基准测试，返回全局指标。

        每个 task 的流程：
          1. Memory 检索
          2. Action 执行
          3. 跨区评估（如果有）
          4. 记录结果

        返回:
            {total, passed, failed, memory_hits, avg_action_time, details}
        """
        results = []
        memory_hits = 0
        action_successes = 0
        total_action_time = 0

        for task in self.tasks:
            # 1. Memory 检索
            mem_results = self.mem.retrieve(task.memory_query, top_k=3)
            mem_hit = len(mem_results) > 0
            if mem_hit:
                memory_hits += 1

            # 2. Action 执行
            action_result = self.action.execute(task.action_command, timeout=task.timeout)
            total_action_time += action_result["duration_ms"]

            # 3. 验证结果
            action_ok = action_result["success"] == task.expected_action_success
            if action_ok:
                action_successes += 1

            # 4. 跨区评估（如果有）
            tag = None
            if self.cross:
                tag = self.cross.memory_evaluates_action(action_result)

            results.append({
                "task_id": task.id,
                "description": task.description,
                "memory_hit": mem_hit,
                "memory_results": len(mem_results),
                "action_success": action_result["success"],
                "action_expected": task.expected_action_success,
                "action_match": action_ok,
                "action_time_ms": action_result["duration_ms"],
                "tag_confidence": tag.confidence if tag else None,
                "tag_recommendation": tag.recommendation if tag else None,
            })

        total = len(results)
        return {
            "total": total,
            "action_matched": action_successes,
            "action_match_rate": action_successes / total if total > 0 else 0,
            "memory_hits": memory_hits,
            "memory_hit_rate": memory_hits / total if total > 0 else 0,
            "avg_action_time_ms": total_action_time / total if total > 0 else 0,
            "details": results,
        }

    def run_regression(self, improvement_fn=None) -> dict:
        """
        运行回归测试。

        参数:
            improvement_fn: 一个函数，执行局部改进（如 Local Improver 循环）

        返回:
            {
                baseline: {...},
                post_improvement: {...},
                passed: bool,
                regression_detected: bool,
                changes: {...}
            }
        """
        # 1. 基准
        baseline = self.run_baseline()

        # 2. 执行改进
        if improvement_fn:
            improvement_fn()

        # 3. 重跑
        post = self.run_baseline()

        # 4. 比较
        changes = {
            "action_match_rate_delta": post["action_match_rate"] - baseline["action_match_rate"],
            "memory_hit_rate_delta": post["memory_hit_rate"] - baseline["memory_hit_rate"],
            "avg_action_time_delta_ms": post["avg_action_time_ms"] - baseline["avg_action_time_ms"],
        }

        # 5. 裁决
        # 核心规则：Action 匹配率不能下降
        regression = changes["action_match_rate_delta"] < -0.01

        return {
            "baseline": baseline,
            "post_improvement": post,
            "passed": not regression,
            "regression_detected": regression,
            "changes": changes,
        }
