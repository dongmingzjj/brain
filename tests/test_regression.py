"""回归测试单元测试"""
import pytest
import tempfile
import shutil
from pathlib import Path
from brain.event_bus import EventBus
from brain.cross_region import CrossRegionEvaluator
from brain.regression import RegressionTester, BenchmarkTask, DEFAULT_TASKS
from brain.regions.memory.executor import MemoryExecutor
from brain.regions.action.executor import ActionExecutor
from brain.regions.memory.local_improver import LocalImprover as MemImprover


@pytest.fixture
def setup():
    d = tempfile.mkdtemp(prefix="brain_regress_test_")
    bus = EventBus()
    mem = MemoryExecutor(str(Path(d) / "memory.db"))
    action = ActionExecutor(workdir=d, timeout=10)
    cross = CrossRegionEvaluator(bus, mem, action)

    # 预填一些记忆
    mem.store("python_basic", "Python 是一种编程语言", mem_type="fact", importance=0.7)
    mem.store("echo_tool", "echo 用于输出文本", mem_type="fact", importance=0.6)

    tester = RegressionTester(mem, action, cross)
    yield mem, action, cross, tester
    shutil.rmtree(d, ignore_errors=True)


class TestRunBaseline:
    """测试基准运行"""

    def test_baseline_runs(self, setup):
        """基准能跑"""
        _, _, _, tester = setup
        result = tester.run_baseline()
        assert result["total"] == 5
        assert result["action_matched"] >= 3  # 大部分应该匹配
        assert "memory_hit_rate" in result
        assert "avg_action_time_ms" in result
        assert len(result["details"]) == 5

    def test_baseline_memory_hits(self, setup):
        """有预填记忆时 memory_hit_rate > 0"""
        _, _, _, tester = setup
        result = tester.run_baseline()
        # "Python 基础" 和 "echo 输出" 应该命中
        assert result["memory_hits"] >= 1

    def test_action_match(self, setup):
        """成功/失败命令正确匹配"""
        _, _, _, tester = setup
        result = tester.run_baseline()
        details = {d["task_id"]: d for d in result["details"]}
        # e2e_001: python print(42) → 应成功
        assert details["e2e_001"]["action_match"] is True
        # e2e_004: python exit(1) → 应失败（期望失败）
        assert details["e2e_004"]["action_match"] is True
        assert details["e2e_004"]["action_success"] is False


class TestRunRegression:
    """测试回归检测"""

    def test_no_regression_without_improvement(self, setup):
        """不做改进 → 不应检测到回归"""
        _, _, _, tester = setup
        result = tester.run_regression(improvement_fn=None)
        assert result["passed"] is True
        assert result["regression_detected"] is False

    def test_memory_improvement_no_regression(self, setup):
        """Memory Local Improver 循环不导致回归"""
        mem, _, _, tester = setup
        improver = MemImprover(mem)

        result = tester.run_regression(
            improvement_fn=lambda: improver.run_cycle()
        )
        assert result["passed"] is True
        assert result["regression_detected"] is False

    def test_changes_reported(self, setup):
        """变化指标被报告"""
        _, _, _, tester = setup
        result = tester.run_regression()
        changes = result["changes"]
        assert "action_match_rate_delta" in changes
        assert "memory_hit_rate_delta" in changes
        assert "avg_action_time_delta_ms" in changes


class TestCustomTasks:
    """自定义任务集"""

    def test_custom_tasks(self, setup):
        """使用自定义任务"""
        mem, action, _, _ = setup
        tester = RegressionTester(mem, action, tasks=[
            BenchmarkTask(
                id="custom_1",
                description="自定义任务",
                memory_query="Python",
                action_command="echo custom",
                expected_action_success=True,
            ),
        ])
        result = tester.run_baseline()
        assert result["total"] == 1
