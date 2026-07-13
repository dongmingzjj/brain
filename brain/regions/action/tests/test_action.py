"""Action Region 单元测试"""
import pytest
import tempfile
import shutil
from pathlib import Path
from brain.regions.action.executor import ActionExecutor
from brain.regions.action.local_improver import LocalImprover
from brain.regions.action.metrics import ActionMetrics


@pytest.fixture
def workdir():
    d = tempfile.mkdtemp(prefix="brain_action_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def executor(workdir):
    return ActionExecutor(workdir=workdir, timeout=10)


@pytest.fixture
def improver(executor):
    return LocalImprover(executor)


@pytest.fixture
def metrics(executor):
    return ActionMetrics(executor)


class TestActionExecutor:
    """测试 Action Executor"""

    def test_execute_echo(self, executor):
        """执行简单命令"""
        result = executor.execute("echo hello")
        assert result["success"] is True
        assert "hello" in result["stdout"]
        assert result["exit_code"] == 0

    def test_execute_python(self, executor):
        """执行 Python"""
        result = executor.execute('python -c "print(1+1)"')
        assert result["success"] is True
        assert "2" in result["stdout"]

    def test_execute_fail(self, executor):
        """执行失败命令"""
        result = executor.execute('python -c "import sys; sys.exit(1)"')
        assert result["success"] is False
        assert result["exit_code"] == 1

    def test_execute_forbidden(self, executor):
        """安全拦截"""
        result = executor.execute("rm -rf /")
        assert result["success"] is False
        assert "安全拦截" in result["stderr"]

    def test_execute_timeout(self, executor):
        """超时"""
        result = executor.execute('python -c "import time; time.sleep(20)"', timeout=2)
        assert result["success"] is False
        assert "超时" in result["stderr"]

    def test_tool_extraction(self, executor):
        """工具名提取"""
        executor.execute("echo test1")
        executor.execute("python -c 'print(1)'")
        stats = executor.get_tool_stats()
        assert "echo" in stats
        assert "python" in stats

    def test_tool_stats_tracking(self, executor):
        """工具统计追踪"""
        executor.execute("echo a")
        executor.execute("echo b")
        executor.execute('python -c "import sys; sys.exit(1)"')

        stats = executor.get_tool_stats()
        assert stats["echo"]["total"] == 2
        assert stats["echo"]["success"] == 2
        assert stats["python"]["total"] == 1
        assert stats["python"]["fail"] == 1

    def test_get_stats(self, executor):
        """全局统计"""
        executor.execute("echo test")
        stats = executor.get_stats()
        assert stats["total_actions"] == 1
        assert stats["success_rate"] == 1.0

    def test_recent_actions(self, executor):
        """最近执行记录"""
        executor.execute("echo first")
        executor.execute("echo second")
        actions = executor.get_recent_actions(limit=5)
        assert len(actions) == 2
        # 最新的在前
        assert "second" in actions[0]["command"]


class TestLocalImprover:
    """测试 Local Improver (Bandit)"""

    def test_run_cycle_empty(self, improver):
        """空历史的改进循环"""
        result = improver.run_cycle()
        assert "actions_taken" in result
        assert "improvement" in result

    def test_degrade_failing(self, improver, executor):
        """检测连续失败的工具"""
        # 制造 3 次失败
        for _ in range(3):
            executor.execute('python -c "import sys; sys.exit(1)"')

        result = improver.run_cycle()
        degraded = any(
            a["action"] == "degrade_failing"
            for a in result["actions_taken"]
        )
        assert degraded

    def test_recommend_tool_exploit(self, improver, executor):
        """Bandit 利用：选成功率最高的"""
        executor.execute("echo success1")
        executor.execute("echo success2")
        executor.execute("python -c 'exit(1)'")

        # explore_rate=0 → 纯利用
        tool = improver.recommend_tool(explore_rate=0)
        assert tool == "echo"

    def test_wilson_confidence(self, improver, executor):
        """Wilson 置信度"""
        # 1 次成功
        executor.execute("echo test")
        conf_1 = improver.get_tool_confidence("echo")

        # 10 次成功
        for _ in range(9):
            executor.execute("echo test")
        conf_10 = improver.get_tool_confidence("echo")

        # 10 次成功的置信度应该高于 1 次
        assert conf_10 > conf_1
        # 但都不到 1.0（Wilson 下界保守）
        assert conf_10 < 1.0


class TestActionMetrics:
    """测试 Action Metrics"""

    def test_empty(self, metrics):
        """空状态"""
        result = metrics.compute_all()
        assert result["total_actions"] == 0

    def test_with_data(self, metrics, executor):
        """有数据"""
        executor.execute("echo a")
        executor.execute("echo b")
        executor.execute('python -c "import sys; sys.exit(1)"')

        result = metrics.compute_all()
        assert result["total_actions"] == 3
        assert result["tool_count"] == 2
        assert result["best_tool"] == "echo"
        assert result["worst_tool"] == "python"
