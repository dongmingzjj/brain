"""Brain Engine 集成测试 — 三 Region 联动"""
import pytest
import tempfile
import shutil
from brain.engine import BrainEngine


@pytest.fixture
def engine():
    d = tempfile.mkdtemp(prefix="brain_engine_test_")
    eng = BrainEngine(brain_dir=d)
    yield eng
    eng.close()
    shutil.rmtree(d, ignore_errors=True)


class TestPerceptionMemoryFlow:
    """Perception → Memory 链路"""

    def test_question_triggers_recall(self, engine):
        """问句触发 Memory 检索"""
        # 预填 Memory
        engine.memory.store("python_gil", "Python GIL 限制多线程", importance=0.8)

        result = engine.process("Python GIL 怎么回事？")
        assert result["perception"]["intent"] == "question"
        assert len(result["recall"]) > 0  # Memory 自动检索了

    def test_no_match_returns_empty(self, engine):
        """Memory 无匹配"""
        result = engine.process("zzz_nonexistent_xyz")
        assert result["perception"]["intent"] != "command"
        # recall 可能为空
        assert isinstance(result["recall"], list)

    def test_perception_summary_as_query(self, engine):
        """Perception 摘要作为 Memory 检索查询"""
        engine.memory.store("async", "asyncio 异步框架", importance=0.7)

        result = engine.process("asyncio 怎么用？")
        assert result["perception"]["intent"] == "question"
        # Memory 应该检索到
        assert len(result["recall"]) > 0


class TestPerceptionActionFlow:
    """Perception → Action 链路"""

    def test_command_triggers_action(self, engine):
        """命令意图触发 Action 执行"""
        result = engine.process("帮我运行 `echo hello`")
        assert result["perception"]["intent"] == "command"
        assert result["action"] is not None
        assert result["action"]["success"] is True
        assert "hello" in result["action"]["stdout"]

    def test_non_command_no_action(self, engine):
        """非命令不触发 Action"""
        result = engine.process("Python 是什么？")
        assert result["action"] is None

    def test_action_with_cross_eval(self, engine):
        """Action 执行后触发跨区评估"""
        result = engine.process("运行 `echo test`")
        assert result["action"] is not None
        assert len(result["tags"]) > 0  # Memory 给 Action 打了标签


class TestEscalation:
    """Global Workspace 升级"""

    def test_low_confidence_escalation(self, engine):
        """低置信度触发升级"""
        result = engine.process("xyzzy")
        assert result.get("escalation") == "low_confidence"

    def test_urgent_escalation(self, engine):
        """紧急情感触发升级"""
        result = engine.process("紧急！马上帮我运行 `echo fast`")
        assert result.get("escalation") == "urgent"

    def test_no_escalation_normal(self, engine):
        """正常输入不触发升级"""
        result = engine.process("Python GIL 是什么？")
        assert "escalation" not in result or result.get("escalation") is None


class TestEngineStatus:
    """引擎状态"""

    def test_status(self, engine):
        engine.process("测试输入")
        status = engine.get_status()
        assert status["cycle_count"] == 1
        assert "perception" in status
        assert "memory" in status
        assert "action" in status
        assert "bus_subscribers" in status

    def test_multiple_cycles(self, engine):
        """多次处理"""
        for i in range(3):
            engine.process(f"测试 {i}")
        status = engine.get_status()
        assert status["cycle_count"] == 3
