"""跨区评估单元测试"""
import pytest
import tempfile
import shutil
from pathlib import Path
from brain.event_bus import EventBus
from brain.cross_region import CrossRegionEvaluator, Tag
from brain.regions.memory.executor import MemoryExecutor
from brain.regions.action.executor import ActionExecutor


@pytest.fixture
def setup():
    d = tempfile.mkdtemp(prefix="brain_cross_test_")
    bus = EventBus()
    mem = MemoryExecutor(str(Path(d) / "memory.db"))
    action = ActionExecutor(workdir=d, timeout=10)
    evaluator = CrossRegionEvaluator(bus, mem, action)
    yield bus, mem, action, evaluator
    shutil.rmtree(d, ignore_errors=True)


class TestMemoryEvaluatesAction:
    """Memory → Action 评估"""

    def test_no_history(self, setup):
        """无历史记录时给中性标签"""
        _, _, _, evaluator = setup
        tag = evaluator.memory_evaluates_action({
            "tool": "curl",
            "command": "curl http://example.com",
            "success": True,
        })
        assert tag.source_region == "memory"
        assert tag.target_region == "action"
        assert tag.confidence == 0.5
        assert tag.recommendation == "approve"

    def test_all_success_history(self, setup):
        """全部成功历史 → 高 confidence"""
        _, mem, _, evaluator = setup
        # 存入成功历史
        import json
        for i in range(5):
            mem.store(
                key=f"action:echo:{i}",
                value=json.dumps({"tool": "echo", "success": True}),
                mem_type="action_history",
                importance=0.5,
            )

        tag = evaluator.memory_evaluates_action({
            "tool": "echo",
            "command": "echo hello",
            "success": True,
        })
        assert tag.confidence >= 0.8
        assert tag.recommendation == "approve"
        assert "成功率高" in tag.reason

    def test_high_failure_history(self, setup):
        """高失败率历史 → warn"""
        _, mem, _, evaluator = setup
        import json
        # 存入失败历史
        for i in range(4):
            mem.store(
                key=f"action:badcmd:{i}",
                value=json.dumps({"tool": "badcmd", "success": False}),
                mem_type="action_history",
                importance=0.5,
            )
        mem.store(
            key="action:badcmd:ok",
            value=json.dumps({"tool": "badcmd", "success": True}),
            mem_type="action_history",
            importance=0.5,
        )

        tag = evaluator.memory_evaluates_action({
            "tool": "badcmd",
            "command": "badcmd --flag",
            "success": False,
        })
        assert tag.recommendation == "warn"
        assert "失败率" in tag.reason

    def test_tag_published_to_bus(self, setup):
        """标签发布到 Event Bus"""
        bus, _, _, evaluator = setup

        received = []
        @bus.subscribe("action_evaluation")
        def on_eval(signal):
            received.append(signal)

        evaluator.memory_evaluates_action({"tool": "echo", "command": "echo", "success": True})

        assert len(received) == 1
        assert received[0].content["target"] == "action"


class TestActionEvaluatesMemory:
    """Action → Memory 评估"""

    def test_high_relevance(self, setup):
        """高相关性"""
        _, _, _, evaluator = setup
        tag = evaluator.action_evaluates_memory({
            "score": 0.8,
            "results": [{"key": "test", "value": "test"}],
        })
        assert tag.confidence >= 0.7
        assert tag.recommendation == "approve"

    def test_no_results(self, setup):
        """Memory 无结果"""
        _, _, _, evaluator = setup
        tag = evaluator.action_evaluates_memory({
            "score": 0,
            "results": [],
        })
        assert tag.confidence <= 0.3
        assert tag.recommendation == "warn"


class TestDecisionMaking:
    """决策聚合"""

    def test_all_approve(self, setup):
        """所有标签通过"""
        _, _, _, evaluator = setup
        tags = [
            Tag("memory", "action", 0.9, "approve", "ok"),
            Tag("action", "memory", 0.8, "approve", "ok"),
        ]
        result = evaluator.make_decision(tags)
        assert result["decision"] == "execute"

    def test_has_warn(self, setup):
        """有 warn 标签"""
        _, _, _, evaluator = setup
        tags = [
            Tag("memory", "action", 0.9, "approve", "ok"),
            Tag("action", "memory", 0.3, "warn", "risky"),
        ]
        result = evaluator.make_decision(tags)
        assert result["decision"] == "warn"

    def test_has_reject(self, setup):
        """有 reject 标签"""
        _, _, _, evaluator = setup
        tags = [
            Tag("memory", "action", 0.1, "reject", "bad"),
            Tag("action", "memory", 0.8, "approve", "ok"),
        ]
        result = evaluator.make_decision(tags)
        assert result["decision"] == "escalate"

    def test_empty_tags(self, setup):
        """无标签"""
        _, _, _, evaluator = setup
        result = evaluator.make_decision([])
        assert result["decision"] == "execute"


class TestTagLog:
    """标签日志"""

    def test_log_by_task(self, setup):
        """按 task_id 查日志"""
        _, _, _, evaluator = setup
        evaluator.memory_evaluates_action({"tool": "echo", "command": "e", "success": True}, task_id="t1")
        evaluator.action_evaluates_memory({"score": 0.5, "results": [{}]}, task_id="t2")

        log1 = evaluator.get_tag_log(task_id="t1")
        log2 = evaluator.get_tag_log(task_id="t2")
        assert len(log1) == 1
        assert len(log2) == 1
