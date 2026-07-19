"""Global Workspace 单元测试"""
import pytest
from brain.event_bus import EventBus, Signal
from brain.global_workspace import GlobalWorkspace, CompetitiveSignal


@pytest.fixture
def setup():
    bus = EventBus()
    gw = GlobalWorkspace(bus, broadcast_threshold=0.5)
    yield bus, gw


class TestCompetitiveScoring:
    """测试竞争打分"""

    def test_high_confidence_signal(self, setup):
        """高置信度 + 正常紧急度 → 广播"""
        _, gw = setup
        sig = Signal(source="memory", type="recall_result", confidence=0.9,
                     content={"results": []})
        cs = CompetitiveSignal(signal=sig, confidence=0.9, urgency=0.3, novelty=0.3)
        score = cs.compute_score()
        assert score >= 0.5  # 超过阈值

    def test_low_confidence_signal(self, setup):
        """低置信度信号得分低"""
        _, gw = setup
        sig = Signal(source="memory", type="recall_result", confidence=0.2,
                     content={"results": []})
        cs = CompetitiveSignal(signal=sig, confidence=0.2, urgency=0.1, novelty=0.2)
        score = cs.compute_score()
        assert score < 0.5  # 低于阈值

    def test_urgent_signal_boosts(self, setup):
        """紧急信号额外加分"""
        _, gw = setup
        sig = Signal(source="action", type="action_error", confidence=0.5,
                     content={"error": "紧急：执行失败"})
        cs = CompetitiveSignal(signal=sig, confidence=0.5, urgency=0.9, novelty=0.2)
        score = cs.compute_score()
        assert score >= 0.5  # urgency 拉高到阈值以上


class TestBroadcast:
    """测试广播决策"""

    def test_no_signals(self, setup):
        """无信号"""
        _, gw = setup
        result = gw.evaluate_and_broadcast([])
        assert result["broadcast"] == []
        assert result["escalated"] is False

    def test_high_signal_broadcasts(self, setup):
        """高分信号广播"""
        bus, gw = setup
        received = []
        @bus.subscribe("broadcast")
        def on_broadcast(signal):
            received.append(signal)

        sig = Signal(source="memory", type="recall", confidence=0.9,
                     content={"found": "something"})
        result = gw.evaluate_and_broadcast([sig], task_id="t1")

        assert len(result["broadcast"]) == 1
        assert len(received) >= 1  # 广播信号被发出

    def test_low_signal_no_broadcast(self, setup):
        """低分信号不广播"""
        _, gw = setup
        sig = Signal(source="memory", type="recall", confidence=0.1,
                     content={"found": "nothing"})
        result = gw.evaluate_and_broadcast([sig], task_id="t1")

        assert len(result["broadcast"]) == 0

    def test_consensus_no_escalation(self, setup):
        """集体一致 → 不升级"""
        _, gw = setup
        signals = [
            Signal(source="region:a", type="result", confidence=0.8, content={}),
            Signal(source="region:b", type="result", confidence=0.75, content={}),
        ]
        result = gw.evaluate_and_broadcast(signals)
        assert result["consensus"] is True
        assert result["escalated"] is False


class TestConflictDetection:
    """测试冲突检测"""

    def test_confidence_mismatch(self, setup):
        """置信度差距大 → 冲突"""
        _, gw = setup
        signals = [
            Signal(source="region:a", type="recall", confidence=0.9, content={}),
            Signal(source="region:b", type="recall", confidence=0.2, content={}),
        ]
        result = gw.evaluate_and_broadcast(signals, task_id="t1")
        assert len(result["conflicts"]) > 0
        assert result["conflicts"][0]["type"] == "confidence_mismatch"

    def test_conflict_triggers_escalation(self, setup):
        """冲突 → 强制升级"""
        bus, gw = setup
        received = []
        @bus.subscribe("escalation")
        def on_escalation(signal):
            received.append(signal)

        signals = [
            Signal(source="region:a", type="recall", confidence=0.9, content={}),
            Signal(source="region:b", type="recall", confidence=0.1, content={}),
        ]
        result = gw.evaluate_and_broadcast(signals, task_id="t1")

        assert result["escalated"] is True
        assert len(received) >= 1


class TestLogs:
    """测试日志"""

    def test_broadcast_log(self, setup):
        _, gw = setup
        sig = Signal(source="test", type="t", confidence=0.8, content={})
        gw.evaluate_and_broadcast([sig], task_id="t1")

        log = gw.get_broadcast_log()
        assert len(log) == 1
        assert log[0]["task_id"] == "t1"

    def test_conflict_log(self, setup):
        _, gw = setup
        signals = [
            Signal(source="a", type="t", confidence=0.9, content={}),
            Signal(source="b", type="t", confidence=0.1, content={}),
        ]
        gw.evaluate_and_broadcast(signals, task_id="t1")

        log = gw.get_conflict_log()
        assert len(log) >= 1
