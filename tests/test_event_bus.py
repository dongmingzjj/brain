"""Event Bus 单元测试"""
import pytest
from brain.event_bus import EventBus, Signal


@pytest.fixture
def bus():
    return EventBus()


class TestSubscribe:
    """测试订阅"""

    def test_subscribe_decorator(self, bus):
        """装饰器模式订阅"""
        @bus.subscribe("test_signal")
        def handler(signal):
            return f"handled: {signal.content}"

        assert "test_signal" in bus.get_subscribers()

    def test_subscribe_direct(self, bus):
        """直接调用订阅"""
        def handler(signal):
            return "ok"

        bus.subscribe("test_signal", handler=handler, subscriber="test_region")
        subs = bus.get_subscribers()
        assert "test_signal" in subs
        assert subs["test_signal"][0]["subscriber"] == "test_region"

    def test_multiple_subscribers(self, bus):
        """同一信号多个订阅者"""
        results = []

        @bus.subscribe("test_signal", subscriber="a")
        def handler_a(signal):
            results.append("a")

        @bus.subscribe("test_signal", subscriber="b")
        def handler_b(signal):
            results.append("b")

        bus.publish(Signal(source="test", type="test_signal"))
        assert set(results) == {"a", "b"}


class TestPublish:
    """测试发布"""

    def test_publish_basic(self, bus):
        """基本发布"""
        received = []

        @bus.subscribe("greeting")
        def handler(signal):
            received.append(signal.content)

        bus.publish(Signal(
            source="test",
            type="greeting",
            content={"msg": "hello"},
        ))

        assert len(received) == 1
        assert received[0]["msg"] == "hello"

    def test_publish_return_values(self, bus):
        """发布返回所有处理结果"""
        @bus.subscribe("compute")
        def double(signal):
            return signal.content["x"] * 2

        @bus.subscribe("compute")
        def triple(signal):
            return signal.content["x"] * 3

        results = bus.publish(Signal(
            source="test",
            type="compute",
            content={"x": 5},
        ))

        assert 10 in results
        assert 15 in results

    def test_publish_priority_order(self, bus):
        """按优先级排序调用"""
        order = []

        @bus.subscribe("test", subscriber="low", priority="low")
        def handler_low(signal):
            order.append("low")

        @bus.subscribe("test", subscriber="high", priority="high")
        def handler_high(signal):
            order.append("high")

        @bus.subscribe("test", subscriber="normal", priority="normal")
        def handler_normal(signal):
            order.append("normal")

        bus.publish(Signal(source="test", type="test"))
        assert order == ["high", "normal", "low"]

    def test_publish_dead_signal(self, bus):
        """无订阅者的死信号"""
        signal = Signal(source="test", type="nobody_listens")
        bus.publish(signal)

        dead = bus.get_dead_signals()
        assert len(dead) == 1
        assert dead[0].type == "nobody_listens"

    def test_publish_handler_error(self, bus):
        """handler 出错不影响其他订阅者"""
        @bus.subscribe("test", subscriber="broken")
        def handler_broken(signal):
            raise ValueError("oops")

        @bus.subscribe("test", subscriber="good")
        def handler_good(signal):
            return "ok"

        results = bus.publish(Signal(source="test", type="test"))
        assert len(results) == 2
        assert any(r == "ok" for r in results)
        assert any("_error" in r if isinstance(r, dict) else False for r in results)


class TestHistory:
    """测试信号历史"""

    def test_history_by_task(self, bus):
        """按 task_id 查历史"""
        bus.publish(Signal(source="a", type="t1", task_id="task_001"))
        bus.publish(Signal(source="b", type="t2", task_id="task_001"))
        bus.publish(Signal(source="c", type="t1", task_id="task_002"))

        history_1 = bus.get_history("task_001")
        history_2 = bus.get_history("task_002")

        assert len(history_1) == 2
        assert len(history_2) == 1

    def test_clear_history(self, bus):
        """清理历史"""
        bus.publish(Signal(source="a", type="t1", task_id="task_001"))
        bus.clear_history("task_001")
        assert bus.get_history("task_001") == []


class TestAndWait:
    """测试 publish_and_wait（join 机制）"""

    def test_join_basic(self, bus):
        """发布多个信号 + 收集响应"""
        # 设置两个订阅者，它们会发回响应
        @bus.subscribe("query_memory", subscriber="memory")
        def memory_handler(signal):
            # Memory 处理后发回结果
            bus.publish(Signal(
                source="region:memory",
                type="recall_result",
                task_id=signal.task_id,
                content={"found": "something"},
            ))
            return "memory_done"

        @bus.subscribe("query_perception", subscriber="perception")
        def perception_handler(signal):
            bus.publish(Signal(
                source="region:perception",
                type="perception_result",
                task_id=signal.task_id,
                content={"intent": "query"},
            ))
            return "perception_done"

        # 发布两个查询，等待两个结果
        task_id = "task_join_001"
        responses = bus.publish_and_wait(
            signals=[
                Signal(source="arbitrator", type="query_memory", task_id=task_id),
                Signal(source="arbitrator", type="query_perception", task_id=task_id),
            ],
            expected_types=["recall_result", "perception_result"],
        )

        assert "recall_result" in responses
        assert len(responses["recall_result"]) >= 1
        assert responses["recall_result"][0].content["found"] == "something"

        assert "perception_result" in responses
        assert len(responses["perception_result"]) >= 1

    def test_join_partial_response(self, bus):
        """只有一个 Region 响应"""
        @bus.subscribe("query_a")
        def handler_a(signal):
            bus.publish(Signal(
                source="region:a",
                type="result_a",
                task_id=signal.task_id,
            ))

        # query_b 没有订阅者
        responses = bus.publish_and_wait(
            signals=[
                Signal(source="test", type="query_a", task_id="t1"),
                Signal(source="test", type="query_b", task_id="t1"),
            ],
            expected_types=["result_a", "result_b"],
        )

        assert len(responses["result_a"]) == 1
        assert len(responses["result_b"]) == 0  # 没有响应


class TestSignalDataclass:
    """测试 Signal 数据结构"""

    def test_signal_defaults(self):
        s = Signal(source="test")
        assert s.target == "broadcast"
        assert s.confidence == 1.0
        assert s.content == {}

    def test_signal_to_dict(self):
        s = Signal(
            source="memory",
            type="recall",
            content={"key": "value"},
            task_id="t1",
        )
        d = s.to_dict()
        assert d["source"] == "memory"
        assert d["type"] == "recall"
        assert d["content"]["key"] == "value"
        assert d["task_id"] == "t1"
