"""
Event Bus — Brain 的信号路由中枢。

Phase 2 最简版：
  - 同步事件循环（无 async，避免 GIL/锁/竞态问题）
  - 订阅制路由（signal_type 匹配）
  - join 机制（等同一 task_id 下多个信号到齐）
  - 死信号检测（无订阅的信号记录到日志）

设计原则：
  - Region 不互相直接调用，只通过 Bus 收发信号
  - Region 不互相改输出，只互相打标签
  - Bus 是被动的：只路由，不处理业务逻辑
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Any
import threading


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Signal:
    """BSP 信号 — Phase 2 简化版"""

    source: str                               # 发送者: "region:memory" | "arbitrator" | ...
    target: str = "broadcast"                 # 目标: "broadcast" | "region:planning" | ...
    timestamp: str = field(default_factory=utc_now)
    task_id: str = ""                         # 任务 ID

    type: str = ""                            # 信号类型: "recall_result" | "action_done" | ...
    confidence: float = 1.0
    content: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "timestamp": self.timestamp,
            "task_id": self.task_id,
            "type": self.type,
            "confidence": self.confidence,
            "content": self.content,
        }


# ─── 订阅规则 ──────────────────────────────────────────────

@dataclass
class Subscription:
    """一条订阅规则"""
    signal_type: str                          # 匹配的信号类型
    handler: Callable                         # 处理函数
    subscriber: str = ""                      # 订阅者名称（debug 用）
    priority: str = "normal"                  # high / normal / low


# ─── Event Bus ─────────────────────────────────────────────

class EventBus:
    """
    同步事件总线。

    用法:
        bus = EventBus()

        @bus.subscribe("recall_result")
        def on_recall(signal):
            print(f"收到: {signal.content}")

        bus.publish(Signal(source="memory", type="recall_result", content={...}))
    """

    def __init__(self):
        # signal_type → list[Subscription]
        self._subscribers: dict[str, list[Subscription]] = {}
        # 死信号日志（无订阅的信号）
        self._dead_signals: list[Signal] = []
        # 信号历史（同一 task_id 的信号聚合）
        self._signal_history: dict[str, list[Signal]] = {}

    # ─── 订阅 ──────────────────────────────────────────

    def subscribe(self, signal_type: str,
                  handler: Callable = None,
                  subscriber: str = "",
                  priority: str = "normal") -> Callable:
        """
        订阅一种信号类型。

        可以用作装饰器：
            @bus.subscribe("recall_result")
            def handler(signal): ...

        也可以直接调用：
            bus.subscribe("recall_result", handler=my_func, subscriber="memory")
        """
        def _register(h: Callable):
            sub = Subscription(
                signal_type=signal_type,
                handler=h,
                subscriber=subscriber,
                priority=priority,
            )
            if signal_type not in self._subscribers:
                self._subscribers[signal_type] = []
            self._subscribers[signal_type].append(sub)
            return h

        if handler is not None:
            return _register(handler)
        return _register

    # ─── 发布 ──────────────────────────────────────────

    def publish(self, signal: Signal) -> list[Any]:
        """
        发布一个信号，同步调用所有订阅者。

        返回:
            所有订阅者的返回值列表
        """
        # 记录到历史
        if signal.task_id:
            if signal.task_id not in self._signal_history:
                self._signal_history[signal.task_id] = []
            self._signal_history[signal.task_id].append(signal)

        # 查找订阅者
        subs = self._subscribers.get(signal.type, [])

        if not subs:
            # 死信号：无订阅者
            self._dead_signals.append(signal)
            return []

        # 按优先级排序
        priority_order = {"high": 0, "normal": 1, "low": 2}
        sorted_subs = sorted(subs, key=lambda s: priority_order.get(s.priority, 1))

        # 同步调用
        results = []
        for sub in sorted_subs:
            try:
                result = sub.handler(signal)
                results.append(result)
            except Exception as e:
                results.append({"_error": str(e), "_subscriber": sub.subscriber})

        return results

    # ─── Join（等待多个信号） ─────────────────────────

    def publish_and_wait(self, signals: list[Signal],
                         expected_types: list[str] = None,
                         timeout: float = 30.0) -> dict[str, list[Signal]]:
        """
        发布多个信号并等待响应到齐。

        用于 Arbitrator 需要等多个 Region 都输出后才能决策的场景。

        参数:
            signals: 要发布的信号列表
            expected_types: 期望收到的响应信号类型列表
            timeout: 超时秒数（最简版：只是记录，实际不阻塞）

        返回:
            {signal_type: [Signal, ...], ...} 收到的响应
        """
        if expected_types is None:
            # 自动推断：所有信号的 type
            expected_types = list(set(s.type for s in signals if s.type))

        # 确保所有信号有相同的 task_id
        task_id = signals[0].task_id if signals else ""
        for s in signals:
            if not s.task_id:
                s.task_id = task_id

        # 记录发布前的历史长度
        before_count = len(self._signal_history.get(task_id, []))

        # 发布所有信号
        for signal in signals:
            self.publish(signal)

        # 收集响应（同步版：直接从历史里取）
        after_signals = self._signal_history.get(task_id, [])
        new_signals = after_signals[before_count:]

        # 按类型分组
        responses = {}
        for stype in expected_types:
            responses[stype] = [s for s in new_signals if s.type == stype]

        return responses

    # ─── 查询 ──────────────────────────────────────────

    def get_history(self, task_id: str) -> list[Signal]:
        """获取某个 task_id 的所有信号历史"""
        return self._signal_history.get(task_id, [])

    def get_dead_signals(self) -> list[Signal]:
        """获取无订阅的死信号（debug 用）"""
        return self._dead_signals

    def get_subscribers(self) -> dict[str, list[dict]]:
        """查看所有订阅关系（debug 用）"""
        return {
            stype: [{"subscriber": s.subscriber, "priority": s.priority}
                    for s in subs]
            for stype, subs in self._subscribers.items()
        }

    def clear_history(self, task_id: str = None):
        """清理信号历史"""
        if task_id:
            self._signal_history.pop(task_id, None)
        else:
            self._signal_history.clear()

    def clear_dead_signals(self):
        """清理死信号日志"""
        self._dead_signals.clear()
