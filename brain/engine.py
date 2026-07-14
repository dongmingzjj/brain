"""
Brain Engine — 连接三个 Region 的主控引擎。

职责：
  1. 初始化三个种子 Region + Event Bus
  2. 接收原始输入 → Perception 理解 → Memory 检索 → Action 执行
  3. 通过 Event Bus 传递信号，不直接调用
  4. Global Workspace: 信号竞争升级，低置信度时通知 Arbitrator
  5. 跨区评估: Region 互相打标签

信号流：
  user_input → [Perception] → perception_result
                                   ↓
              [Memory] ← recall_request
                 ↓
              recall_result → [Cross-Region Eval] → decision
                                                    ↓
              action_request → [Action] → action_result
                                            ↓
              [Memory] ← store_experience
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from brain.event_bus import EventBus, Signal
from brain.cross_region import CrossRegionEvaluator
from brain.regions.perception.executor import PerceptionExecutor
from brain.regions.perception.metrics import PerceptionMetrics
from brain.regions.memory.executor import MemoryExecutor
from brain.regions.memory.metrics import MemoryMetrics
from brain.regions.action.executor import ActionExecutor
from brain.regions.action.metrics import ActionMetrics


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BrainEngine:
    """Brain 主控引擎 — 连接所有 Region"""

    def __init__(self, brain_dir: str = "."):
        import os
        self.brain_dir = brain_dir
        data_dir = os.path.join(brain_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        # Event Bus
        self.bus = EventBus()

        # Regions
        self.perception = PerceptionExecutor(
            db_path=os.path.join(data_dir, "perception.db")
        )
        self.memory = MemoryExecutor(
            db_path=os.path.join(data_dir, "memory.db")
        )
        self.action = ActionExecutor(
            workdir=brain_dir, timeout=30
        )

        # Metrics
        self.perception_metrics = PerceptionMetrics(self.perception)
        self.memory_metrics = MemoryMetrics(self.memory)
        self.action_metrics = ActionMetrics(self.action)

        # Cross-Region Evaluator
        self.evaluator = CrossRegionEvaluator(
            self.bus, self.memory, self.action
        )

        # 注册 Event Bus 订阅
        self._register_subscriptions()

        # 运行统计
        self.cycle_count = 0
        self.last_results = {}

    def _register_subscriptions(self):
        """注册各 Region 的 Event Bus 订阅"""

        # Memory 订阅 perception_result → 自动检索
        @self.bus.subscribe("perception_result", subscriber="memory")
        def on_perception(signal: Signal):
            perception_data = signal.content
            query = perception_data.get("summary", "")

            results = self.memory.retrieve(query, top_k=3)
            if results:
                self.bus.publish(Signal(
                    source="region:memory",
                    type="recall_result",
                    task_id=signal.task_id,
                    content={"results": results, "query": query},
                ))

        # Action 订阅 action_request → 执行命令
        @self.bus.subscribe("action_request", subscriber="action")
        def on_action(signal: Signal):
            command = signal.content.get("command", "")
            result = self.action.execute(command)

            self.bus.publish(Signal(
                source="region:action",
                type="action_result",
                task_id=signal.task_id,
                content=result,
            ))

        # 跨区评估订阅 action_result → Memory 评估
        @self.bus.subscribe("action_result", subscriber="cross_eval")
        def on_action_result(signal: Signal):
            action_data = signal.content
            tag = self.evaluator.memory_evaluates_action(
                action_data, task_id=signal.task_id
            )
            self.last_results[f"tag_{signal.task_id}"] = tag.to_dict()

    # ─── 核心循环 ──────────────────────────────────────────

    def process(self, user_input: str, task_id: str = None) -> dict:
        """
        处理一个用户输入，走完整链路。

        流程:
          1. Perception 理解输入
          2. 通过 Bus 发布 perception_result
          3. Memory 自动检索（订阅触发）
          4. 如果有命令需要执行 → 发布 action_request
          5. Action 执行（订阅触发）
          6. 跨区评估

        返回:
            {
                perception: {...},
                recall: [...],
                action: {...} | None,
                tags: [...],
                decision: {...},
            }
        """
        self.cycle_count += 1
        if not task_id:
            task_id = f"task_{self.cycle_count}"

        result = {
            "task_id": task_id,
            "input": user_input[:100],
            "perception": None,
            "recall": [],
            "action": None,
            "tags": [],
            "decision": None,
        }

        # 1. Perception
        perception_result = self.perception.execute(user_input)
        result["perception"] = perception_result

        # 2. 发布 perception_result 到 Bus → Memory 自动检索
        self.bus.publish(Signal(
            source="region:perception",
            type="perception_result",
            task_id=task_id,
            content=perception_result,
        ))

        # 3. 从历史中获取 Memory 的检索结果
        history = self.bus.get_history(task_id)
        recall_signals = [s for s in history if s.type == "recall_result"]
        if recall_signals:
            result["recall"] = recall_signals[-1].content.get("results", [])

        # 4. 如果 Perception 检测到命令意图 → 发布 action_request
        if perception_result["intent"] == "command":
            # 从实体中提取命令
            command = self._extract_command(user_input, perception_result)
            if command:
                self.bus.publish(Signal(
                    source="engine",
                    type="action_request",
                    task_id=task_id,
                    content={"command": command},
                ))

                # 获取 Action 结果
                action_signals = [s for s in self.bus.get_history(task_id)
                                  if s.type == "action_result"]
                if action_signals:
                    result["action"] = action_signals[-1].content

                    # 收集标签
                    tag_signals = [s for s in self.bus.get_history(task_id)
                                   if s.type == "action_evaluation"]
                    result["tags"] = [s.content for s in tag_signals]

        # 5. 决策
        if result["tags"]:
            from brain.cross_region import Tag
            tags = []
            for tag_data in result["tags"]:
                tags.append(Tag(
                    source_region=tag_data.get("source", ""),
                    target_region=tag_data.get("target", ""),
                    confidence=tag_data.get("confidence", 0.5),
                    recommendation=tag_data.get("recommendation", "approve"),
                    reason=tag_data.get("reason", ""),
                ))
            result["decision"] = self.evaluator.make_decision(tags)

        # 6. Global Workspace: 低置信度升级
        if perception_result["confidence"] < 0.3:
            result["escalation"] = "low_confidence"
        elif perception_result["sentiment"] == "urgent":
            result["escalation"] = "urgent"

        self.last_results[task_id] = result
        self.bus.clear_history(task_id)
        return result

    def _extract_command(self, text: str, perception: dict) -> str | None:
        """从用户输入和感知结果中提取要执行的命令"""
        # 如果有 code_block 或 command 实体
        entities = perception.get("entities", {})
        if "command" in entities:
            # 去掉反引号
            cmd = entities["command"][0].strip("`")
            return cmd

        return None

    # ─── 状态查询 ──────────────────────────────────────────

    def get_status(self) -> dict:
        """获取 Brain 全局状态"""
        return {
            "cycle_count": self.cycle_count,
            "perception": self.perception_metrics.compute_all(),
            "memory": self.memory_metrics.compute_all(),
            "action": self.action_metrics.compute_all(),
            "bus_subscribers": self.bus.get_subscribers(),
            "dead_signals": len(self.bus.get_dead_signals()),
        }

    def close(self):
        self.perception.close()
        self.memory.close()
        self.action.close()
