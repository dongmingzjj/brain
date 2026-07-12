"""BSP (Brain Signal Protocol) — 信号协议 + WAL 条目定义"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional
import json


def utc_now() -> str:
    """UTC ISO 时间戳"""
    return datetime.now(timezone.utc).isoformat()


# ─── BSP 信号 ────────────────────────────────────────────────

@dataclass
class Signal:
    """
    BSP 信号 — Phase 0 简化版。
    完整版 3 层 9 字段在架构文档 §5 中定义。
    """

    # envelope 层
    source: str                           # "capture" | "arbitrator" | "verifier"
    target: str = "broadcast"             # 路由目标
    timestamp: str = field(default_factory=utc_now)
    task_id: str = ""                     # 任务 ID（Phase 0 暂不区分多任务）

    # signal 层
    type: str = ""                        # "calibration_failure" | "advisory" | "verification"
    confidence: float = 1.0
    content: dict[str, Any] = field(default_factory=dict)

    # state_snapshot 层（Phase 0 暂不使用）
    phase: str = ""                       # "perception" | "arbitration" | "execution" | "reflection"

    def to_dict(self) -> dict:
        """序列化为字典（用于 WAL 存储 / Event Bus 传输）"""
        return {
            "envelope": {
                "source": self.source,
                "target": self.target,
                "timestamp": self.timestamp,
                "task_id": self.task_id,
            },
            "signal": {
                "type": self.type,
                "confidence": self.confidence,
                "content": self.content,
            },
            "state_snapshot": {
                "phase": self.phase,
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> Signal:
        """从字典反序列化"""
        env = d.get("envelope", {})
        sig = d.get("signal", {})
        snap = d.get("state_snapshot", {})
        return cls(
            source=env.get("source", ""),
            target=env.get("target", "broadcast"),
            timestamp=env.get("timestamp", ""),
            task_id=env.get("task_id", ""),
            type=sig.get("type", ""),
            confidence=sig.get("confidence", 1.0),
            content=sig.get("content", {}),
            phase=snap.get("phase", ""),
        )


# ─── WAL 条目 ────────────────────────────────────────────────

@dataclass
class WALEntry:
    """
    WAL 条目 — 不可变事件记录，分片存储。

    每个 entry 代表系统中发生的一个事件：
    - failure_recorded:    capture 模块记录了一条校准失败
    - advisory_proposed:   arbitrator 生成了新校准建议
    - advisory_accepted:   verifier 通过了建议
    - advisory_rejected:   verifier 拒绝了建议
    """

    seq: int                              # 全局递增序列号
    timestamp: str                        # ISO 时间戳
    actor: str                            # "capture" | "arbitrator" | "verifier"
    event_type: str                       # 事件类型
    data: dict[str, Any] = field(default_factory=dict)
    evidence: Optional[dict] = None       # 量化证据
    verified: bool = False                # 是否经验证器检查

    def to_json_line(self) -> str:
        """序列化为 WAL 文件中的一行 JSON"""
        return json.dumps({
            "seq": self.seq,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "event_type": self.event_type,
            "data": self.data,
            "evidence": self.evidence,
            "verified": self.verified,
        }, ensure_ascii=False)

    @classmethod
    def from_json_line(cls, line: str) -> WALEntry:
        """从 WAL 文件的一行反序列化"""
        d = json.loads(line.strip())
        return cls(**d)

    def to_dict(self) -> dict:
        """转为字典"""
        return asdict(self)
