"""
WAL (Write-Ahead Log) — 分片追加日志，真相源。

设计:
  - 分片文件: 000001.wal, 000002.wal, ...
  - 每个文件 N 行 JSON，一行一条 entry
  - 单写者追加写入（不允许并发写）
  - 支持顺序读取 + 按 seq 范围读取
  - 支持崩溃恢复（从最后一个分片恢复 seq）

Phase 0 注意:
  - 不处理并发写（Phase 0 单线程）
  - 不实现 checksum（Phase 1 加）
  - 不实现压缩归档（Phase 1 加）
"""

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class WALWriter:
    """分片 WAL 写入器 — 单写者，追加写入"""

    def __init__(self, wal_dir: str, max_entries_per_shard: int = 10000):
        self.wal_dir = Path(wal_dir)
        self.wal_dir.mkdir(parents=True, exist_ok=True)
        self.max_entries_per_shard = max_entries_per_shard
        self._current_seq = self._recover_seq()
        self._current_shard = self._recover_current_shard()
        self._shard_entry_count = self._count_shard_entries(self._current_shard)

    def append(self, actor: str, event_type: str,
               data: dict, evidence: dict | None = None,
               verified: bool = False,
               timestamp: str | None = None) -> int:
        """
        追加一条 WAL entry，返回 seq。

        参数:
            actor:        谁产生的事件 ("capture" | "arbitrator" | "verifier")
            event_type:   事件类型 ("failure_recorded" | "advisory_proposed" | ...)
            data:         事件具体数据
            evidence:     量化证据（可选）
            verified:     是否经验证器检查
            timestamp:    时间戳（可选，默认当前时间）

        返回:
            seq 号
        """
        self._current_seq += 1
        entry = {
            "seq": self._current_seq,
            "timestamp": timestamp or utc_now(),
            "actor": actor,
            "event_type": event_type,
            "data": data,
            "evidence": evidence,
            "verified": verified,
        }
        line = json.dumps(entry, ensure_ascii=False)

        # 检查是否需要轮转分片
        if self._shard_entry_count >= self.max_entries_per_shard:
            self._current_shard += 1
            self._shard_entry_count = 0

        shard_path = self._shard_path(self._current_shard)
        with open(shard_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        self._shard_entry_count += 1
        return self._current_seq

    def read_all(self, from_seq: int = 0) -> list[dict]:
        """
        读取所有 entry（用于重建 SQLite）。

        参数:
            from_seq: 只返回 seq > from_seq 的 entry

        返回:
            entry 字典列表（按 seq 升序）
        """
        entries = []
        for shard_id in self._all_shard_ids():
            shard_path = self._shard_path(shard_id)
            if not shard_path.exists():
                continue
            with open(shard_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        # Phase 0: 跳过损坏行（Phase 1 加 checksum 恢复）
                        continue
                    if entry.get("seq", 0) > from_seq:
                        entries.append(entry)
        entries.sort(key=lambda e: e.get("seq", 0))
        return entries

    def read_latest(self, n: int = 1) -> list[dict]:
        """读取最近 N 条 entry"""
        all_entries = self.read_all()
        return all_entries[-n:] if n > 0 else []

    def get_seq(self) -> int:
        """获取当前 seq（用于外部检查）"""
        return self._current_seq

    def rebuild_check(self) -> dict:
        """
        崩溃恢复检查 — 验证 WAL 完整性。

        返回:
            {total_entries, first_seq, last_seq, shards, integrity: "ok"|"warning"}
        """
        entries = self.read_all()
        if not entries:
            return {"total_entries": 0, "first_seq": 0, "last_seq": 0,
                    "shards": len(self._all_shard_ids()), "integrity": "ok"}

        seqs = [e["seq"] for e in entries]
        expected = list(range(seqs[0], seqs[-1] + 1))
        integrity = "ok" if seqs == expected else "warning"

        return {
            "total_entries": len(entries),
            "first_seq": seqs[0],
            "last_seq": seqs[-1],
            "shards": len(self._all_shard_ids()),
            "integrity": integrity,
        }

    # ─── 内部方法 ────────────────────────────────────────────

    def _shard_path(self, shard_id: int) -> Path:
        return self.wal_dir / f"{shard_id:06d}.wal"

    def _all_shard_ids(self) -> list[int]:
        """所有分片 ID，升序"""
        ids = []
        for p in self.wal_dir.glob("*.wal"):
            name = p.stem
            if name.isdigit():
                ids.append(int(name))
        return sorted(ids)

    def _recover_seq(self) -> int:
        """从最后一个分片恢复 seq"""
        shard_ids = self._all_shard_ids()
        if not shard_ids:
            return 0
        last_seq = 0
        shard_path = self._shard_path(shard_ids[-1])
        with open(shard_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    last_seq = max(last_seq, entry.get("seq", 0))
                except json.JSONDecodeError:
                    continue
        return last_seq

    def _recover_current_shard(self) -> int:
        """恢复当前分片号"""
        shard_ids = self._all_shard_ids()
        return shard_ids[-1] if shard_ids else 1

    def _count_shard_entries(self, shard_id: int) -> int:
        """统计某分片的 entry 数"""
        path = self._shard_path(shard_id)
        if not path.exists():
            return 0
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
