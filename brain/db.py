"""
SQLite 索引 + 物化视图层。

职责:
  - events 表:         WAL entry 的结构化索引（快速查询）
  - calibration_failures 表: 校准失败记录（物化视图）
  - advisories 表:     校准建议（含状态: pending/accepted/rejected）
  - rebuild_from_wal(): 崩溃恢复 — 从 WAL 完全重建

设计原则:
  - WAL 是真相源，SQLite 是加速层
  - SQLite 损坏 → 从 WAL 重建
  - 单写者（Phase 0 不处理并发写）
"""

from __future__ import annotations
import sqlite3
import json
from pathlib import Path
from typing import Optional, Any
from datetime import datetime, timezone


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BrainDB:
    """SQLite 索引 + 物化视图层"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")  # SQLite 自带 WAL 模式
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_tables()

    def _init_tables(self):
        self.conn.executescript("""
            -- WAL entry 索引
            CREATE TABLE IF NOT EXISTS events (
                seq         INTEGER PRIMARY KEY,
                timestamp   TEXT NOT NULL,
                actor       TEXT NOT NULL,
                event_type  TEXT NOT NULL,
                data        TEXT NOT NULL DEFAULT '{}',
                evidence    TEXT,
                verified    INTEGER DEFAULT 0
            );

            -- 校准失败记录（物化视图）
            CREATE TABLE IF NOT EXISTS calibration_failures (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                seq             INTEGER NOT NULL,
                session_id      TEXT DEFAULT '',
                question_type   TEXT DEFAULT '',
                error_type      TEXT DEFAULT '',
                question_summary TEXT DEFAULT '',
                wrong_answer_summary TEXT DEFAULT '',
                correction_summary TEXT DEFAULT '',
                should_have_verified INTEGER DEFAULT 1,
                is_test_set     INTEGER DEFAULT 0,  -- 0=训练集, 1=测试集(留出)
                created_at      TEXT NOT NULL,
                FOREIGN KEY (seq) REFERENCES events(seq)
            );

            -- 校准建议
            CREATE TABLE IF NOT EXISTS advisories (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                seq         INTEGER NOT NULL,
                version     INTEGER NOT NULL,
                content     TEXT NOT NULL,
                status      TEXT DEFAULT 'pending',
                pre_score   REAL,
                post_score  REAL,
                created_at  TEXT NOT NULL,
                tested_at   TEXT,
                FOREIGN KEY (seq) REFERENCES events(seq)
            );

            -- 索引
            CREATE INDEX IF NOT EXISTS idx_failures_type
                ON calibration_failures(error_type);
            CREATE INDEX IF NOT EXISTS idx_failures_session
                ON calibration_failures(session_id);
            CREATE INDEX IF NOT EXISTS idx_events_actor
                ON events(actor, event_type);
            CREATE INDEX IF NOT EXISTS idx_advisories_status
                ON advisories(status);
        """)
        self.conn.commit()

    # ─── events 表 ──────────────────────────────────────────

    def index_event(self, entry: dict):
        """将 WAL entry 同步到 events 表"""
        self.conn.execute(
            """INSERT OR REPLACE INTO events
               (seq, timestamp, actor, event_type, data, evidence, verified)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                entry["seq"],
                entry["timestamp"],
                entry["actor"],
                entry["event_type"],
                json.dumps(entry.get("data", {}), ensure_ascii=False),
                json.dumps(entry.get("evidence"), ensure_ascii=False)
                    if entry.get("evidence") is not None else None,
                int(entry.get("verified", False)),
            ),
        )
        self.conn.commit()

    def get_event(self, seq: int) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM events WHERE seq = ?", (seq,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["data"] = json.loads(d.get("data") or "{}")
        d["evidence"] = json.loads(d["evidence"]) if d.get("evidence") else None
        return d

    # ─── calibration_failures 表 ────────────────────────────

    def add_calibration_failure(self, seq: int, created_at: str, **fields):
        """记录校准失败"""
        valid_cols = {
            "session_id", "question_type", "error_type",
            "question_summary", "wrong_answer_summary",
            "correction_summary", "should_have_verified",
            "is_test_set",
        }
        filtered = {k: v for k, v in fields.items() if k in valid_cols}
        cols = ", ".join(["seq", "created_at"] + list(filtered.keys()))
        placeholders = ", ".join("?" * (2 + len(filtered)))
        values = [seq, created_at] + list(filtered.values())
        self.conn.execute(
            f"INSERT INTO calibration_failures ({cols}) VALUES ({placeholders})",
            values,
        )
        self.conn.commit()

    def get_recent_failures(self, limit: int = 20) -> list[dict]:
        """获取最近的校准失败（全部）"""
        rows = self.conn.execute(
            """SELECT * FROM calibration_failures
               ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_training_failures(self, limit: int = 50) -> list[dict]:
        """获取训练集失败（Arbitrator 用）"""
        rows = self.conn.execute(
            """SELECT * FROM calibration_failures
               WHERE is_test_set = 0
               ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_test_failures(self) -> list[dict]:
        """获取留出集失败（Verifier 用，Arbitrator 没见过的）"""
        rows = self.conn.execute(
            """SELECT * FROM calibration_failures
               WHERE is_test_set = 1
               ORDER BY created_at ASC"""
        ).fetchall()
        return [dict(r) for r in rows]

    def get_failure_stats(self) -> dict:
        """按错误类型统计"""
        rows = self.conn.execute("""
            SELECT error_type,
                   COUNT(*) as total,
                   COUNT(DISTINCT question_type) as question_types
            FROM calibration_failures
            WHERE error_type != ''
            GROUP BY error_type
            ORDER BY total DESC
        """).fetchall()
        return {
            r["error_type"]: {
                "count": r["total"],
                "question_types": r["question_types"],
            }
            for r in rows
        }

    def get_total_failures(self) -> int:
        """总失败数"""
        row = self.conn.execute(
            "SELECT COUNT(*) as n FROM calibration_failures"
        ).fetchone()
        return row["n"] if row else 0

    # ─── advisories 表 ──────────────────────────────────────

    def add_advisory(self, seq: int, version: int, content: str,
                     created_at: str = None):
        """添加新校准建议"""
        self.conn.execute(
            """INSERT INTO advisories (seq, version, content, status, created_at)
               VALUES (?, ?, ?, 'pending', ?)""",
            (seq, version, content, created_at or utc_now()),
        )
        self.conn.commit()

    def update_advisory_status(self, advisory_id: int, status: str,
                               pre_score: float = None,
                               post_score: float = None):
        """更新建议状态"""
        self.conn.execute(
            """UPDATE advisories
               SET status = ?, pre_score = ?, post_score = ?, tested_at = ?
               WHERE id = ?""",
            (status, pre_score, post_score, utc_now(), advisory_id),
        )
        self.conn.commit()

    def get_pending_advisory(self) -> Optional[dict]:
        """获取最新的 pending 建议"""
        row = self.conn.execute(
            """SELECT * FROM advisories WHERE status = 'pending'
               ORDER BY version DESC LIMIT 1"""
        ).fetchone()
        return dict(row) if row else None

    def get_current_advisory(self) -> Optional[dict]:
        """获取当前生效的（accepted）建议"""
        row = self.conn.execute(
            """SELECT * FROM advisories WHERE status = 'accepted'
               ORDER BY version DESC LIMIT 1"""
        ).fetchone()
        return dict(row) if row else None

    def get_advisory_count(self) -> dict:
        """建议统计"""
        rows = self.conn.execute(
            """SELECT status, COUNT(*) as n FROM advisories GROUP BY status"""
        ).fetchall()
        return {r["status"]: r["n"] for r in rows}

    # ─── 崩溃恢复 ────────────────────────────────────────────

    def rebuild_from_wal(self, wal_entries: list[dict]) -> dict:
        """
        从 WAL 完全重建所有表（崩溃恢复）。

        参数:
            wal_entries: WAL entry 字典列表

        返回:
            重建统计
        """
        # 清空所有表
        self.conn.executescript("""
            DELETE FROM calibration_failures;
            DELETE FROM advisories;
            DELETE FROM events;
        """)

        stats = {"events": 0, "failures": 0, "advisories": 0}

        for entry in wal_entries:
            self.index_event(entry)
            stats["events"] += 1

            event_type = entry.get("event_type", "")
            data = entry.get("data", {})

            if event_type == "failure_recorded":
                self.add_calibration_failure(
                    seq=entry["seq"],
                    created_at=entry.get("timestamp", utc_now()),
                    **data,
                )
                stats["failures"] += 1

            elif event_type in ("advisory_proposed", "advisory_accepted", "advisory_rejected"):
                if "version" in data and "content" in data:
                    self.add_advisory(
                        seq=entry["seq"],
                        version=data["version"],
                        content=data["content"],
                        created_at=entry.get("timestamp", utc_now()),
                    )
                    stats["advisories"] += 1

                    # 如果是 accepted/rejected，更新最新 advisory 状态
                    if event_type in ("advisory_accepted", "advisory_rejected"):
                        row = self.conn.execute(
                            """SELECT id FROM advisories ORDER BY id DESC LIMIT 1"""
                        ).fetchone()
                        if row:
                            status = "accepted" if event_type == "advisory_accepted" else "rejected"
                            self.update_advisory_status(
                                row["id"], status,
                                post_score=data.get("prevention_rate"),
                            )
                elif event_type in ("advisory_accepted", "advisory_rejected"):
                    # advisory 状态变更事件（不含 content，只有 advisory_id/version）
                    adv_version = data.get("advisory_version")
                    status = "accepted" if event_type == "advisory_accepted" else "rejected"
                    if adv_version:
                        row = self.conn.execute(
                            """SELECT id FROM advisories WHERE version = ? ORDER BY id DESC LIMIT 1""",
                            (adv_version,)
                        ).fetchone()
                        if row:
                            self.update_advisory_status(
                                row["id"], status,
                                post_score=data.get("prevention_rate"),
                            )

        self.conn.commit()
        return stats

    def close(self):
        self.conn.close()
