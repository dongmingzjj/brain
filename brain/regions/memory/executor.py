"""
Memory Region Executor — SQLite 存储 + 关键词检索。

Phase 1 最简实现：
  - SQLite 存储（内存条目 + 元数据）
  - 关键词匹配检索（LIKE 查询，后续升级 BM25）
  - 访问追踪（供 Local Improver 使用）

后续升级路径：
  - SQLite → Qdrant（向量检索）
  - LIKE → BM25 → 向量相似度
  - 单机 → 分布式
"""

from __future__ import annotations
import sqlite3
import json
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryExecutor:
    """Memory Region 的执行器"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                key         TEXT NOT NULL,
                value       TEXT NOT NULL,
                mem_type    TEXT DEFAULT 'fact',
                importance  REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                created_at  TEXT NOT NULL,
                tags        TEXT DEFAULT '[]',
                source      TEXT DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_memories_type
                ON memories(mem_type);
            CREATE INDEX IF NOT EXISTS idx_memories_importance
                ON memories(importance DESC);

            -- 访问日志（供 Local Improver 分析）
            CREATE TABLE IF NOT EXISTS access_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id   INTEGER NOT NULL,
                query       TEXT,
                relevance   REAL,
                timestamp   TEXT NOT NULL,
                FOREIGN KEY (memory_id) REFERENCES memories(id)
            );
        """)
        self.conn.commit()

    # ─── 存储 ──────────────────────────────────────────────

    def store(self, key: str, value: str,
              mem_type: str = "fact",
              importance: float = 0.5,
              tags: list[str] = None,
              source: str = "") -> int:
        """存储一条记忆，返回 id"""
        ts = utc_now()
        cursor = self.conn.execute(
            """INSERT INTO memories
               (key, value, mem_type, importance, created_at, tags, source)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (key, value, mem_type, importance, ts,
             json.dumps(tags or []), source)
        )
        self.conn.commit()
        return cursor.lastrowid

    # ─── 检索 ──────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5,
                 mem_type: str = None,
                 min_importance: float = 0.0) -> list[dict]:
        """
        关键词检索记忆。

        Phase 1: 简单 LIKE 匹配（按关键词拆分查询）
        后续: BM25 / 向量相似度

        返回:
            [{id, key, value, mem_type, importance, score, access_count}, ...]
        """
        # 提取查询关键词
        keywords = self._extract_keywords(query)
        if not keywords:
            return []

        # 构建 LIKE 查询（任一关键词匹配即返回）
        conditions = []
        params = []
        for kw in keywords:
            conditions.append("(key LIKE ? OR value LIKE ?)")
            params.extend([f"%{kw}%", f"%{kw}%"])

        where_clause = " OR ".join(conditions)

        if mem_type:
            where_clause = f"({where_clause}) AND mem_type = ?"
            params.append(mem_type)

        if min_importance > 0:
            where_clause = f"({where_clause}) AND importance >= ?"
            params.append(min_importance)

        rows = self.conn.execute(
            f"""SELECT id, key, value, mem_type, importance, access_count, tags
                FROM memories
                WHERE {where_clause}
                ORDER BY importance DESC, access_count DESC
                LIMIT ?""",
            params + [top_k]
        ).fetchall()

        results = []
        for row in rows:
            # 计算简单相关性分数
            score = self._compute_score(row, keywords)

            # 记录访问
            self._log_access(row["id"], query, score)

            results.append({
                "id": row["id"],
                "key": row["key"],
                "value": row["value"],
                "mem_type": row["mem_type"],
                "importance": row["importance"],
                "access_count": row["access_count"] + 1,
                "tags": json.loads(row["tags"] or "[]"),
                "score": score,
            })

        # 按分数排序
        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    # ─── 遗忘 ──────────────────────────────────────────────

    def forget(self, memory_id: int) -> bool:
        """删除一条记忆"""
        cursor = self.conn.execute(
            "DELETE FROM memories WHERE id = ?", (memory_id,)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def forget_by_importance(self, threshold: float = 0.2) -> int:
        """删除重要性低于阈值的记忆"""
        cursor = self.conn.execute(
            "DELETE FROM memories WHERE importance < ? AND access_count = 0",
            (threshold,)
        )
        self.conn.commit()
        return cursor.rowcount

    # ─── 更新 ──────────────────────────────────────────────

    def update_importance(self, memory_id: int, new_importance: float):
        """更新记忆重要性"""
        self.conn.execute(
            "UPDATE memories SET importance = ? WHERE id = ?",
            (new_importance, memory_id)
        )
        self.conn.commit()

    def decay_importance(self, decay_rate: float = 0.01):
        """重要性衰减（定期调用，模拟遗忘曲线）"""
        self.conn.execute(
            """UPDATE memories
               SET importance = MAX(0.0, importance - ?)
               WHERE last_accessed IS NULL
                  OR last_accessed < datetime('now', '-7 days')""",
            (decay_rate,)
        )
        self.conn.commit()

    # ─── 统计 ──────────────────────────────────────────────

    def get_stats(self) -> dict:
        """返回 Memory Region 的运行指标"""
        total = self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        by_type = self.conn.execute(
            "SELECT mem_type, COUNT(*) as n FROM memories GROUP BY mem_type"
        ).fetchall()
        avg_importance = self.conn.execute(
            "SELECT AVG(importance) FROM memories"
        ).fetchone()[0] or 0
        total_accesses = self.conn.execute(
            "SELECT SUM(access_count) FROM memories"
        ).fetchone()[0] or 0

        return {
            "total_memories": total,
            "by_type": {r["mem_type"]: r["n"] for r in by_type},
            "avg_importance": round(avg_importance, 3),
            "total_accesses": total_accesses,
        }

    def get_access_log_stats(self) -> dict:
        """访问日志统计（供 Local Improver 使用）"""
        total = self.conn.execute("SELECT COUNT(*) FROM access_log").fetchone()[0]
        avg_relevance = self.conn.execute(
            "SELECT AVG(relevance) FROM access_log WHERE relevance IS NOT NULL"
        ).fetchone()[0] or 0

        # 最常被访问的记忆
        top_accessed = self.conn.execute("""
            SELECT m.id, m.key, m.access_count, m.importance
            FROM memories m
            ORDER BY m.access_count DESC
            LIMIT 5
        """).fetchall()

        return {
            "total_queries": total,
            "avg_relevance": round(avg_relevance, 3),
            "top_accessed": [
                {"id": r["id"], "key": r["key"],
                 "access_count": r["access_count"],
                 "importance": r["importance"]}
                for r in top_accessed
            ],
        }

    # ─── 内部方法 ──────────────────────────────────────────

    def _extract_keywords(self, text: str) -> list[str]:
        """从文本提取关键词（简单分词）"""
        # 中文：取 2-4 字的连续汉字
        cn_words = re.findall(r'[\u4e00-\u9fff]{2,4}', text)
        # 英文：取 3+ 字母的单词
        en_words = re.findall(r'[a-zA-Z_]{3,}', text.lower())
        # 去重
        all_words = list(set(cn_words + en_words))
        # 过滤停用词
        stopwords = {'这个', '那个', '什么', '怎么', '如何', '可以', '需要',
                     '应该', '就是', '不是', '还是', '或者', '以及', '但是',
                     'the', 'and', 'for', 'this', 'that', 'with', 'from'}
        return [w for w in all_words if w not in stopwords]

    def _compute_score(self, row, keywords: list[str]) -> float:
        """计算检索相关性分数"""
        text = f"{row['key']} {row['value']}".lower()
        # 关键词命中率
        hits = sum(1 for kw in keywords if kw.lower() in text)
        hit_rate = hits / len(keywords) if keywords else 0
        # 重要性加权
        importance = row["importance"] or 0.5
        # 访问频率加权（常用的记忆得分更高）
        access_bonus = min(row["access_count"] or 0, 10) / 100

        return hit_rate * 0.6 + importance * 0.3 + access_bonus * 0.1

    def _log_access(self, memory_id: int, query: str, relevance: float):
        """记录访问日志"""
        ts = utc_now()
        self.conn.execute(
            """INSERT INTO access_log (memory_id, query, relevance, timestamp)
               VALUES (?, ?, ?, ?)""",
            (memory_id, query, relevance, ts)
        )
        # 更新访问计数
        self.conn.execute(
            """UPDATE memories
               SET access_count = access_count + 1, last_accessed = ?
               WHERE id = ?""",
            (ts, memory_id)
        )
        self.conn.commit()

    def close(self):
        self.conn.close()
