"""
Memory Region Executor — SQLite 存储 + FTS5 BM25 检索。

检索引擎：
  Phase 1.0: LIKE 关键词匹配（已废弃）
  Phase 1.1: FTS5 trigram 分词 + BM25 排序（当前）
  Phase 1.2: 向量相似度（未来，Qdrant/embedding）

trigram 分词器对中英文混合内容都有很好的支持。
"""

from __future__ import annotations
import sqlite3
import json
import re
import math
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

        # FTS5 虚拟表（trigram 分词，支持中文）
        try:
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                USING fts5(key, value, content='memories',
                           tokenize='trigram')
            """)
        except Exception:
            # 如果 trigram 不支持，退回 unicode61
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                USING fts5(key, value, content='memories',
                           tokenize='unicode61')
            """)

        # 创建 FTS5 触发器（自动同步）
        self.conn.executescript("""
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories
            BEGIN
                INSERT INTO memories_fts(rowid, key, value)
                VALUES (new.id, new.key, new.value);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories
            BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, key, value)
                VALUES ('delete', old.id, old.key, old.value);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories
            BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, key, value)
                VALUES ('delete', old.id, old.key, old.value);
                INSERT INTO memories_fts(rowid, key, value)
                VALUES (new.id, new.key, new.value);
            END;
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
        FTS5 BM25 检索记忆。

        引擎: FTS5 trigram 分词 + BM25 排序
        比 LIKE 查询更准确：BM25 考虑词频、文档长度、逆文档频率。
        trigram 分词器对中英文混合内容都有支持。

        返回:
            [{id, key, value, mem_type, importance, score, access_count}, ...]
        """
        if not query or not query.strip():
            return []

        # 构建 FTS5 MATCH 查询
        # 用 OR 连接多个词，支持部分匹配
        fts_query = self._build_fts_query(query)
        if not fts_query:
            return []

        # 执行 FTS5 查询 + BM25 排序
        sql = """
            SELECT m.id, m.key, m.value, m.mem_type, m.importance,
                   m.access_count, m.tags,
                   rank AS bm25_rank
            FROM memories_fts f
            JOIN memories m ON m.id = f.rowid
            WHERE memories_fts MATCH ?
        """
        params = [fts_query]

        if mem_type:
            sql += " AND m.mem_type = ?"
            params.append(mem_type)

        if min_importance > 0:
            sql += " AND m.importance >= ?"
            params.append(min_importance)

        # FTS5 rank 是负数（越小越相关），取多取一些然后用重要性加权重排
        sql += " ORDER BY rank LIMIT ?"
        params.append(top_k * 3)  # 多取 3 倍候选，后面用重要性加权重排

        try:
            rows = self.conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            # FTS5 查询语法错误，退回 LIKE 查询
            return self._retrieve_like(query, top_k, mem_type, min_importance)

        if not rows:
            # FTS5 没匹配到，退回 LIKE 查询
            return self._retrieve_like(query, top_k, mem_type, min_importance)

        results = []
        for row in rows:
            # BM25 分数转换：FTS5 rank 是负数，转为正数（越小越相关 → 越大越好）
            bm25_score = -row["bm25_rank"] if row["bm25_rank"] else 0

            # 混合分数：BM25 相关性 + 重要性 + 访问频率
            importance = row["importance"] or 0.5
            access_bonus = min(row["access_count"] or 0, 10) / 100

            # 归一化 BM25（粗略：除以 10 把分数压到 0-1 区间附近）
            bm25_normalized = min(bm25_score / 10, 1.0)

            score = bm25_normalized * 0.6 + importance * 0.3 + access_bonus * 0.1

            # 记录访问
            self._log_access(row["id"], query, score)

            results.append({
                "id": row["id"],
                "key": row["key"],
                "value": row["value"],
                "mem_type": row["mem_type"],
                "importance": importance,
                "access_count": row["access_count"] + 1,
                "tags": json.loads(row["tags"] or "[]"),
                "score": round(score, 4),
                "bm25_rank": round(bm25_score, 4),
            })

        # 按混合分数重排
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def _build_fts_query(self, query: str) -> str:
        """
        将用户查询转换为 FTS5 MATCH 表达式。

        策略：
        - 中文 2-4 字词：用引号包起来做精确 trigram 匹配
        - 英文 3+ 字母词：用引号 + * 做前缀匹配
        - 用 OR 连接
        """
        keywords = self._extract_keywords(query)
        if not keywords:
            # 如果关键词提取失败，直接用原始查询做 trigram 搜索
            stripped = query.strip()
            if len(stripped) >= 3:
                return f'"{stripped}"'
            return ""

        terms = []
        for kw in keywords[:10]:
            if len(kw) >= 2:
                terms.append(f'"{kw}"')

        if not terms:
            return ""

        return " OR ".join(terms)

    def _retrieve_like(self, query: str, top_k: int = 5,
                       mem_type: str = None,
                       min_importance: float = 0.0) -> list[dict]:
        """LIKE 退回检索（FTS5 不可用时使用）"""
        keywords = self._extract_keywords(query)
        if not keywords:
            return []

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
            score = self._compute_score(row, keywords)
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
