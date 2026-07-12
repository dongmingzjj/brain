"""
Hermes Session DB 读取器 — 从 Hermes state.db 提取对话轮次。

职责:
  - 连接 Hermes state.db
  - 提取 user + assistant 消息对
  - 按来源/时间/长度过滤
  - 供 CalibrationCapture 使用
"""

from __future__ import annotations
import sqlite3
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ConversationTurn:
    """一轮对话：用户消息 + AI 回复"""
    session_id: str
    session_title: str
    user_msg: str
    assistant_msg: str
    timestamp: float
    user_msg_id: int
    assistant_msg_id: int


class HermesSessionReader:
    """从 Hermes state.db 读取对话轮次"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(
                os.environ.get("LOCALAPPDATA", r"C:\Users\Administrator\AppData\Local"),
                "hermes", "state.db"
            )
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def get_turns(self,
                  min_assistant_len: int = 50,
                  exclude_subagent: bool = True,
                  exclude_compacted: bool = True,
                  session_limit: int = None,
                  offset: int = 0) -> list[ConversationTurn]:
        """
        提取所有 user+assistant 对话轮次。

        参数:
            min_assistant_len: assistant 消息最小长度（跳过太短的）
            exclude_subagent: 排除子代理会话
            exclude_compacted: 排除已压缩的消息
            session_limit: 最多处理多少个会话
            offset: 跳过前 N 个会话

        返回:
            ConversationTurn 列表（按时间升序）
        """
        # 1. 获取候选会话
        where_clauses = ["s.message_count > 2"]
        if exclude_subagent:
            where_clauses.append("s.source != 'subagent'")

        where_sql = " AND ".join(where_clauses)
        limit_sql = f"LIMIT {session_limit}" if session_limit else ""
        offset_sql = f"OFFSET {offset}" if offset else ""

        sessions = self.conn.execute(f"""
            SELECT s.id, s.title, s.source
            FROM sessions s
            WHERE {where_sql}
            ORDER BY s.started_at ASC
            {limit_sql} {offset_sql}
        """).fetchall()

        # 2. 提取每个会话的消息对
        turns = []
        for session in sessions:
            session_id = session["id"]
            session_title = session["title"] or ""

            compacted_filter = "AND m.compacted = 0" if exclude_compacted else ""

            messages = self.conn.execute(f"""
                SELECT m.id, m.role, m.content, m.timestamp, m.tool_calls
                FROM messages m
                WHERE m.session_id = ?
                {compacted_filter}
                AND m.role IN ('user', 'assistant')
                AND m.content IS NOT NULL
                ORDER BY m.timestamp ASC
            """, (session_id,)).fetchall()

            # 配对：每个 assistant 消息找它前面最近的 user 消息
            last_user = None
            for msg in messages:
                if msg["role"] == "user":
                    # 跳过系统消息和空内容
                    content = msg["content"] or ""
                    if content.startswith("[OUT-OF-BAND") or content.startswith("[IMPORTANT"):
                        continue
                    if len(content.strip()) < 3:
                        continue
                    last_user = msg
                elif msg["role"] == "assistant" and last_user:
                    content = msg["content"] or ""
                    if len(content.strip()) < min_assistant_len:
                        continue
                    # 跳过纯工具调用的 assistant 消息（没有实际文本回复）
                    if msg["tool_calls"] and len(content.strip()) < 20:
                        continue

                    turns.append(ConversationTurn(
                        session_id=session_id,
                        session_title=session_title,
                        user_msg=last_user["content"].strip(),
                        assistant_msg=content.strip(),
                        timestamp=msg["timestamp"],
                        user_msg_id=last_user["id"],
                        assistant_msg_id=msg["id"],
                    ))
                    last_user = None  # 避免重复配对

        return turns

    def get_turn_count(self, **kwargs) -> int:
        """获取轮次总数（不实际提取内容，用于预估）"""
        return len(self.get_turns(**kwargs))

    def get_sessions_summary(self) -> list[dict]:
        """获取所有会话的摘要（用于报告）"""
        rows = self.conn.execute("""
            SELECT s.id, s.title, s.source, s.message_count, s.started_at,
                   (SELECT COUNT(*) FROM messages m 
                    WHERE m.session_id = s.id AND m.role = 'user') as user_count,
                   (SELECT COUNT(*) FROM messages m 
                    WHERE m.session_id = s.id AND m.role = 'assistant') as assistant_count
            FROM sessions s
            WHERE s.source != 'subagent'
            ORDER BY s.started_at DESC
        """).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self.conn.close()
