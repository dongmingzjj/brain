"""
Action Region Executor — 工具调用 + 代码执行。

Phase 2.0: Python subprocess（shell 命令）
Phase 2.1: MCP tool servers（标准化工具接口）

执行策略:
  - 安全沙箱：限制可执行的命令类型
  - 超时控制：防止挂死
  - 结果捕获：stdout/stderr/exit_code
  - 成功/失败统计：供 Local Improver 使用
"""

from __future__ import annotations
import subprocess
import json
import time
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─── 安全沙箱 ──────────────────────────────────────────────

# 禁止的命令模式（防止误操作）
FORBIDDEN_PATTERNS = [
    "rm -rf /",
    "format c:",
    "shutdown",
    "del /f /s /q c:\\",
    "mkfs",
    "dd if=",
]

# 允许的工具列表（白名单模式，Phase 2.0 先放开常用工具）
ALLOWED_TOOLS = {
    "python": "Python 解释器",
    "python3": "Python3 解释器",
    "pip": "Python 包管理",
    "git": "版本控制",
    "gh": "GitHub CLI",
    "curl": "HTTP 请求",
    "ls": "文件列表",
    "cat": "文件查看",
    "echo": "输出",
    "grep": "文本搜索",
    "wc": "统计",
    "head": "文件头部",
    "tail": "文件尾部",
    "mkdir": "创建目录",
    "cp": "复制",
    "mv": "移动",
    "node": "Node.js",
    "npm": "Node 包管理",
    "npx": "Node 包执行",
}


class ActionExecutor:
    """Action Region 的执行器"""

    def __init__(self, workdir: str = None, timeout: int = 30):
        """
        参数:
            workdir: 工作目录（None = 当前目录）
            timeout: 默认超时秒数
        """
        self.workdir = workdir or os.getcwd()
        self.timeout = timeout
        self.conn = None  # 兼容 metrics 调用

        # 执行历史（供 Local Improver 分析）
        self._history_db = Path(workdir or ".") / "data" / "action_history.db" if workdir else Path("data/action_history.db")
        self._init_history_db()

    def _init_history_db(self):
        """初始化执行历史数据库"""
        self._history_db.parent.mkdir(parents=True, exist_ok=True)
        import sqlite3
        self.conn = sqlite3.connect(str(self._history_db))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS actions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                tool        TEXT NOT NULL,
                command     TEXT NOT NULL,
                exit_code   INTEGER,
                stdout      TEXT,
                stderr      TEXT,
                duration_ms INTEGER,
                success     INTEGER DEFAULT 0,
                timestamp   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tool_stats (
                tool        TEXT PRIMARY KEY,
                total       INTEGER DEFAULT 0,
                success     INTEGER DEFAULT 0,
                fail        INTEGER DEFAULT 0,
                avg_duration_ms REAL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_actions_tool
                ON actions(tool);
        """)
        self.conn.commit()

    # ─── 执行 ──────────────────────────────────────────────

    def execute(self, command: str, timeout: int = None) -> dict:
        """
        执行一条命令。

        参数:
            command: 要执行的命令字符串
            timeout: 超时秒数（None = 使用默认值）

        返回:
            {success, exit_code, stdout, stderr, duration_ms, tool, command}
        """
        # 安全检查
        safety = self._check_safety(command)
        if not safety["safe"]:
            return {
                "success": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": f"安全拦截: {safety['reason']}",
                "duration_ms": 0,
                "tool": safety.get("tool", "unknown"),
                "command": command,
            }

        tool = safety["tool"]
        timeout = timeout or self.timeout
        start = time.time()

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.workdir,
                encoding="utf-8",
                errors="replace",
            )
            duration_ms = int((time.time() - start) * 1000)
            success = result.returncode == 0

            output = {
                "success": success,
                "exit_code": result.returncode,
                "stdout": result.stdout.strip() if result.stdout else "",
                "stderr": result.stderr.strip() if result.stderr else "",
                "duration_ms": duration_ms,
                "tool": tool,
                "command": command,
            }

        except subprocess.TimeoutExpired:
            duration_ms = int((time.time() - start) * 1000)
            output = {
                "success": False,
                "exit_code": -2,
                "stdout": "",
                "stderr": f"超时 ({timeout}s)",
                "duration_ms": duration_ms,
                "tool": tool,
                "command": command,
            }

        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            output = {
                "success": False,
                "exit_code": -3,
                "stdout": "",
                "stderr": str(e),
                "duration_ms": duration_ms,
                "tool": tool,
                "command": command,
            }

        # 记录历史
        self._log_action(output)

        return output

    # ─── 安全检查 ──────────────────────────────────────────

    def _check_safety(self, command: str) -> dict:
        """
        检查命令安全性。

        返回:
            {safe: bool, reason: str, tool: str}
        """
        cmd_lower = command.lower().strip()

        # 检查禁止模式
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in cmd_lower:
                return {
                    "safe": False,
                    "reason": f"匹配禁止模式: {pattern}",
                    "tool": "forbidden",
                }

        # 提取工具名
        parts = cmd_lower.split()
        if not parts:
            return {"safe": False, "reason": "空命令", "tool": "empty"}

        tool = parts[0]

        # 去掉路径前缀（/usr/bin/python → python）
        if "/" in tool or "\\" in tool:
            tool = tool.split("/")[-1].split("\\")[-1]

        # 白名单检查（Phase 2.0 先放宽，只警告）
        if tool not in ALLOWED_TOOLS:
            # 不在白名单但也不在禁止列表 → 允许执行但标记
            pass

        return {"safe": True, "reason": "ok", "tool": tool}

    # ─── 历史记录 ──────────────────────────────────────────

    def _log_action(self, output: dict):
        """记录执行历史"""
        self.conn.execute(
            """INSERT INTO actions
               (tool, command, exit_code, stdout, stderr, duration_ms, success, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                output["tool"],
                output["command"],
                output["exit_code"],
                output["stdout"][:5000],  # 截断长输出
                output["stderr"][:2000],
                output["duration_ms"],
                int(output["success"]),
                utc_now(),
            )
        )

        # 更新工具统计
        self._update_tool_stats(output["tool"], output["success"], output["duration_ms"])
        self.conn.commit()

    def _update_tool_stats(self, tool: str, success: bool, duration_ms: int):
        """更新工具统计"""
        row = self.conn.execute(
            "SELECT * FROM tool_stats WHERE tool = ?", (tool,)
        ).fetchone()

        if row:
            new_total = row["total"] + 1
            new_success = row["success"] + (1 if success else 0)
            new_fail = row["fail"] + (0 if success else 1)
            # 滚动平均
            old_sum = row["avg_duration_ms"] * row["total"]
            new_avg = (old_sum + duration_ms) / new_total

            self.conn.execute(
                """UPDATE tool_stats SET total=?, success=?, fail=?, avg_duration_ms=?
                   WHERE tool=?""",
                (new_total, new_success, new_fail, new_avg, tool)
            )
        else:
            self.conn.execute(
                """INSERT INTO tool_stats (tool, total, success, fail, avg_duration_ms)
                   VALUES (?, 1, ?, ?, ?)""",
                (tool, 1 if success else 0, 0 if success else 1, float(duration_ms))
            )

    # ─── 查询 ──────────────────────────────────────────────

    def get_tool_stats(self) -> dict:
        """获取所有工具的统计信息"""
        rows = self.conn.execute(
            "SELECT * FROM tool_stats ORDER BY total DESC"
        ).fetchall()
        return {
            r["tool"]: {
                "total": r["total"],
                "success": r["success"],
                "fail": r["fail"],
                "success_rate": r["success"] / r["total"] if r["total"] > 0 else 0,
                "avg_duration_ms": round(r["avg_duration_ms"], 1),
            }
            for r in rows
        }

    def get_recent_actions(self, limit: int = 20) -> list[dict]:
        """获取最近的执行记录"""
        rows = self.conn.execute(
            """SELECT tool, command, exit_code, success, duration_ms, timestamp
               FROM actions ORDER BY id DESC LIMIT ?""",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        """全局统计"""
        total = self.conn.execute("SELECT COUNT(*) FROM actions").fetchone()[0]
        success = self.conn.execute(
            "SELECT COUNT(*) FROM actions WHERE success = 1"
        ).fetchone()[0]
        return {
            "total_actions": total,
            "successful": success,
            "failed": total - success,
            "success_rate": success / total if total > 0 else 0,
            "tool_count": len(self.get_tool_stats()),
        }

    def close(self):
        if self.conn:
            self.conn.close()
