"""
Brain Watchdog — 持续运行模式。

功能：
  1. 定期检查 Hermes session DB 有没有新对话
  2. 新对话积累到阈值 → 自动触发校准循环
  3. 结果写入状态文件（供外部查询）
  4. 可作为后台进程运行

用法：
  python run_watchdog.py                # 前台运行（Ctrl+C 停止）
  python run_watchdog.py --status       # 查看当前状态
  python run_watchdog.py --once         # 跑一次检查后退出
  python run_watchdog.py --interval 30  # 每 30 分钟检查一次

状态文件：data/watchdog_status.json
"""

import sys
import os
import json
import time
import signal
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brain.config import BrainConfig
from brain.hermes_db import HermesSessionReader
from run_auto import run_cycle, utc_now


# ─── 状态管理 ──────────────────────────────────────────────

STATUS_FILE = None  # 初始化时设置


def load_status() -> dict:
    """加载状态文件"""
    if STATUS_FILE and Path(STATUS_FILE).exists():
        with open(STATUS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "last_check": None,
        "last_run": None,
        "total_cycles": 0,
        "last_seen_turns": 0,
        "pending_turns": 0,
        "last_verdict": None,
        "last_injected": False,
        "last_errors": [],
        "prompt_version": 1,
    }


def save_status(status: dict):
    """保存状态文件"""
    if STATUS_FILE:
        Path(STATUS_FILE).parent.mkdir(parents=True, exist_ok=True)
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)


# ─── 检查逻辑 ──────────────────────────────────────────────

def count_new_turns(last_count: int) -> int:
    """统计自上次检查以来的新对话轮次"""
    try:
        reader = HermesSessionReader()
        turns = reader.get_turns(min_assistant_len=80, exclude_subagent=True)
        reader.close()
        current = len(turns)
        new = max(0, current - last_count)
        return new, current
    except Exception as e:
        print(f"  [WARN] 读取 session DB 失败: {e}")
        return 0, last_count


def check_and_run(threshold: int = 5, batch_size: int = 20) -> dict:
    """
    检查是否有足够新对话，有则触发校准循环。

    参数:
        threshold: 新对话数达到此阈值才触发
        batch_size: 每次扫描的最大轮次

    返回:
        {triggered: bool, new_turns: int, result: dict | None}
    """
    status = load_status()

    # 统计新对话
    new_turns, current_total = count_new_turns(status.get("last_seen_turns", 0))
    status["pending_turns"] = new_turns
    status["last_check"] = utc_now()

    print(f"[{utc_now()[:19]}] 检查: 新对话 {new_turns} 轮 (阈值 {threshold})")

    if new_turns < threshold:
        status["last_seen_turns"] = current_total
        save_status(status)
        print(f"  未达阈值，跳过")
        return {"triggered": False, "new_turns": new_turns, "result": None}

    # 触发校准循环
    print(f"  达到阈值，触发校准循环...")
    result = run_cycle(batch_size=min(batch_size, new_turns))

    # 更新状态
    status["last_run"] = utc_now()
    status["last_seen_turns"] = current_total
    status["total_cycles"] += 1
    status["last_verdict"] = result.get("verdict")
    status["last_injected"] = result.get("injected", False)
    status["last_errors"] = result.get("errors", [])
    if result.get("self_improvement", {}).get("applied"):
        status["prompt_version"] = result["self_improvement"].get("new_version", 1)

    save_status(status)
    return {"triggered": True, "new_turns": new_turns, "result": result}


# ─── 命令行 ────────────────────────────────────────────────

def cmd_status():
    """查看当前状态"""
    status = load_status()
    print(f"\n{'='*50}")
    print(f"  Brain Watchdog 状态")
    print(f"{'='*50}")
    print(f"  最后检查: {status.get('last_check', '从未')}")
    print(f"  最后运行: {status.get('last_run', '从未')}")
    print(f"  总循环数: {status.get('total_cycles', 0)}")
    print(f"  最后已知轮次: {status.get('last_seen_turns', 0)}")
    print(f"  待处理轮次: {status.get('pending_turns', 0)}")
    print(f"  最后裁决: {status.get('last_verdict', 'N/A')}")
    print(f"  最后注入: {'是' if status.get('last_injected') else '否'}")
    print(f"  Prompt 版本: v{status.get('prompt_version', 1)}")
    if status.get("last_errors"):
        print(f"  最后错误: {len(status['last_errors'])} 个")
    print(f"{'='*50}")


def cmd_watch(interval: int = 60, threshold: int = 5):
    """持续监控模式"""
    print(f"Brain Watchdog 启动 (间隔 {interval} 分钟, 阈值 {threshold} 轮)")
    print(f"按 Ctrl+C 停止\n")

    # 优雅退出
    running = True
    def handle_signal(sig, frame):
        nonlocal running
        print("\n收到停止信号，正在退出...")
        running = False
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    while running:
        try:
            check_and_run(threshold=threshold)
        except Exception as e:
            print(f"  [ERROR] {e}")

        # 等待下一轮（可中断）
        for _ in range(interval * 60):
            if not running:
                break
            time.sleep(1)

    print("Watchdog 已停止")


def cmd_once():
    """检查一次状态，不触发循环"""
    status = load_status()
    new_turns, current_total = count_new_turns(status.get("last_seen_turns", 0))
    status["last_check"] = utc_now()
    status["pending_turns"] = new_turns
    save_status(status)

    print(f"新对话: {new_turns} 轮 (总 {current_total})")
    print(f"阈值: 5 轮")
    if new_turns >= 5:
        print(f"✅ 达到阈值，建议运行校准循环")
        print(f"   运行: python run_auto.py --batch {min(20, new_turns)}")
    else:
        print(f"⏳ 未达阈值 (还差 {5 - new_turns} 轮)")


if __name__ == "__main__":
    cfg = BrainConfig()
    STATUS_FILE = os.path.join(cfg.brain_dir, "data", "watchdog_status.json")

    if "--status" in sys.argv:
        cmd_status()
    elif "--once" in sys.argv:
        cmd_once()
    else:
        interval = 60  # 默认 60 分钟
        threshold = 5  # 默认 5 轮新对话
        for arg in sys.argv[1:]:
            if arg.startswith("--interval="):
                interval = int(arg.split("=")[1])
            elif arg.startswith("--threshold="):
                threshold = int(arg.split("=")[1])
        cmd_watch(interval=interval, threshold=threshold)
