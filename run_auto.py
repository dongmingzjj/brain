#!/usr/bin/env python
"""
Brain 自动化校准循环 — 供 cron 调用。

完整流程:
  1. scan:      扫描最近的 Hermes 对话，捕获校准失败
  2. arbitrate: 生成新的校准建议
  3. verify:    A/B 行为测试验证建议
  4. inject:    如果建议通过，自动注入 SOUL.md
  5. report:    输出摘要

用法:
  python run_auto.py                    # 完整循环
  python run_auto.py --scan-only        # 只扫描不验证
  python run_auto.py --inject-only      # 只注入最新已通过的建议
  python run_auto.py --batch 10         # 只扫描最近10轮对话（省 API）

退出码:
  0 = 成功（有或无新失败）
  1 = Verifier 拒绝了建议
  2 = 运行错误
"""

import sys
import os
import re
from pathlib import Path
from datetime import datetime, timezone

# 确保能 import brain
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brain.config import BrainConfig
from brain.wal import WALWriter, utc_now
from brain.db import BrainDB
from brain.capture import CalibrationCapture
from brain.arbitrator import Arbitrator
from brain.verifier import Verifier
from brain.hermes_db import HermesSessionReader
from brain.regions.memory.executor import MemoryExecutor
from brain.regions.memory.local_improver import LocalImprover
from brain.regions.memory.metrics import MemoryMetrics
from brain.arbitrator_self_improve import ArbitratorSelfImprover


# ─── SOUL.md 注入 ──────────────────────────────────────────

SOUL_MARKER_START = "## 认知校准（自动生成"
SOUL_MARKER_END = "不猜。"


def inject_soul(advisory_content: str) -> bool:
    """
    将校准建议注入 Hermes SOUL.md。

    如果 SOUL.md 已有校准 section，替换内容。
    如果没有，在末尾追加。

    返回:
        True = 注入成功
        False = 注入失败
    """
    soul_path = os.path.join(
        os.environ.get("LOCALAPPDATA", r"C:\Users\Administrator\AppData\Local"),
        "hermes", "SOUL.md"
    )

    if not os.path.exists(soul_path):
        print(f"  [ERROR] SOUL.md 不存在: {soul_path}")
        return False

    with open(soul_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 构建新的校准 section
    calibration_section = f"""## 认知校准（自动生成，Brain v1）

> 以下规则由 Brain 系统从真实对话中识别校准失败后自动生成，经 A/B 对照测试验证有效。

{advisory_content}"""

    # 检查是否已有校准 section
    if SOUL_MARKER_START in content:
        # 替换已有 section
        # 找到起始和结束位置
        start_idx = content.index(SOUL_MARKER_START)

        # 找结束：以"不猜。"结尾的段落，或者下一个 ## 标题
        end_idx = content.find("\n## ", start_idx + 1)
        if end_idx == -1:
            # 到文件末尾
            old_section = content[start_idx:]
            content = content[:start_idx].rstrip("\n") + "\n\n" + calibration_section + "\n"
        else:
            old_section = content[start_idx:end_idx]
            content = content[:start_idx] + calibration_section + "\n" + content[end_idx:]
    else:
        # 追加到末尾
        content = content.rstrip("\n") + "\n\n" + calibration_section + "\n"

    # 写回
    with open(soul_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"  ✅ SOUL.md 已更新")
    return True


# ─── 主流程 ────────────────────────────────────────────────

def run_cycle(batch_size: int = 20, skip_verify: bool = False):
    """
    运行完整校准循环。

    流程：
      1. scan:      扫描对话
      2. arbitrate: 生成校准建议
      3. verify:    A/B 行为测试
      4. inject:    注入 SOUL.md
      5. improve:   Memory Region 改进
      6. self_improve: Arbitrator prompt 自改进

    返回:
        dict: 运行结果摘要
    """
    cfg = BrainConfig()
    cfg.ensure_dirs()
    wal = WALWriter(cfg.wal_dir, cfg.max_entries_per_shard)
    db = BrainDB(cfg.db_path)
    memory_db = os.path.join(cfg.brain_dir, "data", "memory.db")
    mem_executor = MemoryExecutor(memory_db)
    mem_improver = LocalImprover(mem_executor)
    self_improver = ArbitratorSelfImprover(wal, db)

    result = {
        "timestamp": utc_now(),
        "scanned": 0,
        "failures_found": 0,
        "advisory_generated": False,
        "verdict": None,
        "injected": False,
        "self_improvement": None,
        "errors": [],
    }

    # ─── 1. SCAN ───────────────────────────────────────
    print("[1/6] 扫描对话...")
    try:
        capture = CalibrationCapture(wal, db, mem_executor=mem_executor)
        reader = HermesSessionReader()
        turns = reader.get_turns(min_assistant_len=80, exclude_subagent=True)

        # 只取最近的 batch_size 轮
        if len(turns) > batch_size:
            turns = turns[-batch_size:]

        result["scanned"] = len(turns)

        failures_before = db.get_total_failures()

        for i, turn in enumerate(turns):
            r = capture.analyze_turn(
                user_msg=turn.user_msg,
                assistant_msg=turn.assistant_msg,
                session_id=turn.session_id,
            )
            if r:
                result["failures_found"] += 1

        reader.close()
        print(f"  扫描 {result['scanned']} 轮, 发现 {result['failures_found']} 条新失败")

    except Exception as e:
        result["errors"].append(f"scan: {e}")
        print(f"  [ERROR] scan: {e}")

    # 如果没有新失败，跳过后续步骤
    if result["failures_found"] == 0:
        print("  无新失败，跳过 arbitrate/verify")
        # 但仍然跑 improve
        print("\n[5/5] Memory Region 改进循环...")
        try:
            imp = mem_improver.run_cycle()
            print(f"  改善度: {imp['improvement']:+.3f}")
        except Exception as e:
            result["errors"].append(f"improve: {e}")
        return result

    # ─── 2. ARBITRATE ─────────────────────────────────
    print("\n[2/6] 生成校准建议...")
    try:
        # 使用自改进后的 prompt 模板（如有）
        current_prompt = self_improver.get_current_prompt()
        arb = Arbitrator(wal, db, mem_executor=mem_executor, prompt_template=current_prompt)
        adv_result = arb.propose_advisory()

        if adv_result.get("status") == "skipped":
            print(f"  跳过: {adv_result.get('reason')}")
        else:
            result["advisory_generated"] = True
            result["advisory_version"] = adv_result.get("version")
            result["advisory_content"] = adv_result.get("content", "")
            print(f"  建议 v{result['advisory_version']} 已生成")

    except Exception as e:
        result["errors"].append(f"arbitrate: {e}")
        print(f"  [ERROR] arbitrate: {e}")

    if not result["advisory_generated"] or skip_verify:
        if skip_verify:
            print("\n  --skip_verify: 跳过验证和注入")
        return result

    # ─── 3. VERIFY ────────────────────────────────────
    print("\n[3/6] A/B 行为测试...")
    try:
        ver = Verifier(wal, db)
        verify_result = ver.verify_latest()

        if verify_result.get("status"):
            print(f"  跳过: {verify_result.get('reason')}")
        else:
            result["verdict"] = verify_result.get("verdict")
            result["improvement_rate"] = verify_result.get("improvement_rate")
            print(f"  裁决: {result['verdict'].upper()}")
            print(f"  B 更好率: {result.get('improvement_rate', 0):.0%}")

    except Exception as e:
        result["errors"].append(f"verify: {e}")
        print(f"  [ERROR] verify: {e}")

    # ─── 4. INJECT ────────────────────────────────────
    if result["verdict"] == "accepted":
        print("\n[4/6] 注入 SOUL.md...")
        try:
            result["injected"] = inject_soul(result.get("advisory_content", ""))
        except Exception as e:
            result["errors"].append(f"inject: {e}")
            print(f"  [ERROR] inject: {e}")
    else:
        print(f"\n[4/6] 建议未通过（{result.get('verdict', 'N/A')}），不注入")

    # ─── 5. IMPROVE ───────────────────────────────────
    print("\n[5/6] Memory Region 改进循环...")
    try:
        imp = mem_improver.run_cycle()
        print(f"  改善度: {imp['improvement']:+.3f}")
    except Exception as e:
        result["errors"].append(f"improve: {e}")

    # ─── 6. SELF-IMPROVE ─────────────────────────────
    print("\n[6/6] Arbitrator 自改进分析...")
    try:
        self_imp = self_improver.analyze_and_improve()
        result["self_improvement"] = {
            "analysis": self_imp.get("analysis", ""),
            "improvements": self_imp.get("improvements", []),
            "prompt_changed": self_imp.get("prompt_changed", False),
        }

        if self_imp.get("prompt_changed") and self_imp.get("new_prompt"):
            # 应用新 prompt
            apply_result = self_improver.apply_improvement(self_imp["new_prompt"])
            result["self_improvement"]["applied"] = True
            result["self_improvement"]["new_version"] = apply_result["new_version"]
            print(f"  Prompt 模板已更新 → v{apply_result['new_version']}")
            print(f"  分析: {self_imp.get('analysis', '')[:80]}")
        else:
            print(f"  分析: {self_imp.get('analysis', '')[:80]}")
            print(f"  无需更新 prompt 模板")

    except Exception as e:
        result["errors"].append(f"self_improve: {e}")
        print(f"  [ERROR] self_improve: {e}")

    return result


# ─── CLI ───────────────────────────────────────────────────

if __name__ == "__main__":
    batch = 20
    scan_only = False
    inject_only = False

    args = sys.argv[1:]
    for arg in args:
        if arg == "--scan-only":
            scan_only = True
        elif arg == "--inject-only":
            inject_only = True
        elif arg.startswith("--batch="):
            batch = int(arg.split("=")[1])

    if inject_only:
        # 只注入最新已通过的校准建议
        cfg = BrainConfig()
        db = BrainDB(cfg.db_path)
        current = db.get_current_advisory()
        if current:
            print(f"注入 v{current['version']}...")
            inject_soul(current["content"])
        else:
            print("无已通过的校准建议")
        db.close()
    else:
        result = run_cycle(batch_size=batch, skip_verify=scan_only)

        # 摘要
        print(f"\n{'='*50}")
        print(f"Brain 自动校准摘要")
        print(f"{'='*50}")
        print(f"  扫描轮次: {result['scanned']}")
        print(f"  新失败: {result['failures_found']}")
        print(f"  建议生成: {'是' if result['advisory_generated'] else '否'}")
        print(f"  Verifier 裁决: {result.get('verdict', 'N/A')}")
        print(f"  注入 SOUL.md: {'是' if result['injected'] else '否'}")
        if result.get("self_improvement"):
            si = result["self_improvement"]
            print(f"  Prompt 自改进: {'已更新 v' + str(si.get('new_version', '')) if si.get('applied') else '未更新'}")
        if result["errors"]:
            print(f"  错误: {len(result['errors'])} 个")
            for e in result["errors"]:
                print(f"    - {e}")

        # 退出码
        if result.get("verdict") == "rejected":
            sys.exit(1)
        elif result["errors"]:
            sys.exit(2)
        else:
            sys.exit(0)
