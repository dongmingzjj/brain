"""
Brain Phase 0 主入口

用法:
  python run_phase0.py seed        # 导入种子校准失败数据
  python run_phase0.py scan        # 扫描对话历史（TODO: 接入 Hermes session DB）
  python run_phase0.py arbitrate   # 生成新校准建议
  python run_phase0.py verify      # 验证最新建议
  python run_phase0.py report      # 输出校准报告
  python run_phase0.py rebuild     # 从 WAL 重建 SQLite（崩溃恢复）
  python run_phase0.py run         # 一键跑完: arbitrate → verify → report
"""

import sys
import os
import json
from brain.config import BrainConfig
from brain.wal import WALWriter
from brain.db import BrainDB
from brain.capture import CalibrationCapture
from brain.arbitrator import Arbitrator
from brain.verifier import Verifier


from brain.hermes_db import HermesSessionReader
from brain.regions.memory.executor import MemoryExecutor
from brain.regions.memory.local_improver import LocalImprover
from brain.regions.memory.metrics import MemoryMetrics


def get_components():
    cfg = BrainConfig()
    cfg.ensure_dirs()
    wal = WALWriter(cfg.wal_dir, cfg.max_entries_per_shard)
    db = BrainDB(cfg.db_path)
    # Memory Region
    memory_db = os.path.join(cfg.brain_dir, "data", "memory.db")
    mem_executor = MemoryExecutor(memory_db)
    mem_improver = LocalImprover(mem_executor)
    mem_metrics = MemoryMetrics(mem_executor)
    return cfg, wal, db, mem_executor, mem_improver, mem_metrics


# ─── seed ──────────────────────────────────────────────────

def seed():
    """导入种子校准失败数据（从 Hermes 历史对话中手工挑选的案例）"""
    cfg, wal, db, mem_executor, mem_improver, mem_metrics = get_components()

    seed_failures = [
        # ─── 训练集（8 条，Arbitrator 可见）───
        {
            "session_id": "seed_001", "is_test_set": 0,
            "question_type": "code",
            "error_type": "hallucination",
            "question_summary": "用户问如何使用某个 Python 库的特定方法",
            "wrong_answer_summary": "AI 编造了一个不存在的 API 方法名和参数签名",
            "correction_summary": "应该先查看库的文档或源码确认 API 是否存在，不要凭印象编造",
        },
        {
            "session_id": "seed_002", "is_test_set": 0,
            "question_type": "factual",
            "error_type": "overconfidence",
            "question_summary": "用户问某个 GitHub 项目是否支持某功能",
            "wrong_answer_summary": "AI 直接断言支持该功能，没有查看项目仓库验证",
            "correction_summary": "应该搜索或查看项目 README/issues 确认功能支持情况",
        },
        {
            "session_id": "seed_003", "is_test_set": 0,
            "question_type": "recommendation",
            "error_type": "people_pleasing",
            "question_summary": "用户提出一个技术方案，AI 立即表示赞同",
            "wrong_answer_summary": "AI 迎合用户想法，没有指出方案中的明显缺陷",
            "correction_summary": "应该先分析方案的优缺点，指出潜在问题，而不是直接赞同",
        },
        {
            "session_id": "seed_004", "is_test_set": 0,
            "question_type": "reasoning",
            "error_type": "rigidity",
            "question_summary": "用户描述了一个与之前类似但实际不同的场景",
            "wrong_answer_summary": "AI 直接套用之前的解决方案，没有注意场景差异",
            "correction_summary": "应该先确认当前场景与之前的具体差异，再决定是否套用",
        },
        {
            "session_id": "seed_005", "is_test_set": 0,
            "question_type": "factual",
            "error_type": "hallucination",
            "question_summary": "用户问某个 CLI 工具的参数用法",
            "wrong_answer_summary": "AI 给出了错误的参数格式，编造了不存在的 flag",
            "correction_summary": "应该运行 --help 或查看 man page 确认参数列表",
        },
        {
            "session_id": "seed_006", "is_test_set": 0,
            "question_type": "code",
            "error_type": "overconfidence",
            "question_summary": "用户问某段代码的 bug 原因",
            "wrong_answer_summary": "AI 看了一眼就断言 bug 原因，没有实际运行验证",
            "correction_summary": "应该先运行代码或写测试复现 bug，再分析原因",
        },
        {
            "session_id": "seed_007", "is_test_set": 0,
            "question_type": "recommendation",
            "error_type": "overconfidence",
            "question_summary": "用户问该选哪个技术栈",
            "wrong_answer_summary": "AI 直接推荐了一个方案，没有了解用户的具体需求和约束",
            "correction_summary": "应该先问清楚需求、团队规模、技术约束，再给建议",
        },
        {
            "session_id": "seed_008", "is_test_set": 0,
            "question_type": "factual",
            "error_type": "hallucination",
            "question_summary": "用户问虾评平台的某个 API 返回格式",
            "wrong_answer_summary": "AI 编造了响应格式，字段名和结构都是假的",
            "correction_summary": "应该实际调用 API 或查看文档确认返回格式",
        },
        # ─── 留出集（4 条，Arbitrator 没见过，Verifier 用来测）───
        {
            "session_id": "seed_009", "is_test_set": 1,
            "question_type": "reasoning",
            "error_type": "rigidity",
            "question_summary": "用户在 Windows 上遇到了文件权限问题，怎么解决？",
            "wrong_answer_summary": "AI 给出了 Linux 的 chmod 命令",
            "correction_summary": "应该先确认操作系统，Windows 没有 chmod",
        },
        {
            "session_id": "seed_010", "is_test_set": 1,
            "question_type": "code",
            "error_type": "people_pleasing",
            "question_summary": "用户想用一个复杂的正则表达式解析 JSON，帮我写一个",
            "wrong_answer_summary": "AI 直接帮写了正则，没有建议用 json.loads",
            "correction_summary": "应该指出 json.loads 是更好的方案，不需要正则",
        },
        {
            "session_id": "seed_011", "is_test_set": 1,
            "question_type": "factual",
            "error_type": "hallucination",
            "question_summary": "asyncio.to_thread 和 run_in_executor 有什么区别？",
            "wrong_answer_summary": "AI 说 to_thread 是第三方库的函数",
            "correction_summary": "asyncio.to_thread 是 Python 3.9+ 标准库函数",
        },
        {
            "session_id": "seed_012", "is_test_set": 1,
            "question_type": "reasoning",
            "error_type": "overconfidence",
            "question_summary": "微服务架构是否适合一个 3 人团队的项目？",
            "wrong_answer_summary": "AI 直接说适合，没有指出小团队用微服务的风险",
            "correction_summary": "应该指出 3 人团队用微服务通常过度设计，建议先单体",
        },
    ]

    print(f"导入 {len(seed_failures)} 条种子校准失败...")
    from brain.wal import utc_now

    for i, f in enumerate(seed_failures):
        ts = utc_now()
        seq = wal.append(
            actor="capture",
            event_type="failure_recorded",
            data=f,
            evidence={"source": "seed"},
            verified=False,
            timestamp=ts,
        )
        # 先同步 events 表（满足外键约束）
        db.index_event({
            "seq": seq,
            "timestamp": ts,
            "actor": "capture",
            "event_type": "failure_recorded",
            "data": f,
            "evidence": {"source": "seed"},
            "verified": False,
        })
        db.add_calibration_failure(seq=seq, created_at=ts, **f)
        print(f"  [{i+1}/{len(seed_failures)}] seq={seq} [{f['error_type']}] {f['question_summary'][:40]}...")

    print(f"\n完成。总失败记录: {db.get_total_failures()}")


# ─── scan ──────────────────────────────────────────────────

def scan():
    """扫描 Hermes 真实对话历史，捕获校准失败"""
    cfg, wal, db, mem_executor, mem_improver, mem_metrics = get_components()
    capture = CalibrationCapture(wal, db, mem_executor=mem_executor)
    reader = HermesSessionReader()

    # 预估轮次数量
    sessions = reader.get_sessions_summary()
    total_sessions = len(sessions)
    print(f"Hermes session DB: {total_sessions} 个用户会话")

    # 提取对话轮次
    print("提取对话轮次...")
    turns = reader.get_turns(min_assistant_len=80, exclude_subagent=True)
    print(f"找到 {len(turns)} 个有效对话轮次\n")

    # 限制批次大小（避免 API 成本爆炸）
    batch_size = cfg.capture_batch_size
    if len(turns) > batch_size:
        print(f"只分析最近 {batch_size} 轮（超出部分跳过）")
        turns = turns[-batch_size:]

    # 逐轮分析
    failures_found = 0
    errors_skipped = 0

    for i, turn in enumerate(turns):
        print(f"[{i+1}/{len(turns)}] session={turn.session_title[:25]}...")
        print(f"  Q: {turn.user_msg[:60]}...")

        result = capture.analyze_turn(
            user_msg=turn.user_msg,
            assistant_msg=turn.assistant_msg,
            session_id=turn.session_id,
        )

        if result:
            failures_found += 1
            print(f"  ⚠️  [{result['error_type']}] {result['correction_summary'][:60]}...")
        else:
            print(f"  ✓  无校准失败")

    reader.close()

    print(f"\n扫描完成:")
    print(f"  分析轮次: {len(turns)}")
    print(f"  发现失败: {failures_found}")
    print(f"  总失败记录: {db.get_total_failures()}")


# ─── arbitrate ─────────────────────────────────────────────

def arbitrate():
    """生成新校准建议"""
    cfg, wal, db, mem_executor, mem_improver, mem_metrics = get_components()
    arb = Arbitrator(wal, db, mem_executor=mem_executor)

    print("Arbitrator 生成校准建议...")
    result = arb.propose_advisory()

    if result.get("status") == "skipped":
        print(f"跳过: {result.get('reason')}")
        return

    print(f"\n建议 v{result['version']} 已生成（seq={result['seq']}，基于 {result['failure_count']} 条失败）")
    print(f"\n{'='*60}")
    print(result["content"])
    print(f"{'='*60}")


# ─── verify ────────────────────────────────────────────────

def verify():
    """验证最新建议"""
    cfg, wal, db, mem_executor, mem_improver, mem_metrics = get_components()
    ver = Verifier(wal, db)

    print("Verifier 验证最新建议...")
    result = ver.verify_latest()

    if result.get("status"):
        print(f"跳过: {result.get('reason')}")
        return

    verdict_emoji = "✅" if result["verdict"] in ("accept", "accepted") else "❌"
    print(f"\n{verdict_emoji} 裁决: {result['verdict'].upper()}")
    print(f"   建议版本: v{result['version']}")
    print(f"   B 更好率: {result['improvement_rate']:.0%} ({result['b_better_count']}/{result['case_count']})")
    print(f"   平均分差: {result.get('avg_score_diff', 0):+.2f}")
    print(f"   验证方法: 确定性 A/B（模式匹配打分，不使用 LLM 判断）")


# ─── report ────────────────────────────────────────────────

def report():
    """输出校准报告"""
    cfg, wal, db, mem_executor, mem_improver, mem_metrics = get_components()

    total = db.get_total_failures()
    stats = db.get_failure_stats()
    current = db.get_current_advisory()
    adv_stats = db.get_advisory_count()
    wal_check = wal.rebuild_check()

    # Memory Region 指标
    mem_stats = mem_metrics.compute_all()

    print(f"\n{'='*60}")
    print(f"  Brain 校准报告")
    print(f"{'='*60}")

    print(f"\n📊 存储状态")
    print(f"   WAL: {wal_check['total_entries']} 条事件, {wal_check['shards']} 个分片, 完整性: {wal_check['integrity']}")
    print(f"   SQLite: {db_path_short(cfg.db_path)}")

    print(f"\n🧠 Memory Region")
    print(f"   记忆数: {mem_stats['total_memories']}")
    print(f"   平均重要性: {mem_stats['avg_importance']:.3f}")
    print(f"   总访问: {mem_stats['total_accesses']}")
    print(f"   高重要性占比: {mem_stats['high_importance_ratio']:.0%}")
    print(f"   7天未访问占比: {mem_stats['stale_memory_ratio']:.0%}")

    print(f"\n📈 校准失败统计")
    print(f"   总失败记录: {total}")
    if stats:
        print(f"   按错误类型:")
        for etype, info in stats.items():
            print(f"     {etype}: {info['count']} 次")

    print(f"\n📋 校准建议")
    print(f"   统计: {adv_stats}")
    if current:
        print(f"   当前生效: v{current['version']}")
        rate = current.get('post_score')
        print(f"   B 更好率: {rate:.0%}" if rate is not None else "   B 更好率: N/A")
        print(f"   内容:")
        print(f"   {'-'*50}")
        for line in current["content"].split("\n"):
            print(f"   {line}")
        print(f"   {'-'*50}")
    else:
        print(f"   （无生效建议）")

    print(f"\n{'='*60}")


def db_path_short(path):
    return "..." + path[-40:] if len(path) > 40 else path


# ─── rebuild ───────────────────────────────────────────────

def rebuild():
    """从 WAL 重建 SQLite（崩溃恢复）"""
    cfg, wal, db, mem_executor, mem_improver, mem_metrics = get_components()

    print("从 WAL 重建 SQLite...")
    entries = wal.read_all()
    stats = db.rebuild_from_wal(entries)

    print(f"重建完成:")
    print(f"  事件: {stats['events']}")
    print(f"  失败记录: {stats['failures']}")
    print(f"  建议: {stats['advisories']}")

    check = wal.rebuild_check()
    print(f"  WAL 完整性: {check['integrity']}")


# ─── improve ───────────────────────────────────────────────

def improve():
    """运行 Memory Region Local Improver 循环"""
    cfg, wal, db, mem_executor, mem_improver, mem_metrics = get_components()

    print("Memory Region Local Improver 循环...")
    print()

    # 运行前指标
    before = mem_metrics.compute_all()
    print(f"  运行前: {before['total_memories']} 条记忆, "
          f"平均重要性 {before['avg_importance']:.3f}, "
          f"总访问 {before['total_accesses']}")

    # 执行改进循环
    result = mem_improver.run_cycle()

    # 运行后指标
    after = mem_metrics.compute_all()
    print(f"  运行后: {after['total_memories']} 条记忆, "
          f"平均重要性 {after['avg_importance']:.3f}, "
          f"总访问 {after['total_accesses']}")

    # 执行的动作
    if result["actions_taken"]:
        print(f"\n  执行动作:")
        for action in result["actions_taken"]:
            print(f"    {action['action']}: {action.get('count', action.get('new_threshold', ''))}")
    else:
        print(f"\n  无动作执行")

    improvement = result["improvement"]
    emoji = "📈" if improvement > 0 else "📉" if improvement < 0 else "➖"
    print(f"\n  {emoji} 改善度: {improvement:+.3f}")


# ─── run (一键跑完) ────────────────────────────────────────

def run():
    """一键跑完: scan → arbitrate → verify → improve → report"""
    print("=" * 60)
    print("  Brain 完整循环")
    print("=" * 60)

    print("\n[1/5] 扫描对话...")
    scan()

    print("\n[2/5] 生成校准建议...")
    arbitrate()

    print("\n[3/5] 验证建议...")
    verify()

    print("\n[4/5] Memory Region 改进循环...")
    improve()

    print("\n[5/5] 校准报告...")
    report()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"

    commands = {
        "seed": seed,
        "scan": scan,
        "arbitrate": arbitrate,
        "verify": verify,
        "improve": improve,
        "report": report,
        "rebuild": rebuild,
        "run": run,
    }

    if cmd not in commands:
        print(f"未知命令: {cmd}")
        print(f"可用命令: {', '.join(commands.keys())}")
        sys.exit(1)

    commands[cmd]()
