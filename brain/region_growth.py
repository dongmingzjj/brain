"""
Region 生长协议 — 自动创建新 Region。

机制：
  1. Arbitrator 检测能力缺口（反复出现某类任务但没有专门的 Region 处理）
  2. 提议新 Region（Capability Card + 预期 I/O + 初始 Local Improver）
  3. Verifier 验证：
     a. 不与现有 Region 能力重叠（< 0.3 相似度）
     b. 模拟加入后基准不退化
  4. 通过 → 从历史行为提取种子数据 → 新 Region 上线

Phase 3 实现：检测 + 提议 + 模板生成
完整自动创建（代码生成 + 动态加载）留到后续
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from typing import Optional
from brain.event_bus import EventBus, Signal
from brain.llm import llm_call


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─── 已有 Region 的能力清单 ────────────────────────────────

KNOWN_REGIONS = {
    "memory": {
        "description": "存储、检索和管理长期记忆",
        "capabilities": ["store", "retrieve", "forget", "decay", "boost"],
        "keywords": ["记忆", "存储", "检索", "历史", "经验"],
    },
    "action": {
        "description": "工具调用、代码执行、结果收集",
        "capabilities": ["execute", "call_tool", "run_code"],
        "keywords": ["执行", "运行", "工具", "命令", "代码"],
    },
}


# ─── 能力缺口检测 ──────────────────────────────────────────

GAP_DETECTION_PROMPT = """分析以下校准失败记录，判断是否存在反复出现的能力缺口。

能力缺口 = 某类任务反复失败，但现有 Region 都无法处理。

现有 Region 能力：
{existing_regions}

最近的校准失败：
{failures}

请判断：
1. 是否存在能力缺口？
2. 如果有，缺口是什么？
3. 需要什么新能力来填补？

输出 JSON：
```json
{{
  "has_gap": true/false,
  "gap_description": "缺口描述",
  "gap_keywords": ["关键词1", "关键词2"],
  "suggested_region": {{
    "name": "新 Region 名称",
    "description": "一句话描述",
    "capabilities": ["能力1", "能力2"],
    "subscriptions": ["订阅的信号类型"],
    "outputs": ["输出的信号类型"]
  }},
  "reasoning": "判断理由"
}}
```"""


class RegionGrowthProtocol:
    """Region 生长协议"""

    def __init__(self, bus: EventBus, db):
        self.bus = bus
        self.db = db
        self.proposals = []

    def detect_gap(self) -> dict:
        """
        检测能力缺口。

        返回:
            {
                "has_gap": bool,
                "gap_description": str,
                "suggested_region": dict | None,
            }
        """
        # 收集最近的失败
        failures = self.db.get_recent_failures(limit=20)

        if len(failures) < 5:
            return {"has_gap": False, "reason": "失败数据不足（需要 >= 5 条）"}

        # 格式化已有 Region 能力
        regions_text = "\n".join([
            f"- {name}: {info['description']} ({', '.join(info['capabilities'])})"
            for name, info in KNOWN_REGIONS.items()
        ])

        # 格式化失败记录
        failures_text = "\n".join([
            f"- [{f.get('error_type', '?')}] {f.get('question_summary', '?')} → "
            f"应该: {f.get('correction_summary', '?')}"
            for f in failures
        ])

        prompt = GAP_DETECTION_PROMPT.format(
            existing_regions=regions_text,
            failures=failures_text,
        )

        raw = llm_call(prompt, temperature=0.3)

        # 解析
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            if "```json" in raw:
                start = raw.index("```json") + 7
                end = raw.index("```", start)
                result = json.loads(raw[start:end].strip())
            elif "{" in raw:
                start = raw.index("{")
                end = raw.rindex("}") + 1
                result = json.loads(raw[start:end])
            else:
                return {"has_gap": False, "reason": "LLM 响应解析失败"}

        return result

    def propose_region(self, gap: dict) -> dict:
        """
        基于能力缺口提议新 Region。

        返回:
            {
                "proposed": bool,
                "region_name": str,
                "capability_card": dict,
                "template_code": str,
            }
        """
        if not gap.get("has_gap") or not gap.get("suggested_region"):
            return {"proposed": False, "reason": "无能力缺口"}

        suggested = gap["suggested_region"]
        region_name = suggested.get("name", "new_region")

        # 检查是否与现有 Region 重叠
        overlap = self._check_overlap(region_name, suggested.get("capabilities", []))
        if overlap["overlaps"]:
            return {
                "proposed": False,
                "reason": f"与现有 Region 重叠: {overlap['overlap_with']} (相似度 {overlap['similarity']:.2f})",
            }

        # 生成 Capability Card
        capability_card = {
            "region": region_name,
            "version": "0.1.0",
            "description": suggested.get("description", ""),
            "capabilities": suggested.get("capabilities", []),
            "subscriptions": [
                {"signal_type": st, "priority": "normal"}
                for st in suggested.get("subscriptions", [])
            ],
            "outputs": suggested.get("outputs", []),
            "local_improver": {
                "algorithm": "threshold_adaptation",
                "llm_free": True,
            },
        }

        # 生成模板代码
        template_code = self._generate_template(region_name, capability_card)

        # 记录提议
        proposal = {
            "timestamp": utc_now(),
            "gap_description": gap.get("gap_description", ""),
            "region_name": region_name,
            "capability_card": capability_card,
        }
        self.proposals.append(proposal)

        return {
            "proposed": True,
            "region_name": region_name,
            "capability_card": capability_card,
            "template_code": template_code,
            "gap_description": gap.get("gap_description", ""),
        }

    def get_proposals(self) -> list[dict]:
        """获取所有提议历史"""
        return self.proposals

    # ─── 内部方法 ──────────────────────────────────────────

    def _check_overlap(self, name: str, capabilities: list[str]) -> dict:
        """检查新 Region 是否与现有 Region 重叠"""
        for existing_name, existing_info in KNOWN_REGIONS.items():
            # 能力重叠检测
            existing_caps = set(existing_info["capabilities"])
            new_caps = set(capabilities)
            if existing_caps and new_caps:
                overlap_count = len(existing_caps & new_caps)
                total = len(existing_caps | new_caps)
                similarity = overlap_count / total if total > 0 else 0

                if similarity > 0.3:
                    return {
                        "overlaps": True,
                        "overlap_with": existing_name,
                        "similarity": similarity,
                    }

        return {"overlaps": False}

    def _generate_template(self, name: str, card: dict) -> str:
        """生成 Region 模板代码"""
        return f'''"""
{name} Region — 自动生成的模板

Capability Card:
{json.dumps(card, ensure_ascii=False, indent=2)}

TODO: 实现 executor, local_improver, metrics
"""

from brain.event_bus import EventBus, Signal


class {name.title()}Executor:
    """{card.get("description", "")}"""

    def __init__(self):
        pass

    def execute(self, **kwargs):
        """执行核心功能"""
        # TODO: 实现
        pass

    def get_stats(self) -> dict:
        """返回运行指标"""
        return {{}}


class LocalImprover:
    """{name.title()} Region 的局部改进器"""

    def __init__(self, executor):
        self.executor = executor

    def run_cycle(self) -> dict:
        """执行一轮改进循环"""
        return {{"actions_taken": [], "improvement": 0}}


class {name.title()}Metrics:
    """{name.title()} Region 的局部指标"""

    def __init__(self, executor):
        self.executor = executor

    def compute_all(self) -> dict:
        return {{}}
'''
