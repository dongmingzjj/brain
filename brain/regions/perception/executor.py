"""
Perception Region Executor — 输入理解 + 特征提取。

职责：
  - 接收原始用户输入
  - 提取意图、实体、情感、复杂度
  - 输出结构化感知结果供下游 Region 使用

Phase 4.0 实现（非 LLM，确定性）：
  - 关键词匹配做意图分类
  - 正则提取实体
  - 规则判断复杂度
  - 标点/词汇分析做情感判断

Phase 4.1 升级路径：
  - 接入 Ollama embedding 做语义理解
  - LLM 做深度意图分析
"""

from __future__ import annotations
import re
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─── 意图分类规则 ──────────────────────────────────────────

INTENT_RULES = {
    "question": {
        "keywords": ["什么", "怎么", "如何", "为什么", "哪个", "是不是", "有没有", "多少", "？", "?", "what", "how", "why", "which"],
        "weight": 1.0,
    },
    "command": {
        "keywords": ["帮我", "给我", "执行", "运行", "创建", "删除", "修改", "安装", "配置", "部署", "do", "run", "create", "delete"],
        "weight": 1.0,
    },
    "search": {
        "keywords": ["搜索", "查找", "找一下", "看一下", "查看", "搜", "search", "find", "look"],
        "weight": 0.8,
    },
    "recommendation": {
        "keywords": ["推荐", "建议", "选哪个", "哪个好", "值得", "recommend", "suggest"],
        "weight": 0.9,
    },
    "confirmation": {
        "keywords": ["好的", "嗯", "继续", "可以", "对", "是的", "ok", "yes", "继续吧", "先这样"],
        "weight": 0.5,
    },
    "correction": {
        "keywords": ["不对", "错了", "不是", "重新", "修改", "wrong", "no", "not"],
        "weight": 0.7,
    },
}

# ─── 实体提取规则 ──────────────────────────────────────────

ENTITY_PATTERNS = {
    "url": r'https?://[^\s<>"{}|\\^`\[\]]+',
    "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    "file_path": r'(?:[A-Za-z]:[\\/]|\/|\./)[^\s<>"*|?]+',
    "command": r'`[^`]+`',
    "code_block": r'```[\s\S]*?```',
    "chinese_name": r'[\u4e00-\u9fff]{2,4}',
    "english_word": r'[a-zA-Z_][a-zA-Z0-9_]{2,}',
    "number": r'\b\d+(?:\.\d+)?\b',
}

# ─── 复杂度指标 ────────────────────────────────────────────

def estimate_complexity(text: str) -> str:
    """估算输入复杂度"""
    length = len(text)
    has_code = bool(re.search(r'```|`[^`]+`', text))
    has_url = bool(re.search(r'https?://', text))
    has_multi_question = text.count("？") + text.count("?") > 1
    word_count = len(re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+', text))

    score = 0
    if length > 200: score += 2
    elif length > 50: score += 1
    if has_code: score += 2
    if has_url: score += 1
    if has_multi_question: score += 1
    if word_count > 30: score += 1

    if score >= 4: return "high"
    if score >= 2: return "medium"
    return "low"


class PerceptionExecutor:
    """Perception Region 的执行器"""

    def __init__(self, db_path: str = None):
        if db_path:
            self.conn = sqlite3.connect(db_path)
            self.conn.row_factory = sqlite3.Row
            self._init_db()
        else:
            self.conn = None

        self._process_count = 0

    def _init_db(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS perceptions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                input_hash  TEXT,
                intent      TEXT,
                complexity  TEXT,
                entities    TEXT,
                sentiment   TEXT,
                confidence  REAL,
                timestamp   TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_perceptions_intent
                ON perceptions(intent);
        """)
        self.conn.commit()

    # ─── 核心执行 ──────────────────────────────────────────

    def execute(self, input_text: str) -> dict:
        """
        分析输入文本，返回结构化感知结果。

        返回:
            {
                intent: str          — 主要意图
                intent_scores: dict  — 各意图的得分
                entities: dict       — 提取的实体
                complexity: str      — 复杂度 (low/medium/high)
                sentiment: str       — 情感 (neutral/positive/negative/urgent)
                confidence: float    — 置信度
                summary: str         — 一句话摘要
            }
        """
        self._process_count += 1

        # 1. 意图分类
        intent_scores = self._classify_intent(input_text)
        primary_intent = max(intent_scores, key=intent_scores.get) if intent_scores else "unknown"
        primary_score = intent_scores.get(primary_intent, 0)

        # 2. 实体提取
        entities = self._extract_entities(input_text)

        # 3. 复杂度
        complexity = estimate_complexity(input_text)

        # 4. 情感
        sentiment = self._analyze_sentiment(input_text)

        # 5. 置信度（基于意图得分差距 + 实体数量）
        confidence = self._compute_confidence(intent_scores, entities, len(input_text))

        # 6. 摘要
        summary = self._make_summary(input_text, primary_intent, entities)

        result = {
            "intent": primary_intent,
            "intent_scores": intent_scores,
            "entities": entities,
            "complexity": complexity,
            "sentiment": sentiment,
            "confidence": confidence,
            "summary": summary,
            "input_length": len(input_text),
        }

        # 记录到 DB
        if self.conn:
            self._log_perception(input_text, result)

        return result

    # ─── 意图分类 ──────────────────────────────────────────

    def _classify_intent(self, text: str) -> dict[str, float]:
        """基于关键词的意图分类"""
        text_lower = text.lower()
        scores = {}

        for intent, rule in INTENT_RULES.items():
            score = 0
            for kw in rule["keywords"]:
                if kw.lower() in text_lower:
                    score += rule["weight"]
            if score > 0:
                # 归一化：log(1 + score) 压缩范围
                scores[intent] = round(score / (1 + score * 0.3), 3)

        return scores

    # ─── 实体提取 ──────────────────────────────────────────

    def _extract_entities(self, text: str) -> dict[str, list]:
        """提取文本中的实体"""
        entities = {}

        for entity_type, pattern in ENTITY_PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                # 去重 + 截断
                unique = list(dict.fromkeys(matches))[:10]
                entities[entity_type] = unique

        return entities

    # ─── 情感分析 ──────────────────────────────────────────

    def _analyze_sentiment(self, text: str) -> str:
        """简单的情感分析"""
        text_lower = text.lower()

        urgent_markers = ["急", "快", "马上", "紧急", "尽快", "urgent", "asap", "崩溃", "挂了", "不行"]
        negative_markers = ["不对", "错了", "问题", "失败", "bug", "崩溃", "不行", "坏", "wrong", "error", "fail"]
        positive_markers = ["好的", "不错", "感谢", "谢谢", "很好", "完美", "great", "good", "thanks"]

        if any(m in text_lower for m in urgent_markers):
            return "urgent"
        if any(m in text_lower for m in negative_markers):
            return "negative"
        if any(m in text_lower for m in positive_markers):
            return "positive"
        return "neutral"

    # ─── 置信度计算 ────────────────────────────────────────

    def _compute_confidence(self, intent_scores: dict, entities: dict, length: int) -> float:
        """计算感知置信度"""
        if not intent_scores:
            return 0.2  # 无匹配意图 = 低置信度

        # 意图清晰度：最高分 vs 第二高分的差距
        sorted_scores = sorted(intent_scores.values(), reverse=True)
        if len(sorted_scores) >= 2:
            clarity = sorted_scores[0] - sorted_scores[1]
        else:
            clarity = sorted_scores[0]

        # 实体丰富度
        entity_count = sum(len(v) for v in entities.values())
        entity_bonus = min(entity_count / 10, 0.3)

        # 输入长度（太短 = 模糊）
        length_bonus = min(length / 200, 0.2)

        confidence = clarity * 0.5 + entity_bonus + length_bonus
        return round(min(confidence, 0.99), 3)

    # ─── 摘要生成 ──────────────────────────────────────────

    def _make_summary(self, text: str, intent: str, entities: dict) -> str:
        """生成一句话摘要"""
        # 取前 50 字
        snippet = text[:50].replace("\n", " ")
        if len(text) > 50:
            snippet += "..."

        entity_summary = []
        for etype, values in entities.items():
            if values:
                entity_summary.append(f"{etype}: {values[0]}")

        entity_str = f" [{', '.join(entity_summary[:3])}]" if entity_summary else ""
        return f"[{intent}] {snippet}{entity_str}"

    # ─── 日志 ──────────────────────────────────────────────

    def _log_perception(self, input_text: str, result: dict):
        import hashlib
        input_hash = hashlib.md5(input_text.encode()).hexdigest()[:12]

        self.conn.execute(
            """INSERT INTO perceptions
               (input_hash, intent, complexity, entities, sentiment, confidence, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                input_hash,
                result["intent"],
                result["complexity"],
                json.dumps(result["entities"], ensure_ascii=False),
                result["sentiment"],
                result["confidence"],
                utc_now(),
            )
        )
        self.conn.commit()

    # ─── 统计 ──────────────────────────────────────────────

    def get_stats(self) -> dict:
        if not self.conn:
            return {"total_perceptions": self._process_count}

        total = self.conn.execute("SELECT COUNT(*) FROM perceptions").fetchone()[0]
        by_intent = self.conn.execute(
            "SELECT intent, COUNT(*) as n FROM perceptions GROUP BY intent ORDER BY n DESC"
        ).fetchall()

        return {
            "total_perceptions": total,
            "by_intent": {r["intent"]: r["n"] for r in by_intent},
        }

    def close(self):
        if self.conn:
            self.conn.close()
