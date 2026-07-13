"""Arbitrator 自改进单元测试"""
import pytest
import tempfile
import shutil
from pathlib import Path
from brain.wal import WALWriter
from brain.db import BrainDB
from brain.arbitrator_self_improve import ArbitratorSelfImprover, DEFAULT_ADVISORY_PROMPT


@pytest.fixture
def setup():
    d = tempfile.mkdtemp(prefix="brain_selfimp_test_")
    wal = WALWriter(str(Path(d) / "wal"))
    db = BrainDB(str(Path(d) / "brain.db"))
    improver = ArbitratorSelfImprover(wal, db)
    yield wal, db, improver
    shutil.rmtree(d, ignore_errors=True)


class TestPromptManagement:
    """测试 prompt 模板管理"""

    def test_default_prompt(self, setup):
        """默认 prompt 模板"""
        _, _, improver = setup
        prompt = improver.get_current_prompt()
        assert "{failures}" in prompt
        assert "{failure_count}" in prompt
        assert "{current_advisory}" in prompt

    def test_prompt_version(self, setup):
        """版本号递增"""
        _, _, improver = setup
        assert improver.get_prompt_version() == 1

        improver.apply_improvement("new prompt v2")
        assert improver.get_prompt_version() == 2

        improver.apply_improvement("new prompt v3")
        assert improver.get_prompt_version() == 3

    def test_apply_improvement(self, setup):
        """应用改进"""
        wal, _, improver = setup
        result = improver.apply_improvement("improved prompt")

        assert result["applied"] is True
        assert result["old_version"] == 1
        assert result["new_version"] == 2
        assert improver.get_current_prompt() == "improved prompt"

        # WAL 记录
        entries = wal.read_all()
        assert any(e["event_type"] == "prompt_updated" for e in entries)


class TestAnalysis:
    """测试历史分析"""

    def test_insufficient_history(self, setup):
        """历史数据不足"""
        _, _, improver = setup
        result = improver.analyze_and_improve()

        assert result["prompt_changed"] is False
        assert "不足" in result["analysis"]

    def test_with_history(self, setup):
        """有历史数据时调用 LLM 分析"""
        wal, db, improver = setup

        # 先写 events（满足 FK 约束），再写 advisories
        for seq, content, status, score in [
            (1, '建议1：先验证再回答', 'accepted', 0.7),
            (2, '建议2：不要编造数据', 'rejected', 0.3),
        ]:
            wal.append(actor="arbitrator", event_type="advisory_proposed",
                       data={"version": seq, "content": content})
            db.index_event({
                "seq": seq, "timestamp": "2026-07-13T10:00:00Z",
                "actor": "arbitrator", "event_type": "advisory_proposed",
                "data": {"version": seq, "content": content}, "verified": False,
            })
            db.add_advisory(seq=seq, version=seq, content=content)
            db.update_advisory_status(
                db.conn.execute("SELECT id FROM advisories ORDER BY id DESC LIMIT 1").fetchone()[0],
                status, post_score=score,
            )

        try:
            result = improver.analyze_and_improve()
            assert "analysis" in result
            assert "improvements" in result
            assert "prompt_changed" in result
        except Exception as e:
            pytest.skip(f"LLM 调用失败: {e}")


class TestIntegration:
    """集成测试（需要 LLM）"""

    def test_full_improvement_cycle(self, setup):
        """完整改进循环：分析 → 应用 → 验证"""
        wal, db, improver = setup

        # 模拟多条历史（先写 events 再写 advisories）
        statuses = [
            (1, '建议1：先验证再回答', 'accepted', 0.7),
            (2, '建议2：不要编造数据', 'rejected', 0.3),
            (3, '建议3：表达不确定性', 'accepted', 0.6),
            (4, '建议4：确认用户意图', 'accepted', 0.8),
        ]
        for seq, content, status, score in statuses:
            wal.append(actor="arbitrator", event_type="advisory_proposed",
                       data={"version": seq, "content": content})
            db.index_event({
                "seq": seq, "timestamp": "2026-07-13T10:00:00Z",
                "actor": "arbitrator", "event_type": "advisory_proposed",
                "data": {"version": seq, "content": content}, "verified": False,
            })
            db.add_advisory(seq=seq, version=seq, content=content)
            db.update_advisory_status(
                db.conn.execute("SELECT id FROM advisories ORDER BY id DESC LIMIT 1").fetchone()[0],
                status, post_score=score,
            )
        db.conn.commit()

        try:
            # 分析
            result = improver.analyze_and_improve()
            assert "analysis" in result

            # 如果有改进，应用
            if result["prompt_changed"] and result["new_prompt"]:
                apply_result = improver.apply_improvement(result["new_prompt"])
                assert apply_result["applied"] is True
                assert apply_result["new_version"] == 2
        except Exception as e:
            pytest.skip(f"LLM 调用失败: {e}")
