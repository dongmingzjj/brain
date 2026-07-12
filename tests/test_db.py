"""SQLite 索引层单元测试"""
import pytest
import tempfile
import shutil
from pathlib import Path
from brain.db import BrainDB
from brain.wal import WALWriter


@pytest.fixture
def db_path():
    d = tempfile.mkdtemp(prefix="brain_db_test_")
    path = str(Path(d) / "test.db")
    yield path
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def wal_dir():
    d = tempfile.mkdtemp(prefix="brain_wal_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def make_failure_data(**overrides):
    """生成校准失败测试数据"""
    defaults = {
        "session_id": "test_session",
        "question_type": "factual",
        "error_type": "hallucination",
        "question_summary": "用户问了个事实问题",
        "wrong_answer_summary": "AI 编造了答案",
        "correction_summary": "应该先查证再回答",
        "should_have_verified": 1,
    }
    defaults.update(overrides)
    return defaults


class TestEvents:
    """测试 events 表"""

    def test_index_and_get(self, db_path):
        db = BrainDB(db_path)
        db.index_event({
            "seq": 1,
            "timestamp": "2026-07-12T10:00:00Z",
            "actor": "capture",
            "event_type": "failure_recorded",
            "data": {"error_type": "hallucination"},
            "evidence": {"confidence": 0.8},
            "verified": False,
        })
        event = db.get_event(1)
        assert event is not None
        assert event["actor"] == "capture"
        assert event["data"]["error_type"] == "hallucination"
        assert event["evidence"]["confidence"] == 0.8

    def test_get_nonexistent(self, db_path):
        db = BrainDB(db_path)
        assert db.get_event(999) is None


class TestCalibrationFailures:
    """测试 calibration_failures 表"""

    def test_add_and_retrieve(self, db_path):
        db = BrainDB(db_path)
        db.index_event({
            "seq": 1, "timestamp": "2026-07-12T10:00:00Z",
            "actor": "capture", "event_type": "failure_recorded",
            "data": make_failure_data(), "verified": False,
        })
        db.add_calibration_failure(
            seq=1, created_at="2026-07-12T10:00:00Z", **make_failure_data()
        )

        failures = db.get_recent_failures(limit=10)
        assert len(failures) == 1
        assert failures[0]["error_type"] == "hallucination"

    def test_failure_stats(self, db_path):
        db = BrainDB(db_path)
        types = ["hallucination", "hallucination", "overconfidence", "rigidity"]
        for i, etype in enumerate(types, 1):
            db.index_event({
                "seq": i, "timestamp": f"2026-07-12T10:0{i}:00Z",
                "actor": "capture", "event_type": "failure_recorded",
                "data": make_failure_data(error_type=etype), "verified": False,
            })
            db.add_calibration_failure(
                seq=i, created_at=f"2026-07-12T10:0{i}:00Z",
                **make_failure_data(error_type=etype)
            )

        stats = db.get_failure_stats()
        assert stats["hallucination"]["count"] == 2
        assert stats["overconfidence"]["count"] == 1
        assert stats["rigidity"]["count"] == 1
        assert db.get_total_failures() == 4


class TestAdvisories:
    """测试 advisories 表"""

    def test_add_and_get_pending(self, db_path):
        db = BrainDB(db_path)
        db.index_event({
            "seq": 1, "timestamp": "2026-07-12T10:00:00Z",
            "actor": "arbitrator", "event_type": "advisory_proposed",
            "data": {}, "verified": False,
        })
        db.add_advisory(seq=1, version=1, content="不要编造 API")

        pending = db.get_pending_advisory()
        assert pending is not None
        assert pending["status"] == "pending"
        assert "不要编造" in pending["content"]

    def test_update_status(self, db_path):
        db = BrainDB(db_path)
        db.index_event({
            "seq": 1, "timestamp": "2026-07-12T10:00:00Z",
            "actor": "arbitrator", "event_type": "advisory_proposed",
            "data": {}, "verified": False,
        })
        db.add_advisory(seq=1, version=1, content="测试建议")
        pending = db.get_pending_advisory()
        db.update_advisory_status(
            pending["id"], "accepted", pre_score=0.3, post_score=0.7
        )
        current = db.get_current_advisory()
        assert current["status"] == "accepted"
        assert current["post_score"] == 0.7


class TestRebuildFromWAL:
    """测试从 WAL 重建 — 核心：崩溃恢复一致性"""

    def test_rebuild_preserves_all_data(self, wal_dir, db_path):
        """写入 WAL → 同步 SQLite → 删除 SQLite → 从 WAL 重建 → 数据一致"""
        wal = WALWriter(wal_dir)
        db = BrainDB(db_path)

        # 写入 5 条失败记录
        types = ["hallucination", "overconfidence", "rigidity", "hallucination", "people_pleasing"]
        for i, etype in enumerate(types, 1):
            data = make_failure_data(error_type=etype)
            seq = wal.append(
                actor="capture", event_type="failure_recorded",
                data=data, verified=False,
            )
            db.index_event(wal.read_latest(1)[0])
            db.add_calibration_failure(seq=seq, created_at=wal.read_latest(1)[0]["timestamp"], **data)

        # 验证写入前
        assert db.get_total_failures() == 5

        # 模拟 SQLite 损坏 — 关闭并重建
        db.close()
        db2 = BrainDB(db_path)

        # 清空再重建
        wal_entries = wal.read_all()
        stats = db2.rebuild_from_wal(wal_entries)

        assert stats["events"] == 5
        assert stats["failures"] == 5
        assert db2.get_total_failures() == 5

        # 验证数据一致
        failures = db2.get_recent_failures(limit=10)
        assert len(failures) == 5
        error_types = [f["error_type"] for f in failures]
        assert "hallucination" in error_types
        assert "overconfidence" in error_types

    def test_rebuild_with_advisories(self, wal_dir, db_path):
        """重建含 advisories 的数据"""
        wal = WALWriter(wal_dir)
        db = BrainDB(db_path)

        # 写一条失败 + 一条建议
        wal.append(
            actor="capture", event_type="failure_recorded",
            data=make_failure_data(), verified=False,
        )
        wal.append(
            actor="arbitrator", event_type="advisory_proposed",
            data={"version": 1, "content": "不要编造 API"}, verified=False,
        )

        # 重建
        entries = wal.read_all()
        stats = db.rebuild_from_wal(entries)
        assert stats["events"] == 2
        assert stats["failures"] == 1
        assert stats["advisories"] == 1

    def test_empty_rebuild(self, db_path):
        """空 WAL 重建"""
        db = BrainDB(db_path)
        stats = db.rebuild_from_wal([])
        assert stats["events"] == 0
        assert db.get_total_failures() == 0
