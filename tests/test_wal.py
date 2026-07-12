"""WAL 写入器单元测试"""
import pytest
import tempfile
import shutil
from brain.wal import WALWriter


@pytest.fixture
def wal_dir():
    d = tempfile.mkdtemp(prefix="brain_wal_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


class TestWALWrite:
    """测试 WAL 追加写入"""

    def test_append_single(self, wal_dir):
        wal = WALWriter(wal_dir)
        seq = wal.append(
            actor="capture",
            event_type="failure_recorded",
            data={"error_type": "hallucination", "question": "test"},
        )
        assert seq == 1

    def test_append_multiple_sequential(self, wal_dir):
        wal = WALWriter(wal_dir)
        seqs = []
        for i in range(5):
            s = wal.append(
                actor="arbitrator",
                event_type="advisory_proposed",
                data={"version": i + 1, "content": f"advisory {i}"},
            )
            seqs.append(s)
        assert seqs == [1, 2, 3, 4, 5]

    def test_append_with_evidence(self, wal_dir):
        wal = WALWriter(wal_dir)
        seq = wal.append(
            actor="verifier",
            event_type="advisory_accepted",
            data={"advisory_seq": 1},
            evidence={"prevention_rate": 0.75, "would_prevent": 3, "total": 4},
            verified=True,
        )
        entries = wal.read_all()
        assert len(entries) == 1
        assert entries[0]["evidence"]["prevention_rate"] == 0.75
        assert entries[0]["verified"] is True


class TestWALRead:
    """测试 WAL 读取"""

    def test_read_all(self, wal_dir):
        wal = WALWriter(wal_dir)
        for i in range(3):
            wal.append(
                actor="capture",
                event_type="failure_recorded",
                data={"error_type": f"type_{i}"},
            )
        entries = wal.read_all()
        assert len(entries) == 3
        assert entries[0]["seq"] == 1
        assert entries[2]["seq"] == 3

    def test_read_from_seq(self, wal_dir):
        wal = WALWriter(wal_dir)
        for i in range(5):
            wal.append(actor="capture", event_type="test", data={"i": i})
        entries = wal.read_all(from_seq=3)
        assert len(entries) == 2
        assert entries[0]["seq"] == 4
        assert entries[1]["seq"] == 5

    def test_read_latest(self, wal_dir):
        wal = WALWriter(wal_dir)
        for i in range(5):
            wal.append(actor="capture", event_type="test", data={"i": i})
        latest = wal.read_latest(n=2)
        assert len(latest) == 2
        assert latest[0]["seq"] == 4
        assert latest[1]["seq"] == 5

    def test_read_empty(self, wal_dir):
        wal = WALWriter(wal_dir)
        entries = wal.read_all()
        assert entries == []


class TestWALRecovery:
    """测试崩溃恢复"""

    def test_recover_seq(self, wal_dir):
        wal1 = WALWriter(wal_dir)
        for i in range(5):
            wal1.append(actor="capture", event_type="test", data={})
        seq_after_5 = wal1.get_seq()
        assert seq_after_5 == 5

        # 模拟崩溃 — 新实例从同目录恢复
        wal2 = WALWriter(wal_dir)
        assert wal2.get_seq() == 5

        # 新写入应该从 6 开始
        seq = wal2.append(actor="capture", event_type="test", data={})
        assert seq == 6

    def test_rebuild_check_ok(self, wal_dir):
        wal = WALWriter(wal_dir)
        for i in range(5):
            wal.append(actor="capture", event_type="test", data={})
        check = wal.rebuild_check()
        assert check["total_entries"] == 5
        assert check["first_seq"] == 1
        assert check["last_seq"] == 5
        assert check["integrity"] == "ok"

    def test_rebuild_check_empty(self, wal_dir):
        wal = WALWriter(wal_dir)
        check = wal.rebuild_check()
        assert check["total_entries"] == 0
        assert check["integrity"] == "ok"


class TestWALSharding:
    """测试分片轮转"""

    def test_shard_rotation(self, wal_dir):
        wal = WALWriter(wal_dir, max_entries_per_shard=3)
        for i in range(7):
            wal.append(actor="capture", event_type="test", data={"i": i})

        # 应该有 3 个分片: 3+3+1
        check = wal.rebuild_check()
        assert check["total_entries"] == 7
        assert check["shards"] >= 3

    def test_read_across_shards(self, wal_dir):
        wal = WALWriter(wal_dir, max_entries_per_shard=3)
        for i in range(7):
            wal.append(actor="capture", event_type="test", data={"i": i})
        entries = wal.read_all()
        assert len(entries) == 7
        seqs = [e["seq"] for e in entries]
        assert seqs == [1, 2, 3, 4, 5, 6, 7]
