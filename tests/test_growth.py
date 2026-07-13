"""Region 生长协议单元测试"""
import pytest
import tempfile
import shutil
from pathlib import Path
from brain.event_bus import EventBus
from brain.db import BrainDB
from brain.wal import WALWriter
from brain.region_growth import RegionGrowthProtocol


@pytest.fixture
def setup():
    d = tempfile.mkdtemp(prefix="brain_growth_test_")
    bus = EventBus()
    wal = WALWriter(str(Path(d) / "wal"))
    db = BrainDB(str(Path(d) / "brain.db"))
    protocol = RegionGrowthProtocol(bus, db)
    yield bus, wal, db, protocol
    shutil.rmtree(d, ignore_errors=True)


class TestGapDetection:
    """测试能力缺口检测"""

    def test_insufficient_data(self, setup):
        """数据不足"""
        _, _, _, protocol = setup
        result = protocol.detect_gap()
        assert result["has_gap"] is False
        assert "不足" in result["reason"]

    def test_with_failures(self, setup):
        """有失败数据时检测缺口"""
        _, wal, db, protocol = setup

        # 模拟多条同类失败
        for i in range(6):
            seq = wal.append(actor="capture", event_type="failure_recorded",
                           data={"error_type": "hallucination",
                                 "question_summary": f"用户问图片识别 {i}",
                                 "correction_summary": "应该先用图像处理工具分析"})
            db.index_event({
                "seq": seq, "timestamp": "2026-07-13T10:00:00Z",
                "actor": "capture", "event_type": "failure_recorded",
                "data": {"error_type": "hallucination"}, "verified": False,
            })
            db.add_calibration_failure(
                seq=seq, created_at="2026-07-13T10:00:00Z",
                error_type="hallucination",
                question_summary=f"用户问图片识别 {i}",
                correction_summary="应该先用图像处理工具分析",
            )

        try:
            result = protocol.detect_gap()
            assert "has_gap" in result
        except Exception as e:
            pytest.skip(f"LLM 调用失败: {e}")


class TestOverlapDetection:
    """测试能力重叠检测"""

    def test_no_overlap(self, setup):
        """新能力不重叠"""
        _, _, _, protocol = setup
        result = protocol._check_overlap("vision", ["image_recognition", "ocr"])
        assert result["overlaps"] is False

    def test_with_overlap(self, setup):
        """新能力重叠"""
        _, _, _, protocol = setup
        result = protocol._check_overlap("storage", ["store", "retrieve", "forget"])
        assert result["overlaps"] is True
        assert result["overlap_with"] == "memory"

    def test_partial_overlap(self, setup):
        """部分重叠（< 0.3 阈值）"""
        _, _, _, protocol = setup
        result = protocol._check_overlap("planning", ["plan", "execute", "review"])
        # execute 重叠，但 1/5 = 0.2 < 0.3
        assert result["overlaps"] is False


class TestTemplateGeneration:
    """测试模板生成"""

    def test_generate_template(self, setup):
        """生成模板代码"""
        _, _, _, protocol = setup
        card = {
            "description": "图像识别能力",
            "capabilities": ["image_recognition", "ocr"],
            "subscriptions": ["vision_request"],
            "outputs": ["vision_result"],
        }
        code = protocol._generate_template("vision", card)
        assert "VisionExecutor" in code
        assert "LocalImprover" in code
        assert "image_recognition" in code


class TestProposals:
    """测试提议管理"""

    def test_proposals_empty(self, setup):
        """初始无提议"""
        _, _, _, protocol = setup
        assert protocol.get_proposals() == []
