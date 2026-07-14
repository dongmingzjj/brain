"""Perception Region 单元测试"""
import pytest
import tempfile
import shutil
from pathlib import Path
from brain.regions.perception.executor import PerceptionExecutor
from brain.regions.perception.local_improver import LocalImprover
from brain.regions.perception.metrics import PerceptionMetrics


@pytest.fixture
def setup():
    d = tempfile.mkdtemp(prefix="brain_perc_test_")
    db = str(Path(d) / "perception.db")
    executor = PerceptionExecutor(db_path=db)
    improver = LocalImprover(executor)
    metrics = PerceptionMetrics(executor)
    yield executor, improver, metrics
    executor.close()
    shutil.rmtree(d, ignore_errors=True)


class TestIntentClassification:
    """测试意图分类"""

    def test_question(self, setup):
        executor = setup[0]
        result = executor.execute("Python 怎么读取文件？")
        assert result["intent"] == "question"
        assert result["intent_scores"]["question"] > 0

    def test_command(self, setup):
        executor = setup[0]
        result = executor.execute("帮我创建一个文件")
        assert result["intent"] == "command"

    def test_search(self, setup):
        executor = setup[0]
        result = executor.execute("查看一下这个 GitHub 项目")
        assert result["intent"] in ("search", "command")

    def test_recommendation(self, setup):
        executor = setup[0]
        result = executor.execute("推荐一个好用的数据库")
        assert result["intent"] == "recommendation"

    def test_confirmation(self, setup):
        executor = setup[0]
        result = executor.execute("嗯，继续")
        assert result["intent"] == "confirmation"

    def test_correction(self, setup):
        executor = setup[0]
        result = executor.execute("不对，这个方案有问题")
        assert result["intent"] == "correction"

    def test_unknown_intent(self, setup):
        executor = setup[0]
        result = executor.execute("xyzzy")
        assert result["intent"] == "unknown"
        assert result["confidence"] < 0.5


class TestEntityExtraction:
    """测试实体提取"""

    def test_url(self, setup):
        executor = setup[0]
        result = executor.execute("看一下 https://github.com/test/repo")
        assert "url" in result["entities"]
        assert len(result["entities"]["url"]) >= 1

    def test_code_block(self, setup):
        executor = setup[0]
        result = executor.execute("运行 `python script.py` 这个命令")
        assert "command" in result["entities"]

    def test_file_path(self, setup):
        executor = setup[0]
        result = executor.execute("查看 D:\\projects\\test.py 文件")
        assert "file_path" in result["entities"]

    def test_number(self, setup):
        executor = setup[0]
        result = executor.execute("处理 42 条数据，间隔 3.5 秒")
        assert "number" in result["entities"]


class TestComplexity:
    """测试复杂度"""

    def test_low(self, setup):
        executor = setup[0]
        result = executor.execute("你好")
        assert result["complexity"] == "low"

    def test_high(self, setup):
        executor = setup[0]
        long_text = "请帮我分析这段代码的 bug：" + "x " * 100 + " 然后运行 `python test.py` 查看 https://example.com"
        result = executor.execute(long_text)
        assert result["complexity"] in ("medium", "high")


class TestSentiment:
    """测试情感分析"""

    def test_urgent(self, setup):
        executor = setup[0]
        result = executor.execute("紧急！线上挂了，马上处理")
        assert result["sentiment"] == "urgent"

    def test_negative(self, setup):
        executor = setup[0]
        result = executor.execute("这个代码有 bug，跑不通")
        assert result["sentiment"] == "negative"

    def test_neutral(self, setup):
        executor = setup[0]
        result = executor.execute("Python 的 GIL 是什么？")
        assert result["sentiment"] == "neutral"


class TestConfidence:
    """测试置信度"""

    def test_high_confidence_clear_intent(self, setup):
        executor = setup[0]
        result = executor.execute("帮我运行 python script.py，查看输出结果")
        assert result["confidence"] > 0.4

    def test_low_confidence_ambiguous(self, setup):
        executor = setup[0]
        result = executor.execute("xyzzy")
        assert result["confidence"] < 0.5


class TestSummary:
    """测试摘要"""

    def test_summary_format(self, setup):
        executor = setup[0]
        result = executor.execute("Python 怎么读取 JSON 文件？")
        assert "[" in result["summary"]
        assert result["intent"] in result["summary"]


class TestLocalImprover:
    """测试 Local Improver"""

    def test_run_cycle_empty(self, setup):
        improver = setup[1]
        result = improver.run_cycle()
        assert "actions_taken" in result
        assert "improvement" in result

    def test_low_confidence_detection(self, setup):
        executor, improver, _ = setup
        # 制造低置信度输入
        for _ in range(10):
            executor.execute("xyzzy ambiguous")
        result = improver.run_cycle()
        # 应该检测到低置信度比例高
        low_conf_actions = [a for a in result["actions_taken"] if a["action"] == "low_confidence_alert"]
        assert len(low_conf_actions) > 0


class TestMetrics:
    """测试 Metrics"""

    def test_empty(self, setup):
        metrics = setup[2]
        result = metrics.compute_all()
        assert result["total_perceptions"] == 0

    def test_with_data(self, setup):
        executor, _, metrics = setup
        executor.execute("怎么用 Python？")
        executor.execute("帮我运行代码")
        executor.execute("推荐一个工具")

        result = metrics.compute_all()
        assert result["total_perceptions"] == 3
        assert result["intent_diversity"] >= 2
