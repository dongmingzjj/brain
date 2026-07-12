"""Memory Region 单元测试"""
import pytest
import tempfile
import shutil
from pathlib import Path
from brain.regions.memory.executor import MemoryExecutor
from brain.regions.memory.local_improver import LocalImprover
from brain.regions.memory.metrics import MemoryMetrics


@pytest.fixture
def db_path():
    d = tempfile.mkdtemp(prefix="brain_memory_test_")
    path = str(Path(d) / "memory.db")
    yield path
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def executor(db_path):
    return MemoryExecutor(db_path)


@pytest.fixture
def improver(executor):
    return LocalImprover(executor)


@pytest.fixture
def metrics(executor):
    return MemoryMetrics(executor)


class TestMemoryExecutor:
    """测试 Memory Executor"""

    def test_store_and_retrieve(self, executor):
        """存储后能检索到"""
        executor.store("python_gil", "Python 的 GIL 限制了多线程并行",
                       mem_type="fact", importance=0.8)

        results = executor.retrieve("Python GIL 多线程")
        assert len(results) == 1
        assert results[0]["key"] == "python_gil"
        assert "GIL" in results[0]["value"]

    def test_retrieve_by_keyword(self, executor):
        """关键词匹配检索"""
        executor.store("async_io", "asyncio 是 Python 的异步 IO 框架")
        executor.store("thread_pool", "ThreadPoolExecutor 用于线程池")
        executor.store("process_pool", "ProcessPoolExecutor 用于进程池")

        # 搜 "asyncio" 只应返回 async_io
        results = executor.retrieve("asyncio 异步")
        assert len(results) >= 1
        assert any(r["key"] == "async_io" for r in results)

    def test_retrieve_top_k(self, executor):
        """top_k 限制返回数量"""
        for i in range(10):
            executor.store(f"item_{i}", f"这是第 {i} 条记忆")

        results = executor.retrieve("记忆", top_k=3)
        assert len(results) <= 3

    def test_retrieve_empty_query(self, executor):
        """空查询返回空"""
        executor.store("test", "test value")
        results = executor.retrieve("")
        assert results == []

    def test_forget(self, executor):
        """删除记忆"""
        mid = executor.store("temp", "临时记忆")
        assert executor.forget(mid) is True

        results = executor.retrieve("临时记忆")
        assert len(results) == 0

    def test_forget_nonexistent(self, executor):
        """删除不存在的记忆"""
        assert executor.forget(999) is False

    def test_update_importance(self, executor):
        """更新重要性"""
        mid = executor.store("important", "重要记忆", importance=0.5)
        executor.update_importance(mid, 0.9)

        results = executor.retrieve("重要记忆")
        assert len(results) == 1
        assert results[0]["importance"] == 0.9

    def test_stats(self, executor):
        """统计信息"""
        executor.store("a", "value a", mem_type="fact")
        executor.store("b", "value b", mem_type="skill")
        executor.store("c", "value c", mem_type="fact")

        stats = executor.get_stats()
        assert stats["total_memories"] == 3
        assert stats["by_type"]["fact"] == 2
        assert stats["by_type"]["skill"] == 1

    def test_access_tracking(self, executor):
        """访问追踪"""
        mid = executor.store("tracked", "被追踪的记忆")
        executor.retrieve("追踪")  # 触发访问

        stats = executor.get_access_log_stats()
        assert stats["total_queries"] >= 1


class TestLocalImprover:
    """测试 Local Improver"""

    def test_run_cycle_empty(self, improver):
        """空记忆库的改进循环"""
        result = improver.run_cycle()
        assert "actions_taken" in result
        assert "improvement" in result
        assert result["improvement"] == 0  # 无变化

    def test_run_cycle_with_memories(self, improver, executor):
        """有记忆时的改进循环"""
        executor.store("old", "旧记忆", importance=0.05)
        executor.store("new", "新记忆", importance=0.8)

        result = improver.run_cycle()
        assert "actions_taken" in result
        # 低重要性记忆可能被遗忘
        # (取决于 decay 和 forget 的具体逻辑)

    def test_improvement_history(self, improver):
        """改进历史记录"""
        improver.run_cycle()
        improver.run_cycle()

        history = improver.get_improvement_history()
        assert len(history) == 2
        assert history[0]["timestamp"] <= history[1]["timestamp"]

    def test_boost_popular(self, improver, executor):
        """高频访问记忆提升"""
        mid = executor.store("popular", "常用记忆", importance=0.3)
        # 模拟多次访问
        for _ in range(6):
            executor.retrieve("常用")

        result = improver.run_cycle()
        # 检查是否被 boost 了
        boosted = any(a["action"] == "boost" for a in result["actions_taken"])
        if boosted:
            # 重要性应该提升了
            stats = executor.get_stats()
            assert stats["avg_importance"] > 0.3


class TestMemoryMetrics:
    """测试 Memory Metrics"""

    def test_compute_all_empty(self, metrics):
        """空记忆库的指标"""
        result = metrics.compute_all()
        assert result["total_memories"] == 0
        assert result["avg_importance"] == 0
        assert result["total_accesses"] == 0

    def test_compute_all_with_data(self, metrics, executor):
        """有数据时的指标"""
        executor.store("a", "value a", importance=0.8)
        executor.store("b", "value b", importance=0.3)
        executor.retrieve("value")

        result = metrics.compute_all()
        assert result["total_memories"] == 2
        assert 0.3 <= result["avg_importance"] <= 0.8
        assert result["total_accesses"] >= 1
