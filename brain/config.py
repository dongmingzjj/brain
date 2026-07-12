"""Brain 全局配置"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BrainConfig:
    """Brain 配置 — Phase 0 版"""

    # 存储路径
    brain_dir: str = r"D:\devTools\brain"
    wal_dir: str = r"D:\devTools\brain\data\wal"
    db_path: str = r"D:\devTools\brain\data\brain.db"
    benchmark_dir: str = r"D:\devTools\brain\data\benchmarks"

    # WAL 分片参数
    max_entries_per_shard: int = 10000

    # 校准参数
    capture_batch_size: int = 20        # 每次分析多少轮对话
    advisory_review_interval: int = 20  # 每积累多少条新失败重新生成建议
    test_set_size: int = 10             # 留出集大小

    # LLM 配置（Phase 0 暂不使用，后续接入）
    llm_base_url: str = ""
    llm_model: str = ""
    llm_api_key: str = ""

    def ensure_dirs(self):
        """确保所有目录存在"""
        for d in [self.brain_dir, self.wal_dir, self.benchmark_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)
