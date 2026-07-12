# Brain

> 分区制自改进 Agent 架构 — 让 AI 学会自我校准

Brain 是一个模块化的自改进系统，目标是通过认知校准（cognitive calibration）减少 LLM 的常见偏差：幻觉、过度自信、迎合用户、僵化模式。

## 核心设计

```
Region 分区 (Perception / Memory / Action)
    ↕ Event Bus (订阅制信号路由)
Arbitrator (LLM 决策，只在需要时介入)
    ↕
Verifier (非 LLM，确定性验证)
    ↕
WAL + SQLite (Event Sourcing 存储)
```

**三种改进路径：**
- 区域自改进（非 LLM 确定性算法，零 API 调用）
- 跨区评估（Region 互相打标签，不互相改输出）
- Arbitrator 决策（LLM 语义理解 + Verifier 验证）

**递归破解：** LLM 不验证自己。Verifier（确定性代码）验证 LLM。人负责 Verifier 的规则。三层各司其职。

## Phase 0: 认知校准 MVP

当前阶段验证 Arbitrator + Verifier + WAL 三角能否跑通。

### 场景

离线扫描对话历史 → 识别校准失败 → 生成校准建议 → A/B 行为测试验证效果。

### 使用

```bash
cd D:\devTools\brain

# 导入种子校准失败数据
python run_phase0.py seed

# 生成校准建议（基于训练集）
python run_phase0.py arbitrate

# A/B 行为测试验证（留出集）
python run_phase0.py verify

# 查看报告
python run_phase0.py report

# 崩溃恢复（从 WAL 重建 SQLite）
python run_phase0.py rebuild
```

### 当前结果

| 指标 | 值 |
|------|-----|
| B 更好率（留出集） | 75% (3/4) |
| 防住已知错误 | 100% (4/4) |
| 校准失败类型覆盖 | 幻觉/过度自信/迎合/僵化 |

## 架构文档

详见 `docs/architecture.md`（完整架构蓝图）。

## 目录结构

```
brain/
├── bsp.py          BSP 信号协议
├── wal.py          分片 WAL 写入器
├── db.py           SQLite 索引层
├── capture.py      校准失败捕获
├── arbitrator.py   校准建议生成器
├── verifier.py     A/B 行为测试验证器
├── llm.py          LLM 调用封装
├── config.py       配置
tests/              单元测试
run_phase0.py       Phase 0 主入口
```

## 技术栈

- Python 3.11+
- SQLite（索引 + 物化视图）
- WAL 分片文件（真相源）
- 智谱 GLM API（LLM 调用）
- pytest（测试）

## 参考项目

- [TencentDB Agent Memory](https://github.com/TencentCloud/TencentDB-Agent-Memory) — 四层记忆架构
- [BettaFish](https://github.com/666ghj/BettaFish) — 多 Agent 论坛协作
- [MiroFish](https://github.com/666ghj/MiroFish) — 群体智能引擎
- [Darwin Gödel Machine](https://github.com/jennyzzt/dgm) — 自改进循环
- [HyperAgents](https://github.com/facebookresearch/HyperAgents) — Meta 自指涉自改进

## 许可

MIT
