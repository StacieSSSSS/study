# DECISIONS.md

记录对 `quant-harness` 有架构影响的决策：`core/` 接口变更、策略门槛（`gate:`）调整、回测窗口（`walk_forward:`）调整、新引入的回测引擎等。日常的小修复、单个策略的参数微调不需要记录在这里（可以放在策略自己的 README 或 commit message 里）。

每个决策按下方模板新增一条，追加在文件末尾，不要修改历史条目；如需变更已有决策，新增一条并在"关联"中引用被替代的条目。

## 模板

```
## D{编号} - {决策标题}

- 日期: YYYY-MM-DD
- 状态: 提议 / 已采纳 / 已废弃 / 已替代
- 背景: 为什么需要做这个决策，遇到了什么问题或约束
- 决策: 最终选择的方案
- 备选方案: 列出考虑过但未采用的方案及未采用原因
- 影响: 对架构 / 性能 / 测试 / 后续策略开发的影响
- 关联: 关联的策略 / commit / 其他 D{编号}
```

---

## D001 - 用结构化 PointInTimeFrame 而非纯靠 review 防未来函数

- 日期: 2026-06-20
- 状态: 已采纳
- 背景: 未来函数（look-ahead bias）是回测失真最常见也最隐蔽的来源，单靠代码评审容易漏看（尤其是 `.shift(-1)`、`.bfill()` 这类一行代码就能引入的问题）。
- 决策: 信号生成函数的唯一数据入口是 `core.data.point_in_time.PointInTimeFrame`，结构上只暴露 `as_of(cutoff)`/`latest(cutoff)`，不暴露完整 DataFrame；同时用 `core.validation.bias_check` 做静态扫描作为第二道防线。
- 备选方案:
  - 只靠代码评审 + 文档约定："信号函数别用未来数据"：约定容易被遗忘或在重构中破坏，未采用。
  - 在回测引擎层面整体禁止 reindex 到未来日期（如完全交给 backtrader/zipline 的事件驱动机制）：能防住一部分场景，但项目同时支持 pandas 手写回测风格，无法统一兜底，因此作为补充而非替代，未单独采用为唯一手段。
- 影响: 所有新策略的 `signals/signal.py` 必须按 `generate_signal(pit, as_of)` 签名实现；`core/` 改动需保持这个接口稳定。
- 关联: 见 `CLAUDE.md` 中 Constraints 第 1、2 条。

## D002 - 策略目录按"策略名"而非"阶段"分组

- 日期: 2026-06-20
- 状态: 已采纳
- 背景: 需要决定 `strategies/` 下是按处理阶段（data/signals/backtest 在顶层，策略只是参数）组织，还是按策略名分组（每个策略自带 data/signals/backtest）。
- 决策: 按策略名分组（`strategies/<name>/{data,signals,backtest}`），共享逻辑下沉到 `core/`。
- 备选方案:
  - 按阶段分组、策略只是配置参数：复用度更高，但策略之间逻辑差异往往不止是参数（不同数据源、不同信号结构），按阶段分组会让"策略特有逻辑"和"共享逻辑"边界模糊，且并行开发多个策略时互相干扰风险更高，未采用。
- 影响: 新增策略时复制 `_template/` 整个文件夹；跨策略复用的代码必须主动下沉到 `core/`，否则会重复。
- 关联: 无

## D003 - 回测引擎不强制统一，按策略在 config.yaml 声明

- 日期: 2026-06-20
- 状态: 已采纳
- 背景: 现有研究脚本以 pandas/numpy 手写回测为主，但后续策略可能需要 vectorbt 做参数网格搜索，或 backtrader/zipline 做更贴近实盘的事件驱动模拟。统一强制一种引擎会限制灵活性。
- 决策: 每个策略在自己的 `config.yaml` 用 `engine:` 字段声明使用的引擎（`pandas`/`vectorbt`/`backtrader`），具体依赖通过 `pyproject.toml` 的 optional-dependencies 按需安装，`core/` 的指标计算（`summarize`/`sharpe_ratio`/...）与引擎无关，可被所有策略复用。
- 备选方案:
  - 强制统一用 vectorbt：参数搜索效率高，但事件驱动逻辑表达能力弱于 backtrader/zipline，且与现有 pandas 手写代码风格不兼容，迁移成本高，未采用。
- 影响: `core/metrics/performance.py` 的输入约定为标准 pandas Series/DataFrame（returns、equity curve、weights），任何引擎只要能产出这些标准结构即可复用门槛检查和报告生成。
- 关联: 无
