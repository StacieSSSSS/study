# wind_macro_daily

美国国债、美国 IRS、中国 IRS 与四组核心外汇的日间多因子策略。主验证单元是
`instrument × factor`；跨资产组合只作为可选的资产配置结果。它严格沿用 quant-harness 的数据流：

```text
data/loader.py
  -> PointInTimeFrame
  -> factors/engine.py + signals/signal.py
  -> backtest/run.py
  -> reports/wind_macro_daily/
```

## 数据模式

- `data.mode: synthetic`：确定性合成数据，仅用于离线工程测试。
- `data.mode: wind`：读取 `data/clean/observations.csv`。该文件必须来自当时可得的 Wind 快照；宏观修订值必须带正确的可得时间，不能按 reference date 直接回填。
- `--data-mode manual_excel`：读取手工更新的 `Raw_wind.xlsx`。价格中的 `0` 按缺数处理；宏观数据按 `data/release_dictionary.yaml` 的发布时间规则进入每日面板。
- `python -m strategies.wind_macro_daily.data.wind_fetch --start ... --end ...`：调用 WindPy 与 `series_catalog.csv` 抓取已经启用的代码。

Wind 终端代码因账户权限、终端版本和指标口径而异。目录中 `candidate_verify` 项必须先通过 Wind “API 代码生成器”核验，不能把候选代码当成生产代码。

## 运行

所有命令从 `quant-harness/` 根目录执行：

```bash
python -m strategies.wind_macro_daily.backtest.run
python -m strategies.wind_macro_daily.data.profile_workbook --workbook ../../Raw_wind.xlsx
python -m strategies.wind_macro_daily.backtest.run --data-mode manual_excel --workbook ../../Raw_wind.xlsx --run-id raw_wind_20260823
python -m strategies.wind_macro_daily.validation.run --data-mode manual_excel --workbook ../../Raw_wind.xlsx --run-id raw_wind_20260823
make bias-check STRATEGY=wind_macro_daily
make walk-forward STRATEGY=wind_macro_daily
make perf-gate STRATEGY=wind_macro_daily
```

产物分开保存：

- 数据：`strategies/wind_macro_daily/data/sample/`
- 因子：`strategies/wind_macro_daily/factors/output/sample/`
- 策略持仓：`strategies/wind_macro_daily/strategy_output/sample/`
- 回测明细：`strategies/wind_macro_daily/backtest/output/sample/`
- Performance：`reports/wind_macro_daily/`

手工数据运行时进一步按用途隔离：

- 清洗数据：`data/manual/<run_id>/`（本地保存、git 忽略）
- 参数快照：`reports/wind_macro_daily/parameters/`
- 单次回测：`reports/wind_macro_daily/backtests/<run_id>/`
- 各因子 walk-forward 样本外结果：`reports/wind_macro_daily/walk_forward/<run_id>/`
- 独立策略图：`reports/wind_macro_daily/visualizations/<run_id>/`
- 工作簿与发布时间审计：`reports/wind_macro_daily/data_audits/<run_id>/`

信号在 t 日收盘计算，仓位滞后一日后参与 t+1 收益。美国国债和 IRS 的正仓位表示
多久期/收固定，收益使用 `-duration × Δyield / 100` 的一阶近似；FX 使用对数收益。
回测扣除按持仓变动计算的交易成本，并保留每个产物的 SHA-256 manifest。

有效因子按每个标的独立判定：至少 3 个 OOS 窗口、足够活跃天数、平均 Sharpe ≥ 0.5、
最差窗口 Sharpe ≥ -0.5、至少 2/3 窗口 Sharpe 为正、平均年化收益非负且最差回撤不低于
-20%。完整判定和失败原因写入 `factor_effectiveness.csv`，不是用组合 Sharpe 反推。

## 当前手工工作簿的边界

- `Price_raw` 有美国国债 2Y/5Y/10Y/30Y 和四个目标 FX；这些标的分别回测。
- 工作簿仍没有精确的美国 5Y SOFR IRS 或中国 5Y FR007 IRS。两者输出 data-unavailable 图和缺口记录，不会把美债收益率冒充 IRS。
- 工作簿没有匹配期限的五个货币利率/远期点，`carry_signal` 会被记录为“输入不可用”，而不是用 0 伪装成有效 carry。
- `Macro` 第 4 行只有每列最新一次更新时间，不是历史 vintage 表。因此发布时点已经做保守延迟，但历史修订泄漏尚不能完全排除；详见 `MANUAL_DATA_AND_VALIDATION.md`。
- 持续改进项及 harness 强制规则记录在 `IMPROVEMENTS.yaml`。
