# wind_macro_daily

美国 IRS、中国 IRS 与四组核心外汇的日间多因子策略。它严格沿用 quant-harness 的数据流：

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
- `python -m strategies.wind_macro_daily.data.wind_fetch --start ... --end ...`：调用 WindPy 与 `series_catalog.csv` 抓取已经启用的代码。

Wind 终端代码因账户权限、终端版本和指标口径而异。目录中 `candidate_verify` 项必须先通过 Wind “API 代码生成器”核验，不能把候选代码当成生产代码。

## 运行

所有命令从 `quant-harness/` 根目录执行：

```bash
python -m strategies.wind_macro_daily.backtest.run
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

信号在 t 日收盘计算，仓位滞后一日后参与 t+1 收益。回测扣除按持仓变动计算的交易成本，并保留每个产物的 SHA-256 manifest。

