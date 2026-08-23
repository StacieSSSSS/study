# 手工 Wind 数据与验证 Handbook

## 1. 文件契约

项目接受一个 `.xlsx` 工作簿，默认文件名 `Raw_wind.xlsx`，也可通过命令行 `--workbook` 或环境变量 `WIND_MACRO_WORKBOOK` 指定。

`Price_raw` 必须保持以下结构：第 2 行为 Wind 指标名称，第 3 行为单位，第 4 行起第一列是日历日期。Wind 导出的 `0` 在本 sheet 一律视为非交易日或缺失，不视为真实价格。程序只使用工作日，并最多向前填充 5 个工作日；回测截止日在所有启用资产的最后有效观测日中取最早值。

`Macro` 必须保持以下结构：第 2 行为指标名称，第 3 行为单位，第 4 行为每列最新数据点的更新时间，第 5 行起第一列为 reference period（月末）。列名和列号会同时与 `data/release_dictionary.yaml` 校验，防止手工插列后静默错位。

每次更新前后先运行：

```bash
python -m strategies.wind_macro_daily.data.profile_workbook \
  --workbook ../../Raw_wind.xlsx \
  --run-id raw_wind_YYYYMMDD
```

审计产物位于 `reports/wind_macro_daily/data_audits/<run_id>/`。若列名变化、必需 sheet 缺失或日期重复，程序直接失败，不继续回测。

## 2. 未来信息控制

宏观表中的日期是统计期末，不是市场可得日。加载器先根据 dictionary 中的发布机构、时区、典型时间和保守发布日期规则计算 `release_ts`，再按香港 16:30 决策时钟转换成 `available_session`：

1. 香港 16:30 前已发布且当日为工作日，可在当日决策使用。
2. 16:30 后、周末发布或只有日期没有精确时间的，顺延到下一个香港工作日。
3. 最新一期若能与官方日历完全核实，使用官方时间；若 Wind 更新时间晚于官方 headline 或时间不明，则使用较晚/更保守的可得日。
4. 中国 GDP 和美国 GDP 只保留季末行；中国 1—2 月合并指标删除 1 月占位行。
5. 宏观 `0` 不统一删除。CPI/PPI 等允许真实 0；PMI、失业率等不可能为 0 的指标才依据 dictionary 的 `zero_is_valid` 和合理范围清洗。

所有每日信号仍只能从 `PointInTimeFrame.as_of()` 取得当时可见数据，持仓滞后一日参与收益。

重要限制：当前工作簿是“今天看到的修订后历史”，第 4 行不能还原 2020 年以来每次修订前的数值。因此本项目能防住把 reference date 当 release date 的直接未来函数，但无法凭该文件彻底消除 revision leakage。严格生产回测还需补充四列：`reference_date`、`release_ts`、`vintage_id`、`value`；在此之前，macro 结果标记为 research-only。

## 3. 因子构建

每个宏观序列先按 dictionary 做变换，再用仅包含当时及以前发布值的滚动窗口标准化：月度指标 36 期、至少 12 期；季度指标 20 期、至少 6 期；z-score 截断在 ±3。

- PMI 使用 `value - 50`。
- 同比增长、货币和就业指标多使用 3 个月变化。
- GDP 使用 4 个季度变化。
- 月环比通胀与工业环比使用水平值。

美国国债各期限使用美国增长、就业和通胀篮子，宏观走强或通胀上升对应减少久期；USDCNY 使用中美增长、通胀、就业和流动性差的方向性篮子；USDJPY 使用美国端篮子；AUDUSD 加入中国需求并对美国端取反；EURUSD 因工作簿没有欧元区数据，只能使用美国端的反向代理。具体权重在 `config.yaml -> data.manual_excel.macro_baskets`，属于待 walk-forward 验证的研究参数，不是已证明的经济真值。

价格因子包括 20/60 日动量和 60 日均值回归；宏观因子使用上面的发布时点篮子；综合因子按资产类别权重合成，并在某个输入缺失时只在可用因子间重新归一化。当前工作簿没有可审计的 FX carry 输入，所以 carry 单因子会明确显示为不可用。

## 4. Walk-forward 与参数冻结

运行：

```bash
python -m strategies.wind_macro_daily.validation.run \
  --data-mode manual_excel \
  --workbook ../../Raw_wind.xlsx \
  --run-id raw_wind_YYYYMMDD
```

验证器对每个交易标的的 `momentum`、`carry_signal`、`mean_reversion`、`macro_signal`、`composite_signal` 分别运行相同的滚动窗口，不再用跨资产组合 Sharpe 代替单标的判断。当前为 756 个交易日训练窗口、252 个交易日测试窗口、每次前移 252 日。参数在进入每个 OOS 窗口前固定，不使用测试窗口调参。

参数与样本外结果严格分开：

- `reports/wind_macro_daily/parameters/<run_id>_walk_forward.yaml`：完整配置快照。
- `reports/wind_macro_daily/parameters/<run_id>_instrument_factor_parameters.csv`：每个标的和因子的核心参数表。
- `reports/wind_macro_daily/walk_forward/<run_id>/windows_by_instrument_factor.csv`：逐标的、逐因子、逐窗口 OOS 指标。
- `reports/wind_macro_daily/walk_forward/<run_id>/factor_effectiveness.csv`：跨窗口汇总、有效标签及失败原因。
- `reports/wind_macro_daily/walk_forward/<run_id>/oos_daily_returns_by_instrument_factor.csv.gz`：逐标的逐因子的每日 OOS 收益和仓位。
- `reports/wind_macro_daily/walk_forward/<run_id>/harness_status.json`：供 harness 消费的有效组合、数据缺口、开放改进项和产物路径。
- `reports/wind_macro_daily/visualizations/<run_id>/`：每个标的的净值/回撤图和有效性热图。

任何参数变更都必须生成新的 `run_id`，不能覆盖先前结果。不得根据同一组 OOS 结果反复调整参数后仍称其为样本外；发生这种情况必须推进验证区间或保留最终 untouched holdout。

## 5. 当前还需补充的数据

当前美国国债 2Y/5Y/10Y/30Y 可用久期近似分别回测。仍缺：5Y SOFR OIS/IRS par rate、5Y FR007 IRS par rate、USD/CNY/JPY/AUD/EUR 的匹配期限 OIS 或 FX forward points，以及美国以外经济体的宏观数据。完整清单由 profile 命令写入 `required_data_gaps.csv`，持续改进规则保存在 `IMPROVEMENTS.yaml`。
