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

## D004 - USDCNH 用 USDCNY 代用（yfinance 数据源限制）

- 日期: 2026-06-20
- 状态: 已采纳
- 背景: `fx_correlation` 策略最初的货币对清单包含 USDCNH（离岸人民币），但 yfinance 的 `USDCNH=X`/`CNH=X` 几乎没有可用历史（实测 2 年历史只有 1 行数据），无法用于滚动相关性/回测。
- 决策: 用 `USDCNY=X`（在岸人民币）作为代用，在 `config.yaml` 和 `README.md` 中明确标注这是代用且数据质量一般（~14% 交易日是停滞重复值）。
- 备选方案:
  - 暂时跳过该货币对：保留了清单完整性诉求但少了一个用户明确要求的对，未采用（用户选择了"用 USDCNY 代用"而非跳过）。
  - 接入其他数据源单独拉 CNH：增加架构复杂度（需要额外数据源/凭证管理），且当前阶段 yfinance 已能覆盖其余 6 个对，暂不必为单一货币对引入第二套数据管道，未采用。
- 影响: 任何涉及 USDCNY 的相关性/cointegration 结果都应预期噪音更大；如果未来获得更好的 CNH 数据源（比如现有 repo 里的 `CNH_fwd.xlsx` 手动导入数据风格），应替换 `config.yaml` 的 `ticker_map.USDCNY` 对应的数据接入方式。
- 关联: `strategies/fx_correlation/config.yaml`, `strategies/fx_correlation/README.md`

## D005 - 多模型共享一套交易机制，只在"选哪些对"上分叉

- 日期: 2026-06-20
- 状态: 已采纳
- 背景: `fx_correlation` 要求建立 3 个模型（强度/稳定性、背离、cointegration）并各自给出回测结果用于合成 conviction。如果三个模型各自独立实现交易执行逻辑，会有大量重复代码且难以保证可比性。
- 决策: 三个模型只实现一个 `selection_score(metrics, cfg) -> float` 函数（决定"选哪些货币对组合"），共享的 hedge-ratio z-score 均值回归交易机制统一放在 `models/base.py`。Conviction 通过三模型对全部组合的排名取平均得到（用户在对话中确认排名法而非原始分数加权）。
- 备选方案:
  - 三个模型各自从头实现交易逻辑：更灵活但难以比较三者的"贡献"是否来自选股能力还是执行细节差异，未采用。
  - 模型分数直接加权求和（而非排名）：分数量纲不同（强度分数 vs p-value vs 背离幅度），直接加权需要额外的归一化假设，用户明确选择了排名法以避免这个问题，未采用。
- 影响: 新增第 4 个模型只需实现 `selection_score` + `NEED_ADF`，并在 `backtest/run.py`/`reporting/conviction.py` 的模型列表里注册，详见 `strategies/fx_correlation/README.md`"Extending the framework"。
- 关联: `strategies/fx_correlation/models/base.py`

## D006 - 相关性/cointegration 计算限定在滚动窗口内，而非全样本

- 日期: 2026-06-20
- 状态: 已采纳
- 背景: 初版实现里，回测每个 as_of 日期都对截止当日的*全部*历史（point-in-time 安全，但样本量随回测推进不断增长）计算相关性和 ADF 检验，导致单次全样本回测（~1300 个交易日）耗时从几分钟膨胀到完全跑不完（ADF 检验的 lag 搜索成本随样本量增长，且 21 个组合 × 上千个交易日 × 3 个模型，总调用量级在十万次以上）。
- 决策: 把相关性/z-score/ADF 检验的输入限定在最长配置窗口（当前是 `windows.12m=252` 个交易日）内的最近数据，而不是 point-in-time 视图里全部可见的历史；同时把 `adfuller` 的 `maxlag` 从默认（随样本量增长）固定为 5。
- 备选方案:
  - 保持全样本输入，只优化实现细节（如用 numpy 代替 pandas 算相关系数）：实测有帮助但不够（21 组合 × 上千日仍然太慢），且全样本 ADF 检验在回测后期会用到 4-5 年前的数据，与"用最近相关性判断现在能不能交易"的初始设计意图本身就不一致，未采用为唯一手段。
  - 把 ADF 检验只做一次（在 train 窗口）而非每个 as_of 都重新检验：性能最好，但会让 cointegration 关系的"实时性"判断退化为固定假设，与 Model C 的设计目的（实时筛选当前仍然 cointegrated 的对）冲突，未采用。
- 影响: 相关性/cointegration 结果现在反映"最近 12 个月"的关系，而非"从数据起点到当前"的关系——这本身更贴近实盘判断逻辑，且是预期之中的副作用而非妥协。全量回测（`make walk-forward STRATEGY=fx_correlation`、`python3 -m strategies.fx_correlation.backtest.run`）耗时降到 ~3-4 分钟，可接受。
- 关联: `strategies/fx_correlation/models/base.py::compute_combo_metrics`, `strategies/fx_correlation/lib/correlation.py`, `strategies/fx_correlation/lib/cointegration.py::adf_pvalue`

## D007 - Model C 用 ADF 统计量排序，p-value 只做筛选门槛

- 日期: 2026-06-21
- 状态: 已采纳
- 背景: 用户要求 conviction 报告给出每条记录的具体驱动信号。实现时发现 Model C 原来的排序依据（`-adf_p`）在实盘数据上经常打平——statsmodels 的 ADF p-value 是查表近似值，当检验统计量远超表格范围时会直接饱和成 0.0（实测 21 个组合里有 9 个同时显示 p=0.000），导致这些组合之间的相对排序其实是按枚举顺序而非真实统计证据强弱决定的。
- 决策: `lib/cointegration.py` 新增 `adf_test()` 同时返回检验统计量和 p-value；`model_c_cointegration.selection_score` 改为先用 p-value 做"是否够显著"的门槛过滤（不变），再用统计量（更负=更显著，且不会饱和）排序通过门槛的组合。
- 备选方案:
  - 保持只用 p-value：实现最简单，但排序在"强 cointegration 扎堆"的场景下会失真，与用户要求的"具体原因"诉求冲突（无法解释为什么 A 排在 B 前面，因为数字看起来一样），未采用。
  - 改用更精细的 p-value 计算方式（如 MacKinnon 完整分布而非查表近似）：能解决饱和问题，但 statsmodels 的 adfuller 本身就是查表近似，没有现成的高精度替代，自己实现代价过高，未采用。
- 影响: Model C 在回测中选中的组合可能与改动前不同（依赖统计量而非 p-value 排序），因此重新跑了 `backtest/run.py` 和 `reporting/model_backtests.py` 刷新 `reports/fx_correlation/` 下的指标和图表。`ComboMetrics` 新增 `adf_stat` 字段（默认 NaN，向后兼容）。
- 关联: `strategies/fx_correlation/lib/cointegration.py::adf_test`, `strategies/fx_correlation/models/model_c_cointegration.py`, `strategies/fx_correlation/reporting/conviction.py::_model_detail`

## D008 - fx_correlation 的 reports/ 例外提交到 git

- 日期: 2026-06-21
- 状态: 已采纳
- 背景: 项目默认约定（`CLAUDE.md` Architecture 一节）是 `reports/<strategy_name>/` 不提交，因为是可随时重新生成的运行产物。用户希望能在 GitHub 上直接看到 fx_correlation 跑出来的结果（图表、conviction 报告），不想每次都要本地重新跑。
- 决策: 在 `.gitignore` 里对 `reports/fx_correlation/` 单独开例外（`!reports/fx_correlation/` + `!reports/fx_correlation/**`），其余策略的 `reports/` 仍按默认约定忽略。今后每次为用户刷新 fx_correlation 的回测/conviction 报告后，连同 `reports/fx_correlation/` 下的产物一起提交。
- 备选方案:
  - 改成全局规则（所有策略的 reports/ 都提交）：用户只针对 fx_correlation 提出需求，没必要扩大到尚不存在的未来策略，且大部分策略的回测产物会比这个更大/更频繁变化，未采用。
  - 设置真正的每日定时任务（cron）自动跑回测并推送：是更彻底的"自动"，但涉及无人值守运行 3-4 分钟回测并推送到 GitHub，影响范围更大，需要用户单独确认是否要这种程度的自动化，本次先只做"每次手动刷新时顺带提交"，未直接采用。
- 影响: `conviction_<date>.{csv,md}` 这类按日期累积的文件会让 `reports/fx_correlation/` 随时间持续增长（用户已知晓并接受这个权衡）；其余策略不受影响。
- 关联: `.gitignore`, `quant-harness/CLAUDE.md`（Architecture 一节的默认约定保持不变，仅 fx_correlation 例外）

## D009 - 仓位管理用"z-score 水平 × z-score 动量"两个信号合成 7 档动作

- 日期: 2026-06-21
- 状态: 已采纳
- 背景: 用户要求对正在交易的货币对组合做择时，结合择时信号和动量水平，给出大力买入/买入/谨慎加仓/持有/观望/减仓/获利了结七档动作，并要求结果里给出依据的信号水平和参数。
- 决策: 用现有的 z-score（已经是模型用来交易的"价差有多极端"信号）作为择时信号本身（分三档：极端/中等/中性，门槛 entry_z/exit_z），再算这个 z-score 在过去 `momentum_lookback` 个交易日里的变化方向作为"动量"（是在往均值收敛还是继续背离，门槛 momentum_threshold），3×3=9 种组合映射到 7 档动作（`lib/momentum.py::ACTIONS`）。两组阈值单独放在 `config.yaml` 的 `position_management:` 下，与三个模型自己的 entry_z/exit_z 解耦（可独立调整）。
- 备选方案:
  - 用单个货币对自身的价格动量（趋势跟踪指标，如均线交叉）做择时，与相关性信号无关：这是另一套独立体系，会让"为什么这个动作"难以用同一套已有指标解释，且需要新引入趋势跟踪逻辑，未采用。
  - 给 9 个格子各自独立定义动作（不复用标签）：会变成 9 档而不是用户要求的 7 档，且部分格子（如"中等且背离"和"中性且无动量"）背后的操作建议其实是一样的（都是"没有边际优势，先别动"），强行区分没有意义，未采用。
- 影响: 这一层完全基于已有的 `compute_combo_metrics`/`ComboMetrics` 复用（用同一个 beta，多算一次更早日期的 z-score），没有改动 `core/` 或现有模型逻辑。Conviction 排名和这里的动作可能合理地不一致——conviction 基于背离/强度信号，可能存在滞后，等动作层观察到时价差已经回归，会显示"获利了结"而不是"买入"，这是预期行为不是 bug，已在 README 里写明。
- 关联: `strategies/fx_correlation/lib/momentum.py`, `strategies/fx_correlation/reporting/position_management.py`

## D010 - fx_correlation 增加每日定时云端自动跑+推送

- 日期: 2026-06-21
- 状态: 已采纳
- 背景: D008 当时的备选方案里提到过"真正的每日定时任务"但未采用，只做了"手动刷新时顺带提交"。用户随后明确要求要真正的每天自动跑（无需手动触发）。
- 决策: 用 claude.ai 的云端 routine（`RemoteTrigger`，cron `7 6 * * *`，即每天 UTC 06:07）跑：conviction 报告 → 混合回测 → 三模型对比 → 出图 → 提交+推送到 `StacieSSSSS/study`。云端任务用的是全新 checkout（不共享这个 Codespace 的已装依赖/数据缓存），失败时直接停止并报告，不会硬推残缺结果或自己改策略代码。
- 备选方案:
  - 在本地 Codespace 里用 `CronCreate`/本地 cron 定时跑：本地任务只在 REPL 空闲时触发、且 Codespace 不保证一直开着，不适合"每天必须跑"的需求，未采用。
- 影响: 之后如果改了 `reporting/position_management.py` 等新模块，需要记得同步更新云端 routine 的运行步骤（routine 的 prompt 是写死的文本，不会自动感知代码变化）。routine 管理入口：https://claude.ai/code/routines/trig_01SgYsm7Xz34XYhaeK3BSPQj。
- 关联: D008
