# CLAUDE.md

`quant-harness` 是宏观交易员持续开发、验证、迭代量化策略的标准化工作流框架。任何在本仓库工作的代理在改动代码前必须先阅读本文件。

## Project Overview

- **目标**：把"想法 → 信号 → 回测 → 样本外验证 → 绩效门槛检查"的流程标准化，让每个新策略都复用同一套反偏差与验证基础设施，而不是每次重新发明。
- **使用者**：宏观交易员本人，单人持续迭代多个策略（利率/汇率/久期轮动等），策略数量会随时间增长。
- **核心理念**：回测好看不代表能用——必须结构性地防止未来函数（look-ahead bias），并用样本外滚动窗口检验稳健性，再用绩效门槛把"好看但不及格"的策略挡在外面。

## Commands

统一通过 `Makefile` 入口，禁止绕过 Makefile 直接拼装命令。

| 用途 | 命令 |
|---|---|
| 日常提交前的快速校验 | `make verify` |
| 策略上线/转为实盘信号前的完整校验 | `make verify-full STRATEGY=<name>` |
| 代码风格检查 | `make lint`（`ruff check .`） |
| 静态类型检查 | `make typecheck`（`pyright`） |
| 核心模块单测 | `make test-unit`（`pytest tests/`） |
| 未来函数/偏差扫描 | `make bias-check [STRATEGY=<name>]`（不指定则扫描全部策略） |
| 样本外滚动窗口验证 | `make walk-forward STRATEGY=<name>` |
| 回测绩效门槛检查 | `make perf-gate STRATEGY=<name>` |

`verify` 不需要 `STRATEGY`（只跑代码层面的检查 + 扫描所有策略的偏差）；`verify-full` 针对单个策略，且要求该策略已经跑过一次 `python -m strategies.<name>.backtest.run` 生成 `reports/<name>/metrics.json`。

## Architecture

```
core/                       # 共享基础设施，所有策略复用
  data/point_in_time.py     # PointInTimeFrame：结构性防止读到未来数据
  validation/bias_check.py  # 静态扫描 shift(负数) / bfill 等未来函数模式
  validation/walk_forward.py# 滚动窗口样本外验证引擎
  metrics/performance.py    # sharpe / max_drawdown / turnover 等指标
  reporting/gate.py         # 绩效门槛检查（对照 config.yaml 的 gate: 字段）

strategies/
  _template/                # 新策略起点，复制改名，不要直接改 _template 本身
    config.yaml              # universe / engine / walk_forward 窗口 / gate 门槛
    data/loader.py            # load_raw()：拉数据、清洗、对齐，不含信号逻辑
    signals/signal.py         # generate_signal(pit, as_of)：只能通过 PointInTimeFrame 读数据
    backtest/run.py           # load_data() / evaluate_window() / main()

  <strategy_name>/           # 每个策略一个独立文件夹，结构与 _template 一致
    ...

tests/                      # 只覆盖 core/，策略自身的正确性靠 bias-check + walk-forward + perf-gate 把关
reports/<strategy_name>/    # 每个策略的回测产出（metrics.json 等），不提交大文件/敏感数据
```

数据流方向：

```
data/loader.py → signals/signal.py（经 PointInTimeFrame）→ backtest/run.py → reports/<name>/metrics.json → perf-gate
```

## Constraints

1. **信号逻辑禁止绕过 `PointInTimeFrame` 读取原始数据**：`signals/signal.py` 中的函数只能通过传入的 `PointInTimeFrame.as_of()` / `.latest()` 访问历史，不能直接引用模块级的原始 DataFrame 或闭包捕获未来数据。
2. **禁止使用未来函数**：不得对时间序列使用 `.shift(负数)`、`.bfill()`、`fillna(method='bfill')` 等会把未来值带到当前时刻的操作（`bias-check` 会扫描，但代码评审时也要主动避开）。
3. **策略必须自带绩效门槛**：每个 `strategies/<name>/config.yaml` 必须声明 `gate:`（至少 `min_sharpe`、`max_drawdown`），上线前必须通过 `make perf-gate`。
4. **策略必须自带样本外窗口配置**：`config.yaml` 的 `walk_forward:` 必须配置 `train_size`/`test_size`/`step`，单一全样本回测不能作为上线依据。
5. **新策略从 `_template` 复制，不直接修改 `_template`**：`_template` 永远保持可复制、不可运行（`load_raw()`/`generate_signal()` 默认抛 `NotImplementedError`）的状态。
6. **`core/` 改动影响所有策略**：修改 `core/` 下任何模块前，先确认不会破坏现有策略的接口约定（`evaluate_window` 签名、`PointInTimeFrame` 接口等），改动后要在 `DECISIONS.md` 记录。

## Testing

- `tests/` 只测 `core/` 的共享基础设施（`PointInTimeFrame`、指标计算、滚动窗口切分、偏差扫描器本身），保证地基可靠。
- 单个策略的"正确性"不是靠 `pytest`，而是靠三层检查共同把关：
  1. `make bias-check STRATEGY=<name>`：结构 + 静态扫描双重防未来函数。
  2. `make walk-forward STRATEGY=<name>`：滚动窗口样本外表现，避免单一回测区间的过拟合假象。
  3. `make perf-gate STRATEGY=<name>`：样本外指标是否达到策略自己声明的门槛。
- 新增/修改 `core/` 模块必须补充对应单测，且不能降低现有测试的覆盖范围。
- 怀疑某个策略的信号实现可能有未来函数时，先跑 `make bias-check`，再人工检查 `signals/signal.py` 是否真的只通过 `pit.as_of()`/`pit.latest()` 取数。

## Conventions

- 命名：模块/文件/函数 `snake_case`，类 `PascalCase`，策略文件夹名 `snake_case`（如 `duration_rotation`）。
- 每个策略文件夹自包含：不要让一个策略的 `signals/` 直接 import 另一个策略的内部模块；如需复用逻辑，下沉到 `core/`。
- 类型标注：`core/` 下的函数必须有完整类型标注；策略代码（`strategies/<name>/`）至少给公开函数（`load_raw`/`generate_signal`/`load_data`/`evaluate_window`/`main`）标注类型。
- 绩效门槛只能变严不能悄悄放松：如果要放宽 `gate:` 阈值，必须在 `DECISIONS.md` 记录原因（例如市场结构变化、原阈值设定有误），不能为了让检查通过而直接改数字。
- 提交前本地必须跑过 `make verify`；策略要转向实盘信号前必须跑过 `make verify-full STRATEGY=<name>`。

## Self-Check（6 条自检 + 纠正）

完成改动、提交前逐条自检，任意一条不满足先纠正再继续：

1. **未来函数检查**：本次新增/修改的信号逻辑是否只通过 `PointInTimeFrame` 读取数据？是否引入了 `shift(负数)`/`bfill` 等模式？
   - 不满足 → 改为通过 `pit.as_of()`/`pit.latest()` 取数，并跑 `make bias-check STRATEGY=<name>` 确认清零。
2. **样本外验证检查**：本次策略改动是否只在全样本回测上验证过，没跑滚动窗口？
   - 不满足 → 跑 `make walk-forward STRATEGY=<name>`，确认样本外表现而非仅样本内表现。
3. **绩效门槛检查**：`config.yaml` 的 `gate:` 是否仍然是策略改动前设定的合理门槛？是否为了通过检查而悄悄放宽了阈值？
   - 不满足 → 还原阈值，或在 `DECISIONS.md` 记录放宽的理由后再放宽。
4. **类型与风格检查**：新增/修改函数是否有类型标注，`ruff`/`pyright` 是否无新增告警？
   - 不满足 → 补全标注与风格修复，重跑 `make lint && make typecheck`。
5. **策略隔离检查**：本次改动是否让一个策略依赖了另一个策略的内部模块，而不是通过 `core/` 复用？
   - 不满足 → 把共享逻辑下沉到 `core/`，策略间保持独立。
6. **决策记录检查**：本次改动是否涉及绩效门槛调整、回测窗口调整、`core/` 接口变更等架构性决策？是否已写入 `DECISIONS.md`？
   - 不满足 → 补写 `DECISIONS.md` 条目后再提交。

## Self-Heal Protocol

当 `make verify`（或 `make verify-full`）失败时，按以下流程自动修复，**最多尝试 3 轮**：

1. **第 1 轮**：读取失败输出，定位最小修复范围（lint/typecheck 报错位置、bias-check 报出的具体行、walk-forward 报错的窗口数量、perf-gate 报出的具体超限指标），只改导致失败的代码。重跑同一条命令确认。
2. **第 2 轮**：若仍失败，重新分析新的失败输出（可能是第 1 轮修复引入的次生问题，或暴露出更深层的问题，例如 bias-check 通过但 walk-forward 揭示过拟合），收窄范围继续修复。重跑确认。
3. **第 3 轮**：最后一次修复尝试，重跑确认。
4. **3 轮后仍失败**：停止自动修复。如实向用户报告：失败的具体命令与输出、已尝试的 3 轮修复内容及结果、怀疑的根因（尤其是否怀疑数据质量问题或策略本身不稳健而非代码 bug）。不得擅自放宽 `gate:` 门槛或跳过 bias-check 来让检查通过。

**特别注意**：`perf-gate` 失败常常意味着"策略本身在样本外不行"，这不是 bug，自动修复不应该尝试"调整阈值让它通过"——这类失败应直接报告用户，由用户判断策略是否值得继续迭代。

## Change Log

| 日期 | 变更 | 说明 |
|---|---|---|
| 2026-06-20 | 初始化 | 创建 quant-harness：core/（point-in-time 数据访问、偏差扫描、滚动窗口验证、绩效指标、门槛检查）+ strategies/_template 脚手架 + Makefile/CLAUDE.md/DECISIONS.md |
| 2026-06-20 | 新增策略 | `strategies/fx_correlation`：多模型外汇相关性交易框架（强度/稳定性、背离、cointegration 三模型 + conviction 排名 + 每日刷新）。首次出现"策略内部多模型共享单一交易机制"模式，详见 DECISIONS.md D004-D006。`signals/signal.py` 的返回类型从 `_template` 的单一 float 改为 `pd.Series`（多资产权重向量），属有意为之的偏离，已在该策略 README 注明。 |
