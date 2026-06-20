# CLAUDE.md

本文件为 Claude Code 在 `test06` 项目中工作时的指导文档。所有自动化代理在修改代码前必须先阅读本文件。

## Project Overview

- **项目名**: test06
- **目标**: 基于 FastAPI + SQLite 的后端服务，遵循分层架构（路由 / 业务逻辑 / 数据访问分离）。
- **运行环境**: Python 3.11
- **核心原则**: 路由层（`api/`）只负责参数校验和协议转换，绝不直接访问数据库；业务规则集中在 `services/`；所有数据库读写集中在 `repositories/`。

## Commands

所有命令通过 `Makefile` 统一入口，禁止绕过 Makefile 直接拼装命令（便于 CI 与本地行为一致）。

| 用途 | 命令 | 说明 |
|---|---|---|
| 完整校验（默认） | `make verify` | lint + typecheck + test-critical，日常提交前必跑 |
| 完整校验（含全量测试） | `make verify-full` | lint + typecheck + test-full，发版/合并前跑 |
| 代码风格检查 | `make lint` | 等价 `ruff check .` |
| 静态类型检查 | `make typecheck` | 等价 `pyright` |
| 关键路径测试 | `make test-critical` | 等价 `pytest -m critical` |
| 全量测试 | `make test-full` | 等价 `pytest` |

底层工具链：

- Lint: `ruff check .`
- 类型检查: `pyright`
- 测试: `pytest`

## Architecture

```
api/            # 路由层：FastAPI 路由、请求/响应模型、参数校验、协议转换
services/       # 业务逻辑层：领域规则、用例编排、事务边界
repositories/   # 数据访问层：SQLite 读写、SQL 语句、ORM 映射
tests/          # 测试：与上面三层结构对应
```

数据流方向严格为单向依赖：

```
api/  →  services/  →  repositories/  →  SQLite
```

- `api/` 可以依赖 `services/`，不能依赖 `repositories/`。
- `services/` 可以依赖 `repositories/`，不能依赖 `api/`。
- `repositories/` 不依赖上层任何模块。

## Constraints

以下约束为硬性规则，任何代码改动违反即视为缺陷，必须修复后才能提交：

1. **路由层禁止直接查库**：`api/` 目录下的任何文件不得出现 SQL 语句、SQLite 连接对象（如 `sqlite3.connect`、`cursor.execute`）或直接调用 `repositories/` 模块。
2. **业务逻辑不下沉到路由层**：条件判断、状态机、业务校验（区别于参数格式校验）必须写在 `services/`。
3. **数据访问不上浮到业务层**：`services/` 不直接拼接 SQL 或操作数据库连接，必须通过 `repositories/` 暴露的函数/方法。
4. **跨层调用不允许跳层**：`api/` 不能直接 import `repositories/` 中的任何符号。
5. **每层职责单一**：新增代码前先确认应落在哪一层，不确定时优先放在 `services/`。

## Testing

- 测试目录结构镜像源码结构：`tests/api/`、`tests/services/`、`tests/repositories/`。
- 使用 `pytest` marker 区分用例等级：
  - `@pytest.mark.critical`：核心路径（鉴权、数据一致性、对外契约），`make test-critical` 仅运行这些。
  - 未标记或其他 marker 的用例归入 `make test-full`。
- 新增/修改业务逻辑必须补充或更新对应 `services/` 层测试；新增/修改数据访问必须补充或更新 `repositories/` 层测试。
- 路由层测试只验证参数校验、状态码、协议转换，不验证业务规则细节（业务规则由 services 层测试覆盖）。

## Conventions

- 命名：模块、文件、函数使用 `snake_case`；类使用 `PascalCase`。
- 类型标注：所有函数签名必须有完整类型标注（pyright 严格模式下应无 `Any` 泄漏）。
- 异常：业务异常在 `services/` 中定义并抛出专用异常类型，由 `api/` 层统一捕获并转换为 HTTP 响应，不在 `services/` 中直接构造 HTTP 响应对象。
- 数据库连接：统一通过 `repositories/` 内的连接管理函数获取，禁止在其他层新建连接。
- 提交前：本地运行 `make verify` 必须通过。

## Self-Check（6 条自检 + 纠正）

在每次完成代码修改、提交前，逐条自检；任意一条不满足，先纠正后再继续：

1. **架构边界检查**：`api/` 中是否出现 `sqlite3`、`cursor`、原始 SQL 字符串，或对 `repositories/` 的直接 import？
   - 不满足 → 将数据库调用迁移到 `services/`，由 `services/` 调用 `repositories/`。
2. **职责归位检查**：本次新增的业务判断/规则是否写在 `services/` 而非 `api/` 或 `repositories/`？
   - 不满足 → 将业务逻辑搬迁至对应 `services/` 模块。
3. **类型完整性检查**：新增/修改函数是否都有完整类型标注，`pyright` 是否无新增告警？
   - 不满足 → 补全类型标注，重跑 `make typecheck` 确认清零。
4. **测试覆盖检查**：新增/修改的 `services/` 与 `repositories/` 代码是否有对应测试？关键路径是否标记 `@pytest.mark.critical`？
   - 不满足 → 补充测试用例后重跑 `make test-critical`。
5. **命令一致性检查**：是否通过 `Makefile` target 执行校验，而非手写零散命令？
   - 不满足 → 改用 `make verify` / `make lint` 等统一入口重新验证。
6. **变更记录检查**：本次改动是否涉及架构性决策（新增依赖、改变分层、调整数据模型）？若是，是否已记录到 `DECISIONS.md`？
   - 不满足 → 补写 `DECISIONS.md` 条目后再提交。

## Self-Heal Protocol

当 `make verify` 失败时，按以下流程自动修复，**最多尝试 3 轮**：

1. **第 1 轮**：读取失败输出（lint / typecheck / test 报错），定位最小可修复范围，仅修改导致失败的代码，不做无关重构。修复后重跑 `make verify`。
2. **第 2 轮**：若仍失败，重新分析新的失败输出（可能是第 1 轮修复引入的新问题，或暴露出的深层问题），收窄范围继续修复。重跑 `make verify`。
3. **第 3 轮**：若仍失败，做最后一次修复尝试，修复后重跑 `make verify`。
4. **3 轮后仍失败**：停止自动修复，不得继续尝试第 4 轮。如实向用户报告：
   - 当前失败的具体命令与报错信息
   - 已尝试的 3 轮修复内容及结果
   - 怀疑的根因（如有）
   - 请求用户介入决策，不擅自跳过校验或使用 `--no-verify` 等方式绕过。

每轮修复只允许针对 `make verify` 报告的失败项，不允许借机引入与修复无关的改动。

## Change Log

记录本文件自身的重大调整（而非项目代码的变更，项目代码变更见 git 历史 / `DECISIONS.md`）。

| 日期 | 变更 | 说明 |
|---|---|---|
| 2026-06-20 | 初始化 | 创建 CLAUDE.md，确立项目概览、命令、架构、约束、测试、规范、自检与自愈协议 |
