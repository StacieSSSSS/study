.PHONY: verify verify-full lint typecheck test-critical test-full

# 日常提交前默认校验：lint + typecheck + 关键路径测试
verify: lint typecheck test-critical
	@echo "verify: OK"

# 发版 / 合并前的完整校验：lint + typecheck + 全量测试
verify-full: lint typecheck test-full
	@echo "verify-full: OK"

lint:
	ruff check .

typecheck:
	pyright

test-critical:
	pytest -m critical

test-full:
	pytest
