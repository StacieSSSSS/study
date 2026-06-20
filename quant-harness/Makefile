PYTHON ?= python3
STRATEGY ?=

.PHONY: verify verify-full lint typecheck test-unit bias-check walk-forward perf-gate

# Fast loop — run before every commit.
verify: lint typecheck test-unit bias-check
	@echo "verify: OK"

# Full validation — run before promoting a strategy / before trading on it live.
# Requires STRATEGY=<name> and an existing reports/<name>/metrics.json (run the
# strategy's backtest main() first).
verify-full: verify walk-forward perf-gate
	@echo "verify-full: OK"

lint:
	ruff check .

typecheck:
	pyright

test-unit:
	pytest tests/

# Scans every strategy under strategies/ (excluding the _template) for
# look-ahead bias patterns, or just STRATEGY=<name> if given.
bias-check:
ifeq ($(STRATEGY),)
	@status=0; \
	for d in strategies/*/; do \
		name=$$(basename "$$d"); \
		[ "$$name" = "_template" ] && continue; \
		[ -f "$$d/config.yaml" ] || continue; \
		$(PYTHON) -m core.validation.bias_check "$$d" || status=1; \
	done; \
	exit $$status
else
	$(PYTHON) -m core.validation.bias_check strategies/$(STRATEGY)
endif

walk-forward:
	@if [ -z "$(STRATEGY)" ]; then echo "usage: make walk-forward STRATEGY=<name>"; exit 1; fi
	$(PYTHON) -m core.validation.walk_forward run --strategy=$(STRATEGY)

perf-gate:
	@if [ -z "$(STRATEGY)" ]; then echo "usage: make perf-gate STRATEGY=<name>"; exit 1; fi
	$(PYTHON) -m core.reporting.gate --strategy=$(STRATEGY)
