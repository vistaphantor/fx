# Quick USC Scalper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a separate `quick.py` USC scalper that opens many M1-direction trades and closes each quick trade at tiny floating profit.

**Architecture:** Add quick settings to `src.config.Settings`, create `src.quick_scalp_loop` for M1 direction, profit exits, margin checks, and quick entries, then add `quick.py` as the MT5 runner that forces quick mode on. Reuse `TradeExecutor` and `Mt5Session`.

**Tech Stack:** Python, pytest, MetaTrader5 module adapter.

---

### Task 1: Quick Settings

**Files:**
- Modify: `src/config.py`
- Test: `tests/test_config.py`

- [ ] Write tests for `QUICK_*` defaults and env overrides.
- [ ] Run `python -m pytest tests/test_config.py -q` and verify the new tests fail because fields do not exist.
- [ ] Add quick settings fields and parsing.
- [ ] Run `python -m pytest tests/test_config.py -q` and verify pass.

### Task 2: Quick Scalping Loop

**Files:**
- Create: `src/quick_scalp_loop.py`
- Test: `tests/test_quick_scalp_loop.py`

- [ ] Write tests for M1 direction, profit exits, max position entries, and margin stop.
- [ ] Run `python -m pytest tests/test_quick_scalp_loop.py -q` and verify failure.
- [ ] Implement quick loop helpers and `run_quick_scalp_loop`.
- [ ] Run `python -m pytest tests/test_quick_scalp_loop.py -q` and verify pass.

### Task 3: Quick Runner

**Files:**
- Create: `quick.py`
- Test: `tests/test_quick.py`

- [ ] Write tests proving `quick.py` forces quick mode on and exits when disabled code path is not active.
- [ ] Run `python -m pytest tests/test_quick.py -q` and verify failure.
- [ ] Implement `quick.py`.
- [ ] Run `python -m pytest tests/test_quick.py -q` and verify pass.

### Task 4: Regression

**Files:**
- Verify existing files only.

- [ ] Run focused tests: `python -m pytest tests/test_config.py tests/test_trade_executor.py tests/test_quick_scalp_loop.py tests/test_quick.py -q`.
- [ ] Run the full suite if practical: `python -m pytest -q`.
