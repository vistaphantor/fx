# HFM MT5 Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python skeleton that launches MetaTrader 5, logs into an HFM demo account from `.env`, opens one tiny test trade, waits briefly, closes it, and exits with clear logs.

**Architecture:** `run.py` remains a thin orchestration entrypoint while focused modules handle config loading, MT5 session management, and trade execution. Tests cover deterministic local behavior such as config validation and request construction, while broker connectivity remains a manual end-to-end check.

**Tech Stack:** Python 3, `MetaTrader5`, `python-dotenv`, `pytest`

---

### Task 1: Project Scaffolding And Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
def test_placeholder_project_structure():
    import src
    assert src is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests -q`
Expected: FAIL because `src` package does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create the package markers and dependency manifest.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests -q`
Expected: PASS for the placeholder import test.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt src/__init__.py tests/__init__.py
git commit -m "chore: scaffold mt5 bot project"
```

### Task 2: Environment Config Loader

**Files:**
- Create: `src/config.py`
- Create: `tests/test_config.py`
- Create: `.env.example`

- [ ] **Step 1: Write the failing test**

```python
def test_load_settings_reads_required_values(tmp_path, monkeypatch):
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -q`
Expected: FAIL because `load_settings` is not implemented.

- [ ] **Step 3: Write minimal implementation**

Implement a settings dataclass and `load_settings()` that reads `.env`, validates required fields, and coerces numeric settings.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -q`
Expected: PASS for config validation cases.

- [ ] **Step 5: Commit**

```bash
git add .env.example src/config.py tests/test_config.py
git commit -m "feat: add environment config loader"
```

### Task 3: MT5 Session Management

**Files:**
- Create: `src/mt5_client.py`
- Create: `tests/test_mt5_client.py`

- [ ] **Step 1: Write the failing test**

```python
def test_launch_terminal_raises_when_path_missing():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mt5_client.py -q`
Expected: FAIL because the session helper does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Implement terminal launch and MT5 initialize/login wrappers that depend on injectable collaborators for testability.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mt5_client.py -q`
Expected: PASS for path validation and login/init behavior.

- [ ] **Step 5: Commit**

```bash
git add src/mt5_client.py tests/test_mt5_client.py
git commit -m "feat: add mt5 session management"
```

### Task 4: Trade Request Construction And Execution

**Files:**
- Create: `src/trade_executor.py`
- Create: `tests/test_trade_executor.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_open_order_request_for_buy_uses_ask_price():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trade_executor.py -q`
Expected: FAIL because trade request helpers do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Implement symbol selection, open-order request building, close-order request building, and a small executor that sends and validates requests through an MT5 adapter.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_trade_executor.py -q`
Expected: PASS for request construction and close-targeting behavior.

- [ ] **Step 5: Commit**

```bash
git add src/trade_executor.py tests/test_trade_executor.py
git commit -m "feat: add test trade execution workflow"
```

### Task 5: Entrypoint Wiring

**Files:**
- Create: `run.py`
- Modify: `src/config.py`
- Modify: `src/mt5_client.py`
- Modify: `src/trade_executor.py`
- Create: `tests/test_run.py`

- [ ] **Step 1: Write the failing test**

```python
def test_main_runs_open_then_close_sequence(monkeypatch):
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_run.py -q`
Expected: FAIL because the main orchestration flow is not implemented.

- [ ] **Step 3: Write minimal implementation**

Implement `run.py` orchestration that loads settings, launches MT5, logs in, executes the test trade, waits, closes the trade, and returns a process exit code.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_run.py -q`
Expected: PASS for orchestration behavior.

- [ ] **Step 5: Commit**

```bash
git add run.py src/config.py src/mt5_client.py src/trade_executor.py tests/test_run.py
git commit -m "feat: add runnable hfm mt5 skeleton"
```

### Task 6: End-To-End Verification

**Files:**
- Modify: `README.md` (optional follow-up if we add one)

- [ ] **Step 1: Install dependencies**

Run: `python -m pip install -r requirements.txt`
Expected: Dependencies install successfully.

- [ ] **Step 2: Run automated verification**

Run: `pytest -q`
Expected: All tests pass.

- [ ] **Step 3: Run manual broker smoke test**

Run: `python run.py`
Expected: MT5 launches, login succeeds, one tiny test trade opens and closes on the HFM demo account.

- [ ] **Step 4: Capture any follow-up gaps**

Document any environment-specific issues such as missing MT5 install path or credentials.

- [ ] **Step 5: Commit**

```bash
git add .
git commit -m "test: verify hfm mt5 skeleton workflow"
```
