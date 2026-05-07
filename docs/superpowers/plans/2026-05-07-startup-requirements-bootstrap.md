# Startup Requirements Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Auto-install Python requirements before `run.py` or `quick.py` starts the trading bot.

**Architecture:** Add one standard-library-only helper responsible for reading `requirements.txt`, detecting missing imports, and invoking pip. Call that helper at entrypoint startup before third-party-dependent modules are imported.

**Tech Stack:** Python, `importlib.util`, `subprocess`, `pytest`.

---

### Task 1: Requirements Bootstrap Helper

**Files:**
- Create: `src/requirements_check.py`
- Test: `tests/test_requirements_check.py`

- [ ] **Step 1: Write failing tests**

Create tests that verify requirement parsing, missing import detection, and pip invocation through injected fake functions.

- [ ] **Step 2: Run helper tests red**

Run: `python -m pytest tests/test_requirements_check.py -q`

Expected: fails because `src.requirements_check` does not exist.

- [ ] **Step 3: Implement helper**

Create `ensure_requirements_satisfied()` with injectable `import_available_fn`, `install_fn`, and `python_executable` parameters for testability.

- [ ] **Step 4: Run helper tests green**

Run: `python -m pytest tests/test_requirements_check.py -q`

Expected: all helper tests pass.

### Task 2: Entrypoint Wiring

**Files:**
- Modify: `run.py`
- Modify: `quick.py`
- Test: `tests/test_run.py`
- Test: `tests/test_quick.py`

- [ ] **Step 1: Write failing entrypoint tests**

Assert both `run.main()` and `quick.main()` invoke `ensure_requirements_satisfied()` before the runtime setup proceeds.

- [ ] **Step 2: Run entrypoint tests red**

Run: `python -m pytest tests/test_run.py tests/test_quick.py -q`

Expected: fails because the helper is not called.

- [ ] **Step 3: Wire entrypoints**

Import `ensure_requirements_satisfied` from `src.requirements_check` and call it at the start of `main()` in both entrypoints.

- [ ] **Step 4: Run entrypoint tests green**

Run: `python -m pytest tests/test_run.py tests/test_quick.py -q`

Expected: all entrypoint tests pass.

### Task 3: Full Validation and Publish

**Files:**
- All intended app changes

- [ ] **Step 1: Run full tests**

Run: `python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Review git diff and ignored files**

Run: `git status -sb` and inspect the staged scope before pushing.

- [ ] **Step 3: Commit and push**

Commit the app changes and push the current branch to `origin`.
