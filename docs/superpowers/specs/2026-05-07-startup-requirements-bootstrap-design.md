# Startup Requirements Bootstrap Design

## Goal

When `run.py` or `quick.py` starts on a freshly pulled machine, the app should ensure packages from `requirements.txt` are installed before the trading bot imports dependencies such as `python-dotenv`, `MetaTrader5`, `numpy`, or `lightgbm`.

## Design

Add a small standard-library-only helper in `src/requirements_check.py`. It will read the repository-level `requirements.txt`, map known package names to import names, check imports, and run `python -m pip install -r requirements.txt` only when at least one required package is missing.

Both entrypoints will call the helper before app imports that depend on third-party packages. This keeps the startup path simple: pull the repository, run `python run.py` or `python quick.py`, and let the entrypoint repair missing Python dependencies before continuing.

## Error Handling

If `requirements.txt` is missing, the helper raises a clear `FileNotFoundError`. If pip installation fails, `subprocess.check_call` raises and startup stops with the pip failure rather than continuing into a half-installed runtime.

## Testing

Tests will cover package-name parsing, missing-package detection, no-op behavior when packages are already importable, and pip invocation when packages are missing. Entrypoint tests will verify `run.py` and `quick.py` call the helper before their main runtime setup.
