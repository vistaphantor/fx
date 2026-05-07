from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence


PACKAGE_IMPORT_ALIASES = {
    "python-dotenv": "dotenv",
}


def default_requirements_path() -> Path:
    return Path(__file__).resolve().parents[1] / "requirements.txt"


def requirement_import_name(requirement: str) -> str:
    package_name = _requirement_package_name(requirement)
    return PACKAGE_IMPORT_ALIASES.get(package_name.lower(), package_name)


def load_requirement_import_names(requirements_path: str | Path) -> list[str]:
    path = Path(requirements_path)
    if not path.exists():
        raise FileNotFoundError(f"requirements file not found: {path}")

    import_names = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        import_names.append(requirement_import_name(line))
    return import_names


def import_available(import_name: str) -> bool:
    return importlib.util.find_spec(import_name) is not None


def find_missing_requirements(
    import_names: Iterable[str],
    *,
    import_available_fn: Callable[[str], bool] = import_available,
) -> list[str]:
    return [import_name for import_name in import_names if not import_available_fn(import_name)]


def ensure_requirements_satisfied(
    *,
    requirements_path: str | Path | None = None,
    python_executable: str | None = None,
    import_available_fn: Callable[[str], bool] = import_available,
    install_fn: Callable[[Sequence[str]], object] = subprocess.check_call,
) -> None:
    path = Path(requirements_path) if requirements_path is not None else default_requirements_path()
    import_names = load_requirement_import_names(path)
    missing_imports = find_missing_requirements(import_names, import_available_fn=import_available_fn)
    if not missing_imports:
        return

    executable = python_executable or sys.executable
    install_fn([executable, "-m", "pip", "install", "-r", str(path)])


def _requirement_package_name(requirement: str) -> str:
    without_marker = requirement.split(";", 1)[0].strip()
    without_extra = without_marker.split("[", 1)[0].strip()
    match = re.match(r"^[A-Za-z0-9_.-]+", without_extra)
    if not match:
        raise ValueError(f"unsupported requirement line: {requirement}")
    return match.group(0)
