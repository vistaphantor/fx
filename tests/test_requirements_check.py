from pathlib import Path

import pytest


def test_requirement_import_name_handles_known_package_aliases():
    from src.requirements_check import requirement_import_name

    assert requirement_import_name("python-dotenv") == "dotenv"
    assert requirement_import_name("MetaTrader5") == "MetaTrader5"
    assert requirement_import_name("lightgbm") == "lightgbm"


def test_load_requirement_import_names_ignores_blank_lines_and_comments(tmp_path):
    from src.requirements_check import load_requirement_import_names

    requirements_path = tmp_path / "requirements.txt"
    requirements_path.write_text(
        "\n# dev note\nMetaTrader5\npython-dotenv==1.0.0\nnumpy>=1.26\n",
        encoding="utf-8",
    )

    assert load_requirement_import_names(requirements_path) == ["MetaTrader5", "dotenv", "numpy"]


def test_find_missing_requirements_returns_only_unavailable_imports():
    from src.requirements_check import find_missing_requirements

    available = {"MetaTrader5", "dotenv"}

    missing = find_missing_requirements(
        ["MetaTrader5", "dotenv", "lightgbm"],
        import_available_fn=lambda name: name in available,
    )

    assert missing == ["lightgbm"]


def test_ensure_requirements_satisfied_installs_when_any_requirement_is_missing(tmp_path):
    from src.requirements_check import ensure_requirements_satisfied

    requirements_path = tmp_path / "requirements.txt"
    requirements_path.write_text("MetaTrader5\npython-dotenv\n", encoding="utf-8")
    install_calls = []

    ensure_requirements_satisfied(
        requirements_path=requirements_path,
        python_executable="python-test",
        import_available_fn=lambda name: name == "MetaTrader5",
        install_fn=lambda command: install_calls.append(command),
    )

    assert install_calls == [["python-test", "-m", "pip", "install", "-r", str(requirements_path)]]


def test_ensure_requirements_satisfied_skips_install_when_all_imports_are_available(tmp_path):
    from src.requirements_check import ensure_requirements_satisfied

    requirements_path = tmp_path / "requirements.txt"
    requirements_path.write_text("MetaTrader5\npython-dotenv\n", encoding="utf-8")
    install_calls = []

    ensure_requirements_satisfied(
        requirements_path=requirements_path,
        import_available_fn=lambda name: True,
        install_fn=lambda command: install_calls.append(command),
    )

    assert install_calls == []


def test_ensure_requirements_satisfied_reports_missing_requirements_file(tmp_path):
    from src.requirements_check import ensure_requirements_satisfied

    with pytest.raises(FileNotFoundError, match="requirements.txt"):
        ensure_requirements_satisfied(requirements_path=tmp_path / "requirements.txt")
