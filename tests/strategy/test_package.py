import importlib


def test_strategy_package_exposes_no_public_api_yet():
    module = importlib.import_module("src.strategy")

    assert module.__name__ == "src.strategy"
    assert module.__package__ == "src.strategy"
    assert module.__all__ == ()
