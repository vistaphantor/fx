from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.language.training_runtime import main, set_token_scheduled_lr

# Backward import name for existing architecture tests; the implementation itself
# exists only once in training_budget.py.
_set_token_scheduled_lr = set_token_scheduled_lr


if __name__ == "__main__":
    main()
