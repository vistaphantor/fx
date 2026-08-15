from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


IMMUTABLE_KEYS = (
    "trainer_version",
    "training_state_version",
    "loss_objective_version",
    "profile",
    "training_stage",
    "canonical_contract_version",
    "tokenizer_algorithm_version",
    "curriculum_fingerprint",
    "split_fingerprint",
    "hf_config_fingerprint",
    "tokenizer_fingerprint",
    "model_config",
    "target_prediction_tokens",
)


def _atomic_torch_save(payload: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def reconcile_runtime_cadence(bundle: Path, *, keep_backup: bool = True) -> tuple[int, int]:
    """Reconcile only the measured exam cadence for a resumable checkpoint.

    ``exam_steps`` is derived from a wall-clock throughput probe. It may change
    across processes even when the model/data contract is identical, so it must
    not force checkpoint abandonment. Every immutable contract field is checked
    before this function changes the saved cadence.
    """
    work = bundle / ".training"
    state_path = work / "training_state.pt"
    preflight_path = work / "preflight.json"
    if not state_path.exists():
        raise RuntimeError(f"training_state_missing:{state_path}")
    if not preflight_path.exists():
        raise RuntimeError(f"preflight_manifest_missing:{preflight_path}")

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))

    mismatches: list[str] = []
    for key in IMMUTABLE_KEYS:
        if key in state and key in preflight and state[key] != preflight[key]:
            mismatches.append(key)
    if mismatches:
        raise RuntimeError("immutable_resume_contract_mismatch:" + ",".join(mismatches))

    old_steps = int(state.get("exam_steps", 0))
    new_steps = int(preflight.get("exam_steps", 0))
    if old_steps <= 0 or new_steps <= 0:
        raise RuntimeError(
            f"invalid_exam_cadence:checkpoint={old_steps}:preflight={new_steps}"
        )
    if old_steps == new_steps:
        print(f"[ResumeCadence] already aligned exam_steps={old_steps}")
        return old_steps, new_steps

    resume_step = int(state.get("step", 0))
    if resume_step >= new_steps:
        raise RuntimeError(
            f"resume_step_exceeds_new_cadence:step={resume_step}:exam_steps={new_steps}"
        )

    if keep_backup:
        backup = state_path.with_suffix(state_path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(state_path, backup)
            print(f"[ResumeCadence] backup={backup}")

    state["exam_steps"] = new_steps
    _atomic_torch_save(state, state_path)
    print(
        f"[ResumeCadence] aligned exam_steps {old_steps} -> {new_steps}; "
        f"resume_step={resume_step}; immutable_contract=PASS"
    )
    return old_steps, new_steps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely reconcile measured exam cadence before --resume."
    )
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()
    reconcile_runtime_cadence(
        Path(args.bundle),
        keep_backup=not args.no_backup,
    )


if __name__ == "__main__":
    main()
