from __future__ import annotations

TRAINER_VERSION = 9
TRAINING_STATE_VERSION = 9
PREFLIGHT_MANIFEST_VERSION = 9
SEED = 42
UPGRADABLE_DATA_CONTRACT_KEYS = frozenset({
    "source_corpus_fingerprint",
    "curriculum_fingerprint",
    "split_fingerprint",
    "hf_sources_fingerprint",
})
