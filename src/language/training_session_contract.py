from __future__ import annotations

TRAINER_VERSION = 10
TRAINING_STATE_VERSION = 10
PREFLIGHT_MANIFEST_VERSION = 10
SEED = 42
UPGRADABLE_DATA_CONTRACT_KEYS = frozenset({
    "source_corpus_fingerprint",
    "curriculum_fingerprint",
    "split_fingerprint",
    "hf_sources_fingerprint",
})
