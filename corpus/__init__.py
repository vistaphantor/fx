from __future__ import annotations

"""Corpus package.

Import concrete capabilities from their authoritative modules, e.g.
`corpus.source`, `corpus.streamer`, or `corpus.manager`.  The package root is
intentionally side-effect free so importing one corpus component cannot pull
unrelated indexing/registry dependencies into the language trainer.
"""
