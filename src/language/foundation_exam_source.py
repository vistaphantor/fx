from __future__ import annotations

from typing import Iterator

from corpus.source import DatasetSource, SourceMetadata
from src.language.canonical_contract import CanonicalMessage, serialize_messages
from src.language.foundation_exam import FOUNDATION_EXAM

FOUNDATION_EXAM_SOURCE_VERSION = 1


def _chat(prompt: str, answer: str) -> str:
    return serialize_messages((
        CanonicalMessage("user", prompt),
        CanonicalMessage("assistant", answer),
    ))


class FoundationExamCurriculumSource(DatasetSource):
    """Own the learnable neighborhood around every reserved foundation exam row.

    The exact exam rows are intentionally emitted by this source so the normal
    GuardedSource exclusion contract can prove they are removed before gradient
    updates. Practice rows use distinct prompt families and remain trainable.
    The exam bank and its curriculum neighborhood therefore cannot drift apart.
    """

    @property
    def source_id(self) -> str:
        return f"generated:foundation_exam_curriculum:v{FOUNDATION_EXAM_SOURCE_VERSION}"

    def scan(self) -> SourceMetadata:
        return SourceMetadata(
            source_type="generated",
            path=self.source_id,
            estimated_docs=len(FOUNDATION_EXAM) * 4,
            description="Reserved 50-question skill exams plus non-heldout practice paraphrases",
        )

    def metadata(self) -> dict:
        return {
            "source_type": "generated",
            "source_id": self.source_id,
            "version": FOUNDATION_EXAM_SOURCE_VERSION,
            "reserved_exam_rows": len(FOUNDATION_EXAM),
            "practice_variants_per_row": 3,
            "holdout_contract": "exact_exam_rows_must_be_removed_by_guarded_source",
            "skills": sorted({question.skill for question in FOUNDATION_EXAM if question.skill}),
        }

    def stream(self) -> Iterator[str]:
        for question in FOUNDATION_EXAM:
            answer = question.diagnostic_target
            if not answer:
                raise RuntimeError(f"foundation_exam_row_missing_training_target:{question.question_id}")

            # This row is the reserved sample. build_training_stream supplies the
            # exam prompt families to GuardedSource, so it must never reach loss.
            yield _chat(question.prompt, answer)

            if question.conceptual_gate:
                practice_prompts = (
                    f"Teach the underlying idea without quoting a definition: {question.prompt}",
                    f"Give a beginner-friendly explanation of this idea: {question.prompt}",
                    f"Explain the same concept using different wording: {question.prompt}",
                )
            else:
                practice_prompts = (
                    f"Practice A — solve or explain this related item: {question.prompt}",
                    f"Practice B — answer this skill exercise carefully: {question.prompt}",
                    f"Practice C — work through this foundation exercise: {question.prompt}",
                )
            for prompt in practice_prompts:
                yield _chat(prompt, answer)
