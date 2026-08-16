from __future__ import annotations

from typing import Iterator

from corpus.source import DatasetSource, SourceMetadata
from src.language.canonical_contract import CanonicalMessage, serialize_document, serialize_messages

LANGUAGE_QUALITY_SOURCE_VERSION = 2


def _chat(prompt: str, answer: str) -> str:
    return serialize_messages((
        CanonicalMessage("user", prompt),
        CanonicalMessage("assistant", answer),
    ))


class LanguageQualityContrastSource(DatasetSource):
    """Contrastive English sense, grammar, anti-loop repair and expressive variety."""

    @property
    def source_id(self) -> str:
        return f"generated:language_quality_contrast:v{LANGUAGE_QUALITY_SOURCE_VERSION}"

    def scan(self) -> SourceMetadata:
        return SourceMetadata(
            source_type="generated",
            path=self.source_id,
            estimated_docs=96,
            description="Grammar, semantic plausibility, repetition repair and paraphrase diversity",
        )

    def metadata(self) -> dict:
        return {
            "source_type": "generated",
            "source_id": self.source_id,
            "version": LANGUAGE_QUALITY_SOURCE_VERSION,
            "curriculum": "language_quality_contrast",
        }

    def stream(self) -> Iterator[str]:
        principles = (
            "A grammatical sentence must fit its subject, verb and complements together.",
            "A sentence can be grammatical yet nonsensical; fluent language also requires plausible relations between actions and objects.",
            "Repetition is useful only when meaning calls for it. Accidental loops should be replaced by one clear statement.",
            "Different wording can express the same fact. Good answers should vary naturally without changing the underlying meaning.",
            "When uncertain, a relevant partial answer is better than unrelated fluent text, but it must not invent facts merely to sound complete.",
        )
        for principle in principles:
            yield serialize_document(principle)

        contrasts = (
            ("They ate lunch in the park.", "They ate the park.", "People can eat lunch; a park is normally a place, not the object being eaten."),
            ("She drank a glass of water.", "She drank the chair.", "Water is something a person can drink; a chair is not."),
            ("He wore a warm coat.", "He wore a sandwich.", "A coat is clothing; a sandwich is food."),
            ("The children are playing outside.", "The children is playing outside.", "The plural subject 'children' agrees with 'are', not 'is'."),
            ("The dog runs quickly.", "The dog run quickly.", "With the singular subject 'dog' in simple present English, 'runs' is the agreeing form."),
            ("We were ready to leave.", "We was ready to leave.", "The plural subject 'we' takes 'were'."),
            ("The book is on the table.", "The book are on the table.", "The singular subject 'book' takes 'is'."),
            ("The farmer planted maize in the field.", "The farmer planted the field in maize.", "The ordinary relation is that the crop is planted in the field."),
            ("A mechanic repaired the car.", "A mechanic repaired the sunshine.", "A car can be repaired; sunshine is not a repairable object."),
            ("The nurse measured the patient's temperature.", "The nurse measured the patient's laughter with a thermometer.", "A thermometer measures temperature, not laughter."),
        )
        for good, bad, why in contrasts:
            yield _chat(f"Which sentence is better English and makes sense? A) {good} B) {bad}", f"A is better: {good} {why}")
            yield _chat(f"Correct this sentence if needed: {bad}", f"A clear correction is: {good}")
            yield _chat(f"Why is this sentence problematic: {bad}", why)
            yield _chat(f"Choose the sensible sentence: A) {good} B) {bad}", f"A) {good} {why}")

        loops = (
            ("A good time to get a new time to get a new time to get a new time.", "It is a good time to start something new."),
            ("The answer is three, the answer is three, the answer is three.", "The answer is three."),
            ("We can also try using a method or a method or a method or a method.", "We can also try another method."),
            ("The market fell and fell and fell and fell and fell in the same sentence.", "The market fell sharply."),
        )
        for broken, repaired in loops:
            yield _chat(f"Rewrite this without accidental repetition: {broken}", repaired)
            yield _chat(f"Is this answer stuck in a repetition loop? {broken}", f"Yes. A concise version is: {repaired}")

        facts = (
            ("The red kite is high in the sky.", (
                "A red kite floats high above the ground.",
                "High in the sky, a red kite moves with the wind.",
                "The wind carries a bright red kite across the sky.",
                "A small red kite is visible overhead.",
            )),
            ("Rain makes the road wet.", (
                "The road becomes wet when rain falls.",
                "After the rain, the road is wet.",
                "Falling rain leaves water on the road.",
                "Rainwater wets the road surface.",
            )),
            ("A shop closes at six.", (
                "The shop shuts at six o'clock.",
                "At six, the shop closes for the day.",
                "Customers must arrive before six because the shop then closes.",
                "Six o'clock is the shop's closing time.",
            )),
        )
        for fact, variants in facts:
            for index, variant in enumerate(variants, start=1):
                yield _chat(f"Express this fact naturally in a different way: {fact}", variant)
                yield _chat(f"Give wording #{index} for the same meaning without copying the sentence: {fact}", variant)

        partials = (
            ("You know a triangle has three sides but do not know its area. What can you still say?", "I can say that a triangle has three sides. Its area cannot be determined without more information."),
            ("You know revenue is 20 shillings but the cost is missing. Can you calculate profit?", "Not yet. Profit is revenue minus cost, so the missing cost is needed for the numerical profit."),
            ("You understand that inflation raises general prices but do not know the exact inflation rate. What can you conclude?", "I can conclude that prices are rising generally, but I cannot give the exact rate without the missing data."),
        )
        for prompt, answer in partials:
            yield _chat(prompt, answer)
