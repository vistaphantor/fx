from __future__ import annotations

from itertools import product
from typing import Iterator

from corpus.source import DatasetSource, SourceMetadata
from src.language.canonical_contract import CanonicalMessage, serialize_document, serialize_messages

CONCEPTUAL_FOUNDATION_VERSION = 1


def _chat(question: str, answer: str) -> str:
    return serialize_messages((
        CanonicalMessage("user", question),
        CanonicalMessage("assistant", answer),
    ))


class ConceptualArithmeticSource(DatasetSource):
    """Arithmetic invariants, counterexamples, operand binding and error correction.

    PrimitiveArithmeticSource owns exhaustive small-number drills. This source is
    intentionally different: it teaches the operation-level invariants that must
    generalize across surface forms and explicitly trains correction of plausible
    wrong answers without ever supervising the wrong answer as assistant output.
    """

    @property
    def source_id(self) -> str:
        return f"generated:conceptual_arithmetic:v{CONCEPTUAL_FOUNDATION_VERSION}"

    def scan(self) -> SourceMetadata:
        return SourceMetadata(
            source_type="generated",
            path=self.source_id,
            estimated_docs=75_000,
            description="Arithmetic theory, invariants, counterexamples, operand binding and correction",
        )

    def metadata(self) -> dict:
        return {
            "source_type": "generated",
            "source_id": self.source_id,
            "version": CONCEPTUAL_FOUNDATION_VERSION,
            "curriculum": "conceptual_arithmetic",
            "teaches": [
                "cardinality", "addition_as_combination", "subtraction_as_inverse",
                "commutativity", "associativity", "identity", "multiplication_as_groups",
                "division_as_inverse", "operand_binding", "counterexample_rejection",
            ],
        }

    def stream(self) -> Iterator[str]:
        theory = (
            "Addition combines quantities into a total. The symbols name an operation, not a memorized sentence.",
            "Adding zero leaves a quantity unchanged. This is the additive identity property.",
            "For ordinary numbers, changing the order of two addends does not change their sum. This is commutativity.",
            "When only additions are involved, regrouping the same addends does not change the total. This is associativity.",
            "Subtraction can undo addition: if a + b = c, then c - b = a and c - a = b.",
            "Multiplication describes equal groups and repeated addition. Three groups of four contain twelve items.",
            "Multiplying by one leaves a number unchanged, while multiplying by zero gives zero.",
            "Division can undo multiplication when the divisor is nonzero.",
            "A valid calculation must preserve the operands from the question. Replacing an operand creates a different problem.",
            "A correct final number with a corrupted expression is not a correct derivation because the reasoning no longer answers the given problem.",
        )
        for statement in theory:
            yield serialize_document(statement)

        # Addition as cardinality and combination, with paraphrases that share the
        # invariant but not a fixed answer pattern.
        for a, b in product(range(0, 11), repeat=2):
            total = a + b
            yield _chat(
                f"There are {a} counters in one group and {b} counters in another. How many counters are there altogether?",
                f"There are {total} counters altogether because addition combines the two quantities: {a} + {b} = {total}.",
            )
            yield _chat(
                f"Explain why {a} + {b} equals {total}.",
                f"Starting with {a} units and combining {b} more gives a total of {total} units.",
            )
            yield _chat(
                f"Does swapping the addends change the total of {a} + {b}?",
                f"No. {a} + {b} = {total} and {b} + {a} = {total}. Addition is commutative for these numbers.",
            )
            wrong = total + 1 if total != 0 else 2
            yield _chat(
                f"A student says {a} + {b} = {wrong}. Is that correct? Explain briefly.",
                f"No. Combining {a} and {b} gives {total}, so {a} + {b} = {total}, not {wrong}.",
            )

        # Identity and inverse relations force a concept to survive substitutions.
        for n in range(0, 31):
            yield _chat(f"Why is {n} + 0 = {n}?", "Adding zero adds no new units, so the quantity stays unchanged.")
            yield _chat(f"What property is shown by {n} + 0 = {n}?", "The additive identity property.")

        for a, b in product(range(0, 16), repeat=2):
            total = a + b
            yield _chat(
                f"If {a} + {b} = {total}, what subtraction undoes adding {b}?",
                f"{total} - {b} = {a}.",
            )
            yield _chat(
                f"Check the claim {total} - {a} = {b} using the inverse relationship between addition and subtraction.",
                f"The claim is correct because {a} + {b} = {total}, so subtracting {a} from {total} leaves {b}.",
            )

        # Associativity: same operands, different grouping, same result.
        for a, b, c in product(range(0, 7), repeat=3):
            total = a + b + c
            yield _chat(
                f"Compare ({a} + {b}) + {c} with {a} + ({b} + {c}).",
                f"Both equal {total}. Regrouping the same addends does not change their sum.",
            )

        # Operand binding attacks the observed failure mode where the model changes
        # an operand while pretending to solve the original expression.
        for a, b, c in product(range(0, 10), repeat=3):
            total = a + b + c
            replacement = (c + 1) % 10
            if replacement == c:
                replacement = (c + 2) % 10
            altered_total = a + b + replacement
            yield _chat(
                f"Solve {a} + {b} + {c}. Do not change any operand.",
                f"The operands are {a}, {b}, and {c}. Keeping them unchanged gives {a} + {b} + {c} = {total}.",
            )
            yield _chat(
                f"Someone answered {a} + {b} + {c} by calculating {a} + {b} + {replacement} = {altered_total}. What is the error?",
                f"The last operand was changed from {c} to {replacement}. That solves a different expression. The requested expression equals {total}.",
            )

        # Equal groups and the multiplication/division inverse.
        for groups, size in product(range(0, 13), repeat=2):
            product_value = groups * size
            yield _chat(
                f"There are {groups} equal groups with {size} items in each group. How many items are there?",
                f"There are {product_value} items because {groups} groups of {size} means {groups} × {size} = {product_value}.",
            )
            if groups > 0:
                yield _chat(
                    f"If {product_value} items are split into {groups} equal groups, how many are in each group?",
                    f"{product_value} ÷ {groups} = {size}, so each group has {size} items.",
                )

        # Counterfactuals teach that changing an input changes the mathematical
        # object, which is more useful than memorizing a table of outputs.
        for a, b in product(range(0, 16), repeat=2):
            base = a + b
            changed = a + (b + 1)
            yield _chat(
                f"If {a} + {b} = {base}, what happens to the sum when the second addend increases by one?",
                f"The sum also increases by one: {a} + {b + 1} = {changed}.",
            )


class EconomicsCausalSource(DatasetSource):
    """Foundation economics through mechanisms, counterfactuals and calculations."""

    @property
    def source_id(self) -> str:
        return f"generated:economics_causal:v{CONCEPTUAL_FOUNDATION_VERSION}"

    def scan(self) -> SourceMetadata:
        return SourceMetadata(
            source_type="generated",
            path=self.source_id,
            estimated_docs=45_000,
            description="Causal and counterfactual introductory economics with arithmetic links",
        )

    def metadata(self) -> dict:
        return {
            "source_type": "generated",
            "source_id": self.source_id,
            "version": CONCEPTUAL_FOUNDATION_VERSION,
            "curriculum": "economics_causal",
        }

    def stream(self) -> Iterator[str]:
        mechanisms = (
            ("Why does scarcity force choices?", "Resources are limited, so choosing one use can mean giving up another use."),
            ("What connects a choice to opportunity cost?", "Opportunity cost is the value of the best alternative that is given up when the choice is made."),
            ("If demand rises while supply is constrained, what pressure can appear on price?", "There can be upward pressure on price because more buyers compete for limited supply."),
            ("If supply rises while demand stays unchanged, what pressure can appear on price?", "There can be downward pressure on price because more units are available for the same demand."),
            ("If general prices rise while nominal income is unchanged, what happens to purchasing power?", "Purchasing power falls because the same money buys fewer goods and services."),
            ("If a firm's revenue rises while all of its costs stay unchanged, what happens to profit?", "Profit rises by the same amount because profit equals revenue minus cost."),
            ("If a firm's costs rise while revenue is unchanged, what happens to profit?", "Profit falls because a larger cost is subtracted from the same revenue."),
            ("Why can interest be viewed as a price?", "Interest is the price paid for using borrowed money over time, or the return received for providing funds."),
            ("Why does a budget involve trade-offs?", "Income is limited, so allocating more to one category leaves less available for other spending or saving."),
        )
        for question, answer in mechanisms:
            yield _chat(question, answer)
            yield serialize_document(answer)

        for revenue in range(20, 201, 10):
            for cost in range(0, revenue + 21, 10):
                profit = revenue - cost
                state = "profit" if profit > 0 else "break-even" if profit == 0 else "loss"
                yield _chat(
                    f"Revenue is {revenue} shillings and cost is {cost} shillings. Calculate the result and say whether it is profit, break-even or loss.",
                    f"{revenue} - {cost} = {profit}. This is {state}.",
                )
                wrong = profit + 10
                yield _chat(
                    f"A learner says profit is {wrong} when revenue is {revenue} and cost is {cost}. Check the claim.",
                    f"The claim is incorrect. Profit equals revenue minus cost, so {revenue} - {cost} = {profit}.",
                )

        for income in range(20, 201, 10):
            for spending in range(0, income + 1, 20):
                saving = income - spending
                yield _chat(
                    f"Income is {income} shillings and spending is {spending}. What amount remains for saving, and why?",
                    f"{saving} shillings remain because saving here is income minus spending: {income} - {spending} = {saving}.",
                )
