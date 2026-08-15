from __future__ import annotations

from itertools import product
from typing import Iterator

from corpus.source import DatasetSource, SourceMetadata
from src.language.canonical_contract import CanonicalMessage, serialize_document, serialize_messages

FOUNDATION_SKILL_SOURCE_VERSION = 1

_NUMBER_WORDS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty",
)


def _chat(question: str, answer: str) -> str:
    return serialize_messages((
        CanonicalMessage("user", question),
        CanonicalMessage("assistant", answer),
    ))


class PrimitiveArithmeticSource(DatasetSource):
    """Deterministic, exact-by-construction arithmetic curriculum.

    Foundation deliberately stays below algebra: small integer addition and
    subtraction, three-term chains, multiplication tables, exact division,
    comparison and number-word arithmetic. The stream is generated rather than
    stored, so it needs no network connection and cannot contain wrong labels.
    """

    @property
    def source_id(self) -> str:
        return f"generated:primitive_arithmetic:v{FOUNDATION_SKILL_SOURCE_VERSION}"

    def scan(self) -> SourceMetadata:
        return SourceMetadata(
            source_type="generated",
            path=self.source_id,
            estimated_docs=120_000,
            description="Primitive integer arithmetic and number language",
        )

    def metadata(self) -> dict:
        return {
            "source_type": "generated",
            "source_id": self.source_id,
            "version": FOUNDATION_SKILL_SOURCE_VERSION,
            "curriculum": "primitive_arithmetic",
            "max_operand": 20,
            "multiplication_table_max": 12,
        }

    def stream(self) -> Iterator[str]:
        # Two-operand addition/subtraction. Multiple phrasings teach the
        # operation rather than one fixed surface form.
        add_templates = (
            "Calculate {a} + {b}.",
            "What is {a} plus {b}?",
            "Add {a} and {b}.",
            "Find the sum of {a} and {b}.",
        )
        sub_templates = (
            "Calculate {a} - {b}.",
            "What is {a} minus {b}?",
            "Subtract {b} from {a}.",
            "Find the difference when {b} is taken from {a}.",
        )
        for a, b in product(range(21), repeat=2):
            for template in add_templates:
                question = template.format(a=a, b=b)
                yield _chat(question, f"{a} + {b} = {a + b}.")
            for template in sub_templates:
                question = template.format(a=a, b=b)
                yield _chat(question, f"{a} - {b} = {a - b}.")

        # Three-term chains are still arithmetic, but require keeping an
        # intermediate value. This directly covers tasks such as 2 + 2 + 3.
        for a, b, c in product(range(11), repeat=3):
            yield _chat(
                f"Calculate {a} + {b} + {c}.",
                f"{a} + {b} + {c} = {a + b + c}.",
            )
            yield _chat(
                f"Calculate {a} + {b} - {c}.",
                f"{a} + {b} - {c} = {a + b - c}.",
            )

        # Multiplication tables and exact integer division only. Fractions,
        # probability and algebra belong to later curricula.
        for a, b in product(range(13), repeat=2):
            yield _chat(
                f"What is {a} times {b}?",
                f"{a} times {b} is {a * b}.",
            )
        for divisor in range(1, 13):
            for quotient in range(13):
                dividend = divisor * quotient
                yield _chat(
                    f"What is {dividend} divided by {divisor}?",
                    f"{dividend} divided by {divisor} is {quotient}.",
                )

        # Comparison and number words bind symbols to ordinary English.
        for a, b in product(range(21), repeat=2):
            if a == b:
                answer = f"{a} and {b} are equal."
            elif a > b:
                answer = f"{a} is greater than {b}."
            else:
                answer = f"{b} is greater than {a}."
            yield _chat(f"Which number is greater, {a} or {b}?", answer)

            yield _chat(
                f"What is {_NUMBER_WORDS[a]} plus {_NUMBER_WORDS[b]}?",
                f"{_NUMBER_WORDS[a]} plus {_NUMBER_WORDS[b]} is {a + b}.",
            )


_ECONOMICS_CONCEPTS: tuple[tuple[str, str], ...] = (
    ("scarcity", "Scarcity means resources are limited while wants can be greater than the resources available."),
    ("choice", "A choice is a decision between alternatives when a person cannot have every option at once."),
    ("opportunity cost", "Opportunity cost is the value of the best alternative given up when a choice is made."),
    ("price", "A price is the amount of money asked or paid for a good or service."),
    ("market", "A market is a setting where buyers and sellers exchange goods, services or assets."),
    ("demand", "Demand is the quantity buyers are willing and able to buy at different prices."),
    ("supply", "Supply is the quantity sellers are willing and able to offer at different prices."),
    ("income", "Income is money received by a person or business over a period of time."),
    ("expense", "An expense is a cost paid to obtain or use goods and services."),
    ("revenue", "Revenue is the money a business receives from selling goods or services before costs are deducted."),
    ("profit", "Profit is revenue left after the costs of producing and selling are deducted."),
    ("loss", "A business makes a loss when its costs are greater than its revenue."),
    ("saving", "Saving means keeping part of current income for future use instead of spending it now."),
    ("borrowing", "Borrowing means receiving money now with an obligation to repay it later."),
    ("interest", "Interest is the price paid for borrowing money or the return earned for lending or saving money."),
    ("inflation", "Inflation is a broad rise in prices over time that reduces what a unit of money can buy."),
    ("wage", "A wage is payment received for work, commonly linked to hours or units of labour."),
    ("budget", "A budget is a plan for expected income, spending and saving over a period of time."),
    ("competition", "Competition occurs when sellers try to attract the same buyers or buyers compete for limited supply."),
    ("consumer", "A consumer is a person or household that buys or uses goods and services."),
)


class FoundationEconomicsSource(DatasetSource):
    """Introductory economics vocabulary and one-step causal relationships."""

    @property
    def source_id(self) -> str:
        return f"generated:foundation_economics:v{FOUNDATION_SKILL_SOURCE_VERSION}"

    def scan(self) -> SourceMetadata:
        return SourceMetadata(
            source_type="generated",
            path=self.source_id,
            estimated_docs=20_000,
            description="Introductory economics language and causal examples",
        )

    def metadata(self) -> dict:
        return {
            "source_type": "generated",
            "source_id": self.source_id,
            "version": FOUNDATION_SKILL_SOURCE_VERSION,
            "curriculum": "foundation_economics",
            "concepts": [name for name, _ in _ECONOMICS_CONCEPTS],
        }

    def stream(self) -> Iterator[str]:
        # Definitions appear both as ordinary prose and as question/answer turns
        # so economics vocabulary strengthens English LM and assistant behavior.
        for name, definition in _ECONOMICS_CONCEPTS:
            yield serialize_document(definition)
            yield _chat(f"What does {name} mean in basic economics?", definition)
            yield _chat(f"Explain {name} in simple words.", definition)

        # Simple quantitative economics remains arithmetic, not algebra.
        # Varying values creates thousands of distinct, exact examples.
        for revenue in range(20, 201, 10):
            for cost in range(10, revenue + 1, 10):
                profit = revenue - cost
                if profit > 0:
                    result = f"Revenue is {revenue} and cost is {cost}, so profit is {profit}."
                else:
                    result = f"Revenue and cost are both {revenue}, so profit is zero."
                yield _chat(
                    f"A small business receives {revenue} shillings and has costs of {cost} shillings. What is its profit?",
                    result,
                )

        for income in range(20, 201, 10):
            for spending in range(0, income + 1, 10):
                saving = income - spending
                yield _chat(
                    f"A person receives {income} shillings and spends {spending} shillings. How much is left to save?",
                    f"{income} - {spending} = {saving}, so {saving} shillings are left to save.",
                )

        # One-step causal language only; no elasticity, optimization or models.
        causal_examples = (
            ("If many buyers want the same limited item, what can happen to its price?", "With demand high and supply limited, the price can rise."),
            ("If sellers bring much more of a product to market while demand stays the same, what can happen to price?", "With more supply and unchanged demand, the price can fall."),
            ("If general prices rise while income stays unchanged, what happens to purchasing power?", "Purchasing power falls because the same money buys fewer goods and services."),
            ("Why does a person make a budget?", "A budget helps plan income, spending and saving before money is used."),
            ("Why can saving be useful?", "Saving keeps some income available for future needs or goals."),
            ("Why does a lender charge interest?", "Interest is compensation for providing money now and taking the risk of repayment later."),
        )
        for question, answer in causal_examples:
            yield _chat(question, answer)
