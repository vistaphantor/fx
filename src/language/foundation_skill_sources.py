from __future__ import annotations

from itertools import product
from typing import Iterator

from corpus.source import DatasetSource, SourceMetadata
from src.language.canonical_contract import CanonicalMessage, serialize_document, serialize_messages

FOUNDATION_SKILL_SOURCE_VERSION = 3

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


def _answer_forms(value: int, expression: str) -> tuple[str, ...]:
    """Make the mathematical result the dominant first-token decision.

    A tiny foundation model must first bind the operands and operation to the
    result. Equation rendering remains in the curriculum, but it is deliberately
    the minority form so copying an expression cannot become a shortcut for
    answering the question.
    """
    return (
        f"{value}.",
        f"The answer is {value}.",
        f"{value} is the result.",
        f"{value}.",
        f"{expression} = {value}.",
    )


class PrimitiveArithmeticSource(DatasetSource):
    """Deterministic exact primitive arithmetic, deliberately below algebra."""

    @property
    def source_id(self) -> str:
        return f"generated:primitive_arithmetic:v{FOUNDATION_SKILL_SOURCE_VERSION}"

    def scan(self) -> SourceMetadata:
        return SourceMetadata(
            source_type="generated",
            path=self.source_id,
            estimated_docs=190_000,
            description="Primitive integer arithmetic, number language and result-first answers",
        )

    def metadata(self) -> dict:
        return {
            "source_type": "generated",
            "source_id": self.source_id,
            "version": FOUNDATION_SKILL_SOURCE_VERSION,
            "curriculum": "primitive_arithmetic",
            "max_operand": 20,
            "multiplication_table_max": 12,
            "answer_contract": "result_first_with_minor_equation_rendering",
        }

    def stream(self) -> Iterator[str]:
        add_templates = (
            "What is {a} + {b}?",
            "Calculate {a} + {b}.",
            "What is {a} plus {b}?",
            "Add {a} and {b}.",
            "Find the sum of {a} and {b}.",
        )
        sub_templates = (
            "What is {a} - {b}?",
            "Calculate {a} - {b}.",
            "What is {a} minus {b}?",
            "Subtract {b} from {a}.",
            "Find the difference when {b} is taken from {a}.",
        )
        for a, b in product(range(21), repeat=2):
            for index, template in enumerate(add_templates):
                q = template.format(a=a, b=b)
                forms = _answer_forms(a + b, f"{a} + {b}")
                yield _chat(q, forms[index % len(forms)])
            for index, template in enumerate(sub_templates):
                q = template.format(a=a, b=b)
                forms = _answer_forms(a - b, f"{a} - {b}")
                yield _chat(q, forms[index % len(forms)])

        # Chains require operand retention. Both common prompt surfaces answer with
        # the result first; equation production is trained under an explicit
        # equation-rendering instruction rather than leaking into every answer.
        for a, b, c in product(range(11), repeat=3):
            expression = f"{a} + {b} + {c}"
            value = a + b + c
            yield _chat(f"Calculate {expression}.", f"{value}.")
            yield _chat(f"What is {expression}?", f"{value}.")
            yield _chat(f"Write the completed equation for {expression}.", f"{expression} = {value}.")

            expression = f"{a} + {b} - {c}"
            value = a + b - c
            yield _chat(f"Calculate {expression}.", f"{value}.")
            yield _chat(f"What is {expression}?", f"{value}.")
            yield _chat(f"Write the completed equation for {expression}.", f"{expression} = {value}.")

        for a, b in product(range(13), repeat=2):
            value = a * b
            yield _chat(f"What is {a} times {b}?", f"{value}.")
            yield _chat(f"Calculate {a} multiplied by {b}.", f"{value}.")
            yield _chat(f"Write the multiplication equation for {a} and {b}.", f"{a} times {b} is {value}.")

        for divisor in range(1, 13):
            for quotient in range(13):
                dividend = divisor * quotient
                yield _chat(f"What is {dividend} divided by {divisor}?", f"{quotient}.")
                yield _chat(f"Calculate {dividend} divided by {divisor}.", f"{quotient}.")
                yield _chat(
                    f"Write the division equation for {dividend} divided by {divisor}.",
                    f"{dividend} divided by {divisor} is {quotient}.",
                )

        # Number sense: ordering, successor/predecessor, inverse relationships,
        # and decompositions form the substrate needed before algebra.
        for n in range(1, 21):
            yield _chat(f"What number comes after {n - 1}?", f"{n}.")
            yield _chat(f"What number comes before {n}?", f"{n - 1}.")
            yield _chat(f"Give the successor of {n - 1}.", f"{n}.")
            yield _chat(f"Give the predecessor of {n}.", f"{n - 1}.")

        for a, b in product(range(11), repeat=2):
            total = a + b
            yield _chat(f"If {a} + {b} = {total}, what is {total} - {a}?", f"{b}.")
            yield _chat(f"If {a} + {b} = {total}, what is {total} - {b}?", f"{a}.")

        for n in range(2, 21):
            left = n // 2
            right = n - left
            yield _chat(f"Split {n} into two numbers that add to {n}.", f"{left} + {right} = {n}.")

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
                f"{a + b}.",
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
            estimated_docs=45_000,
            description="Introductory economics language, concise definitions and causal examples",
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
        for name, definition in _ECONOMICS_CONCEPTS:
            yield serialize_document(definition)
            yield _chat(f"What does {name} mean in basic economics?", definition)
            yield _chat(f"Explain {name} in simple words.", definition)
            yield _chat(f"In simple economics, define {name}.", definition)
            # Short discriminative answer prevents every definition prompt from
            # collapsing into one generic long sentence.
            yield _chat(f"Name this concept: {definition}", f"{name}.")

        # Use dense integer costs rather than only multiples of ten. This teaches
        # the revenue-minus-cost relationship itself instead of a decimal-grid
        # shortcut while exact exam prompts remain excluded by prompt family.
        for revenue in range(10, 61):
            for cost in range(0, revenue + 1):
                profit = revenue - cost
                yield _chat(
                    f"A business receives {revenue} shillings and has costs of {cost} shillings. What is its profit?",
                    f"{profit} shillings.",
                )
                yield _chat(
                    f"Revenue is {revenue} and cost is {cost}. Calculate profit.",
                    f"{profit}.",
                )
                yield _chat(
                    f"What remains after a business earns {revenue} shillings and pays {cost} shillings in costs?",
                    f"{profit} shillings remain as profit because profit is revenue minus cost.",
                )

        for income in range(20, 201, 10):
            for spending in range(0, income + 1, 10):
                saving = income - spending
                yield _chat(
                    f"A person receives {income} shillings and spends {spending} shillings. How much is left to save?",
                    f"{saving} shillings.",
                )

        causal_examples = (
            ("If many buyers want the same limited item, what can happen to its price?", "The price can rise."),
            ("If sellers bring much more of a product to market while demand stays the same, what can happen to price?", "The price can fall."),
            ("If general prices rise while income stays unchanged, what happens to purchasing power?", "Purchasing power falls."),
            ("When prices rise but a person's income does not, can the same income buy as much?", "No. Purchasing power falls because the same income buys less."),
            ("Why does a person make a budget?", "A budget helps plan income, spending and saving."),
            ("Why can saving be useful?", "Saving keeps income available for future needs or goals."),
            ("Why does a lender charge interest?", "Interest compensates the lender for providing money and taking repayment risk."),
        )
        for question, answer in causal_examples:
            yield _chat(question, answer)
