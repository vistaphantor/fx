from __future__ import annotations

from src.language.exam_types import ExamQuestion
from src.language.foundation_contract import FOUNDATION_EXAM_QUESTIONS_PER_SKILL, FOUNDATION_SKILLS


def _conceptual(
    skill: str,
    category: str,
    prompt: str,
    *,
    expected_all: tuple[str, ...] = (),
    expected_any: tuple[str, ...] = (),
    target: str,
) -> ExamQuestion:
    return ExamQuestion(
        question_id=f"{skill}_concept",
        category=category,
        prompt=prompt,
        expected_all=expected_all,
        expected_any=expected_any,
        diagnostic_target=target,
        skill=skill,
        conceptual_gate=True,
    )


def _arithmetic_exam(skill: str) -> tuple[ExamQuestion, ...]:
    gates = {
        "addition": (
            "What is addition? Explain what it means.",
            ("combine", "combining", "total", "sum", "together"),
            "Addition combines quantities to make a total or sum.",
        ),
        "subtraction": (
            "What is subtraction? Explain what it means.",
            ("remove", "taking", "difference", "left", "subtract"),
            "Subtraction removes one quantity from another or finds the difference between them.",
        ),
        "multiplication": (
            "What is multiplication? Explain what it means.",
            ("equal groups", "groups", "repeated addition", "times", "product"),
            "Multiplication combines equal groups and can be understood as repeated addition.",
        ),
    }
    prompt, keywords, target = gates[skill]
    questions = [_conceptual(skill, "primitive_arithmetic", prompt, expected_any=keywords, target=target)]
    for i in range(1, FOUNDATION_EXAM_QUESTIONS_PER_SKILL):
        if skill == "addition":
            a, b = 7 + (i * 3) % 43, 5 + (i * 5) % 31
            question, answer = (
                f"A tray has {a} counters and {b} more counters are added. How many counters are there now?",
                a + b,
            )
        elif skill == "subtraction":
            a = 30 + (i * 7) % 61
            b = 2 + (i * 5) % max(3, a - 1)
            question, answer = (
                f"A box contains {a} items. If {b} items are removed, how many remain?",
                a - b,
            )
        else:
            a, b = 2 + i % 11, 2 + (i * 3) % 11
            question, answer = (
                f"There are {a} equal groups with {b} objects in each group. How many objects are there altogether?",
                a * b,
            )
        questions.append(ExamQuestion(
            question_id=f"{skill}_{i:02d}",
            category="primitive_arithmetic",
            prompt=question,
            numeric_answer=str(answer),
            diagnostic_target=f"{answer}.",
            skill=skill,
        ))
    return tuple(questions)


# Each record is: category, conceptual prompt, gate keywords, canonical gate answer,
# then a small deterministic concept bank. The bank is cycled with a numbered
# surface to produce the 49 fixed held-out probes without bloating the contract.
_DOMAIN_BANKS: dict[str, tuple[
    str, str, tuple[str, ...], str,
    tuple[tuple[str, tuple[str, ...], str], ...],
]] = {
    "english": (
        "grammar",
        "What does it mean for an English sentence to be grammatical? Explain briefly.",
        ("grammar", "rules", "correct", "sentence"),
        "A grammatical English sentence follows the language's rules for forming a meaningful sentence.",
        (
            ("What is a noun?", ("person", "place", "thing", "name"), "A noun names a person, place, thing or idea."),
            ("What is a verb?", ("action", "state", "doing"), "A verb expresses an action or state."),
            ("What is an adjective?", ("describe", "noun"), "An adjective describes a noun."),
            ("Why does punctuation matter?", ("meaning", "clarity", "sentence"), "Punctuation helps organize sentences and make meaning clear."),
        ),
    ),
    "swahili": (
        "language_control",
        "Lugha ya Kiswahili ni nini? Eleza kwa kifupi.",
        ("kiswahili", "lugha", "mawasiliano", "kuzungumza"),
        "Kiswahili ni lugha inayotumiwa kuwasiliana kwa kuzungumza na kuandika.",
        (
            ("Nomino ni nini?", ("jina", "mtu", "kitu", "mahali"), "Nomino ni neno linalotaja mtu, kitu, mahali au wazo."),
            ("Kitenzi ni nini?", ("tendo", "hali"), "Kitenzi ni neno linaloonyesha tendo au hali."),
            ("Sentensi ni nini?", ("maneno", "maana"), "Sentensi ni mpangilio wa maneno unaotoa maana kamili."),
            ("Aya ni nini katika maandishi?", ("sentensi", "wazo"), "Aya ni kundi la sentensi zinazohusu wazo moja kuu."),
        ),
    ),
    "economics": (
        "foundation_economics",
        "What is economics? Explain what it studies.",
        ("scarcity", "resources", "choices", "wants"),
        "Economics studies how people and societies make choices when resources are scarce relative to wants.",
        (
            ("What is scarcity?", ("limited", "resources", "wants"), "Scarcity means resources are limited relative to wants."),
            ("What is demand?", ("buyers", "willing", "able", "buy"), "Demand is the quantity buyers are willing and able to buy."),
            ("What is supply?", ("sellers", "offer", "sell"), "Supply is the quantity sellers are willing and able to offer."),
            ("What is inflation?", ("prices", "rise", "purchasing"), "Inflation is a broad rise in prices over time."),
        ),
    ),
    "business": (
        "foundation_economics",
        "What is a business? Explain its basic purpose.",
        ("goods", "services", "customers", "value"),
        "A business organizes resources to provide goods or services that create value for customers.",
        (
            ("Who is a customer?", ("buys", "goods", "services"), "A customer buys or uses a business's goods or services."),
            ("What is revenue?", ("money", "sales", "before", "cost"), "Revenue is money received from sales before costs are deducted."),
            ("What is profit?", ("revenue", "cost", "left"), "Profit is what remains when costs are deducted from revenue."),
            ("Who is a supplier?", ("provides", "goods", "services"), "A supplier provides goods or services to a business."),
        ),
    ),
    "finance": (
        "foundation_economics",
        "What is finance? Explain what it deals with.",
        ("money", "funds", "capital", "manage"),
        "Finance deals with obtaining, allocating and managing money, funds and capital over time.",
        (
            ("What is saving?", ("income", "future", "spend"), "Saving means keeping part of current income for future use."),
            ("What is borrowing?", ("money", "repay"), "Borrowing means receiving money now with an obligation to repay it."),
            ("What is interest?", ("borrowing", "return", "money"), "Interest is the price of borrowing money or the return for providing funds."),
            ("What is financial risk?", ("loss", "uncertainty"), "Financial risk is uncertainty about outcomes that can cause loss."),
        ),
    ),
    "commerce": (
        "foundation_economics",
        "What is commerce? Explain what it involves.",
        ("trade", "exchange", "goods", "services"),
        "Commerce involves the trade and exchange of goods and services and the activities that support that exchange.",
        (
            ("What is trade?", ("buy", "sell", "exchange"), "Trade is the buying, selling or exchange of goods and services."),
            ("What is wholesale?", ("large", "quantities", "retail"), "Wholesale is selling goods in larger quantities, often to retailers."),
            ("What is retail?", ("consumer", "final", "sell"), "Retail is selling goods or services to final consumers."),
            ("What is an invoice?", ("bill", "payment", "goods", "services"), "An invoice is a document requesting payment for supplied goods or services."),
        ),
    ),
    "government": (
        "foundation_economics",
        "What is government? Explain its basic role in society.",
        ("public", "laws", "services", "state", "society"),
        "Government is the public authority that makes and administers laws and provides public functions for society.",
        (
            ("What is a law?", ("rule", "authority", "society"), "A law is a rule recognized and enforced by public authority."),
            ("Why does government collect taxes?", ("revenue", "public", "services"), "Taxes provide public revenue used to fund government functions and services."),
            ("What is a government budget?", ("revenue", "spending", "plan"), "A government budget is a plan for public revenue and spending."),
            ("What does a legislature do?", ("laws", "legislation"), "A legislature debates and makes laws."),
        ),
    ),
    "central_banking": (
        "foundation_economics",
        "What is a central bank? Explain its core role.",
        ("monetary", "currency", "financial", "stability", "bank"),
        "A central bank manages key monetary and currency functions and supports monetary and financial stability.",
        (
            ("What is monetary policy?", ("money", "monetary", "economy"), "Monetary policy uses central-bank tools to influence monetary and economic conditions."),
            ("What is a policy interest rate?", ("central bank", "rate", "interest"), "A policy rate is a central-bank interest rate used to influence broader financial conditions."),
            ("Why do central banks watch inflation?", ("prices", "stability", "purchasing"), "Central banks monitor inflation because persistent price instability affects purchasing power and the economy."),
            ("What is financial stability?", ("financial", "system", "shocks"), "Financial stability means the financial system can continue performing its functions through shocks."),
        ),
    ),
    "financial_news_comprehension": (
        "foundation_economics",
        "What is financial news? Explain what information it communicates.",
        ("markets", "companies", "economy", "finance", "prices"),
        "Financial news reports information about companies, markets, finance and economic conditions that can affect decisions and prices.",
        (
            ("A headline says company revenue rose. What does revenue refer to?", ("sales", "money", "before", "cost"), "Revenue is money received from sales before costs are deducted."),
            ("A headline says profit fell. What does profit mean?", ("revenue", "cost", "left"), "Profit is revenue remaining after costs are deducted."),
            ("A report says inflation accelerated. What changed?", ("prices", "faster", "rise"), "Prices are generally rising at a faster rate."),
            ("A currency depreciates. What does that mean?", ("value", "falls", "currency"), "Depreciation means the currency loses value relative to another currency or benchmark."),
        ),
    ),
    "poetry": (
        "creativity",
        "What is poetry? Explain what makes text a poem rather than ordinary factual prose.",
        ("language", "rhythm", "imagery", "verse", "expression"),
        "Poetry uses deliberately shaped language, sound, rhythm, imagery or verse to express ideas and experience.",
        (
            ("What is rhyme in poetry?", ("sound", "ending"), "Rhyme is a repetition or correspondence of sounds, often at line endings."),
            ("What is a stanza?", ("lines", "group", "poem"), "A stanza is a grouped set of lines within a poem."),
            ("What is a metaphor?", ("comparison", "describing"), "A metaphor makes an implicit comparison by describing one thing as another."),
            ("What is imagery in poetry?", ("sensory", "images"), "Imagery uses language that evokes sensory images or experiences."),
        ),
    ),
    "shairi": (
        "creativity",
        "Shairi ni nini? Eleza kwa kifupi.",
        ("ushairi", "beti", "mishororo", "vina", "mizani"),
        "Shairi ni tungo ya kishairi inayopangwa kwa mishororo na beti na inaweza kutumia vina, mizani na lugha ya picha.",
        (
            ("Ubeti ni nini katika shairi?", ("mishororo", "kundi"), "Ubeti ni kundi la mishororo katika shairi."),
            ("Mshororo ni nini katika shairi?", ("mstari", "shairi"), "Mshororo ni mstari mmoja wa shairi."),
            ("Vina ni nini katika shairi?", ("sauti", "mwisho", "mishororo"), "Vina ni ulinganifu wa sauti, mara nyingi mwishoni mwa mishororo."),
            ("Mizani ni nini katika shairi?", ("silabi", "idadi"), "Mizani huhusu mpangilio au idadi ya silabi katika mshororo."),
        ),
    ),
}


def _domain_exam(skill: str) -> tuple[ExamQuestion, ...]:
    category, gate_prompt, gate_keywords, gate_target, bank = _DOMAIN_BANKS[skill]
    questions = [
        _conceptual(skill, category, gate_prompt, expected_any=gate_keywords, target=gate_target)
    ]
    for i in range(1, FOUNDATION_EXAM_QUESTIONS_PER_SKILL):
        prompt, expected_any, target = bank[(i - 1) % len(bank)]
        questions.append(ExamQuestion(
            question_id=f"{skill}_{i:02d}",
            category=category,
            prompt=f"Held-out item {i}: {prompt}",
            expected_any=expected_any,
            diagnostic_target=target,
            skill=skill,
        ))
    return tuple(questions)


FOUNDATION_EXAM: tuple[ExamQuestion, ...] = (
    _arithmetic_exam("addition")
    + _arithmetic_exam("subtraction")
    + _arithmetic_exam("multiplication")
    + tuple(question for skill in FOUNDATION_SKILLS[3:] for question in _domain_exam(skill))
)

if len(FOUNDATION_EXAM) != len(FOUNDATION_SKILLS) * FOUNDATION_EXAM_QUESTIONS_PER_SKILL:
    raise RuntimeError("foundation_exam_question_count_contract_broken")
