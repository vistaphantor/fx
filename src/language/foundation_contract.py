from __future__ import annotations

FOUNDATION_CONTRACT_VERSION = 2
FOUNDATION_MIN_AVAILABLE_CURRICULUM_TOKENS = 8_000_000_000
FOUNDATION_TARGET_PREDICTION_TOKENS = 8_000_000_000
FOUNDATION_EXAM_INTERVAL_SECONDS = 4 * 60 * 60
FOUNDATION_EXAM_QUESTIONS_PER_SKILL = 50
FOUNDATION_SKILL_MASTERY_THRESHOLD = 0.80

FOUNDATION_SKILLS: tuple[str, ...] = (
    "addition",
    "subtraction",
    "multiplication",
    "english",
    "swahili",
    "economics",
    "business",
    "finance",
    "commerce",
    "government",
    "central_banking",
    "financial_news_comprehension",
    "poetry",
    "shairi",
)

FOUNDATION_SKILL_SET = frozenset(FOUNDATION_SKILLS)
