"""Problem type classifier for v12 metacognitive reasoning."""

from typing import Callable, Optional

PROBLEM_TYPES = {"SEQUENTIAL", "CONSTRAINT", "MULTI_HOP", "EXPLORATION", "VERIFICATION"}

CLASSIFICATION_PROMPT = """Classify the following problem into exactly one of these types:
- SEQUENTIAL: steps that must be performed in order
- CONSTRAINT: satisfying conditions or restrictions
- MULTI_HOP: reasoning through a chain of linked facts
- EXPLORATION: searching or discovering in a space of possibilities
- VERIFICATION: checking or validating claims against evidence

Problem: {problem}

Respond with only the type name (e.g. SEQUENTIAL)."""


def classify_problem(problem: str, generate_fn: Optional[Callable[[str], str]] = None) -> str:
    """Classify problem type.

    Args:
        problem: Natural language description of the problem.
        generate_fn: Optional callable(prompt) -> response string for LLM-based
            classification. If None, returns SEQUENTIAL.

    Returns:
        One of the strings in PROBLEM_TYPES. Defaults to "SEQUENTIAL" when the
        LLM response cannot be parsed or generate_fn is not provided.
    """
    if generate_fn is None:
        return "SEQUENTIAL"

    prompt = CLASSIFICATION_PROMPT.format(problem=problem)
    response = generate_fn(prompt)

    # Extract the first matching problem type from the response
    upper = response.strip().upper()
    for ptype in PROBLEM_TYPES:
        if ptype in upper:
            return ptype

    return "SEQUENTIAL"
