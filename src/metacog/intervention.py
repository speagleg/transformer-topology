"""Generate natural-language intervention strings from health reports."""

from src.metacog.health_report import ReasoningHealthReport, Action, decide_action


def generate_intervention(report: ReasoningHealthReport) -> str | None:
    """Generate a natural-language correction prompt from a health report.

    Returns None if no intervention is needed (action is CONTINUE).
    When multiple issues are detected, all relevant corrections are combined.

    Args:
        report: A ReasoningHealthReport from TopologyAnalyzer.

    Returns:
        A correction string, or None if reasoning is healthy.
    """
    if decide_action(report) == Action.CONTINUE:
        return None

    parts: list[str] = []

    if report.curl_energy > 0.4:
        parts.append(
            "Your reasoning appears to be cycling: steps are referencing each other "
            "in loops rather than progressing toward a conclusion. Try breaking the "
            "cycle by introducing a new premise or re-examining your assumptions."
        )

    if report.spectral_gap < 0.05:
        parts.append(
            "There is an information bottleneck in your reasoning chain. Some steps "
            "are weakly connected to the rest, creating a fragile bridge. Strengthen "
            "the connection by providing additional supporting evidence or reasoning."
        )

    if report.harmonic_energy > 0.3:
        parts.append(
            "A large portion of your reasoning is trapped in global oscillation "
            "patterns rather than flowing toward resolution. Consider grounding your "
            "argument with concrete facts or intermediate conclusions."
        )

    if report.connected_components > 1:
        parts.append(
            f"Your reasoning is fragmented into {report.connected_components} "
            "disconnected threads. Link them together by explaining how the "
            "separate lines of thought relate to each other."
        )

    if report.betti_1 > 2:
        parts.append(
            f"Your reasoning contains {report.betti_1} independent cycles, "
            "suggesting redundant or circular argument paths. Simplify by "
            "removing redundant steps or consolidating overlapping arguments."
        )

    return " ".join(parts)
