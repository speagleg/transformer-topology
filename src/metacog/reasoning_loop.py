"""MetacognitiveReasoner — the full metacognitive reasoning loop (Task 6)."""

from __future__ import annotations

import json
import re
import torch

from src.metacog.graph_constructor import ReasoningGraphConstructor
from src.metacog.topology_analyzer import TopologyAnalyzer
from src.metacog.intervention import generate_intervention
from src.metacog.problem_classifier import classify_problem
from src.metacog.graph_templates import create_template
from src.metacog.health_report import ReasoningHealthReport, Action, decide_action

# ---------------------------------------------------------------------------
# Mock step sequences keyed by step index (0-based).  Each entry is a dict
# matching the JSON schema the LLM is expected to return.
# ---------------------------------------------------------------------------
_MOCK_STEPS = [
    {
        "step": "Identify the operands: 2 and 3.",
        "depends_on": [],
        "confidence": 0.99,
        "type": "observation",
    },
    {
        "step": "Recall that addition combines two numbers.",
        "depends_on": [1],
        "confidence": 0.99,
        "type": "deduction",
    },
    {
        "step": "Compute 2 + 3 = 5.",
        "depends_on": [1, 2],
        "confidence": 0.99,
        "type": "deduction",
    },
    {
        "step": "The answer is 5.",
        "depends_on": [3],
        "confidence": 1.0,
        "type": "answer",
    },
]

_REASONING_PROMPT_TEMPLATE = """\
You are a careful, step-by-step reasoner.

Problem: {problem}

{prior_steps_section}\
{intervention_section}\
Produce the next reasoning step as a JSON object with these exact keys:
  "step"       – one-sentence natural-language reasoning step
  "depends_on" – list of 1-indexed step numbers this step relies on (may be empty)
  "confidence" – float in [0, 1]
  "type"       – one of: "observation", "deduction", "hypothesis", "answer"

If you have reached a final answer, set "type" to "answer".

Respond with valid JSON only, no markdown, no extra text.
"""


def _format_prior_steps(steps: list[dict]) -> str:
    if not steps:
        return ""
    lines = ["Prior steps:"]
    for i, s in enumerate(steps, start=1):
        lines.append(f"  {i}. [{s.get('type', '?')}] {s.get('step', '')}")
    return "\n".join(lines) + "\n\n"


def _parse_step_json(text: str) -> dict:
    """Try to extract a JSON step dict from LLM output, with fallback."""
    # Direct parse
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict) and "step" in obj:
            return obj
    except json.JSONDecodeError:
        pass

    # Regex: grab first {...} block
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict) and "step" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    # Fallback: treat raw text as the step text
    return {
        "step": text.strip(),
        "depends_on": [],
        "confidence": 0.5,
        "type": "deduction",
    }


class MetacognitiveReasoner:
    """Full metacognitive loop: classify → template → step → analyze → intervene.

    Args:
        embedding_dim: Dimension for all cell embeddings.
        max_steps: Hard cap on the number of reasoning steps.
        use_mock: If True, skip LLM and return canned mock steps (for testing).
        model: HuggingFace-style causal LM (optional).
        tokenizer: Corresponding tokenizer (optional).
        projection: nn.Linear that maps hidden_dim -> embedding_dim (optional).
        device: torch.device for LLM inference (optional).
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        max_steps: int = 15,
        use_mock: bool = False,
        model=None,
        tokenizer=None,
        projection=None,
        device=None,
    ):
        self.embedding_dim = embedding_dim
        self.max_steps = max_steps
        self.use_mock = use_mock
        self.model = model
        self.tokenizer = tokenizer
        self.projection = projection
        self.device = device or torch.device("cpu")

        self._analyzer = TopologyAnalyzer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, problem: str) -> dict:
        """Run the metacognitive reasoning loop on *problem*.

        Returns:
            dict with keys:
                answer        – string answer (last "answer"-type step, or last step)
                steps         – list of step dicts (raw parsed JSON)
                health_reports – list of ReasoningHealthReport (one per step)
                interventions – list of intervention strings generated (may be empty)
                graph         – final CellComplex snapshot
                problem_type  – string problem type
        """
        # 1. Classify
        generate_fn = None if self.use_mock else self._generate_text
        problem_type = classify_problem(problem, generate_fn=generate_fn)

        # 2. Template (informs structural expectations; not used directly in loop)
        _template = create_template(problem_type, embedding_dim=self.embedding_dim)

        # 3. Build graph incrementally
        gc = ReasoningGraphConstructor(embedding_dim=self.embedding_dim)

        steps: list[dict] = []
        health_reports: list[ReasoningHealthReport] = []
        interventions: list[str] = []
        last_action: Action | None = None
        same_action_count: int = 0

        pending_intervention: str | None = None

        for step_idx in range(self.max_steps):
            # Generate next step
            if self.use_mock:
                raw_step = self._mock_step(step_idx, steps, pending_intervention)
            else:
                raw_step = self._llm_step(problem, steps, pending_intervention)

            # Parse
            parsed = _parse_step_json(raw_step) if isinstance(raw_step, str) else raw_step

            # Add to graph constructor
            embedding = self._embed(parsed.get("step", ""), parsed)
            depends_on = parsed.get("depends_on", [])
            if not isinstance(depends_on, list):
                depends_on = []
            step_type = parsed.get("type", "deduction")
            gc.add_step(
                step_text=parsed.get("step", ""),
                embedding=embedding,
                depends_on=depends_on,
                step_type=step_type,
            )

            steps.append(parsed)

            # Analyze topology
            snapshot = gc.get_snapshot()
            report = self._analyzer.analyze(snapshot)
            health_reports.append(report)

            # Decide action
            action = decide_action(report)

            # Diminishing returns: same issue flagged 2 consecutive times → break
            if action == last_action and action != Action.CONTINUE:
                same_action_count += 1
                if same_action_count >= 2:
                    break
            else:
                same_action_count = 0
            last_action = action

            # Intervene if needed
            if action == Action.INTERVENE:
                intervention_text = generate_intervention(report)
                if intervention_text:
                    interventions.append(intervention_text)
                    pending_intervention = intervention_text
            else:
                pending_intervention = None

            # Stop if this step is an answer
            if step_type == "answer":
                break

        # 4. Assemble answer
        answer = self._extract_answer(steps)

        return {
            "answer": answer,
            "steps": steps,
            "health_reports": health_reports,
            "interventions": interventions,
            "graph": gc.get_snapshot(),
            "problem_type": problem_type,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mock_step(
        self,
        step_idx: int,
        prior_steps: list[dict],
        intervention: str | None,
    ) -> dict:
        """Return a canned mock step dict."""
        if step_idx < len(_MOCK_STEPS):
            return dict(_MOCK_STEPS[step_idx])
        # If we run past the canned sequence, produce a generic answer step
        return {
            "step": "The reasoning is complete.",
            "depends_on": [len(prior_steps)] if prior_steps else [],
            "confidence": 1.0,
            "type": "answer",
        }

    def _llm_step(
        self,
        problem: str,
        prior_steps: list[dict],
        intervention: str | None,
    ) -> str:
        """Generate a step via the LLM. Returns raw string for parsing."""
        prior_section = _format_prior_steps(prior_steps)
        intervention_section = (
            f"Correction hint: {intervention}\n\n" if intervention else ""
        )
        prompt = _REASONING_PROMPT_TEMPLATE.format(
            problem=problem,
            prior_steps_section=prior_section,
            intervention_section=intervention_section,
        )
        return self._generate_text(prompt)

    def _generate_text(self, prompt: str) -> str:
        """Run the LLM to generate a response to *prompt*."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "model and tokenizer must be provided when use_mock=False"
            )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # Decode only the newly generated tokens
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    def _embed(self, text: str, step_dict: dict) -> torch.Tensor:
        """Produce a (embedding_dim,) embedding for a step."""
        if self.use_mock or self.model is None:
            # Deterministic mock: hash text to a seed, then randn
            seed = hash(text) % (2**31)
            rng = torch.Generator()
            rng.manual_seed(seed)
            return torch.randn(self.embedding_dim, generator=rng)

        # Real: extract last hidden state of last token, project
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        # (batch, seq, hidden) → last token → (hidden,)
        hidden = outputs.hidden_states[-1][0, -1, :]
        if self.projection is not None:
            hidden = self.projection(hidden)
        # Truncate or pad to embedding_dim
        if hidden.shape[0] > self.embedding_dim:
            hidden = hidden[: self.embedding_dim]
        elif hidden.shape[0] < self.embedding_dim:
            pad = torch.zeros(self.embedding_dim - hidden.shape[0], device=hidden.device)
            hidden = torch.cat([hidden, pad])
        return hidden.float().cpu()

    @staticmethod
    def _extract_answer(steps: list[dict]) -> str:
        """Find the last 'answer'-type step, or fall back to the last step."""
        for step in reversed(steps):
            if step.get("type") == "answer":
                return step.get("step", "")
        if steps:
            return steps[-1].get("step", "")
        return ""
