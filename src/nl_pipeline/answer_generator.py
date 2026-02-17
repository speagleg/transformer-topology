# src/nl_pipeline/answer_generator.py
"""Answer generator: converts model predictions to NL explanations.

Phase 5a: Template-based generation (no LLM needed).
Phase 5b: Llama-based generation with LoRA adapter.
"""

from src.nl_pipeline.data_types import GraphSpec, TaskRoute

TOPOLOGY_DESCRIPTIONS = {
    "ba": "hub-spoke network",
    "tree": "hierarchical tree",
    "sbm": "community-structured network",
    "ws": "small-world network",
    "grid": "grid structure",
    "ladder": "dual-path ladder",
    "caveman": "densely clustered network",
    "er": "random network",
}

# Maps (task_type, class_idx) → interpretation string
CLASS_INTERPRETATIONS = {
    "bfs": lambda idx, spec: (
        f"The shortest path from {spec.query_node} to {spec.target_node} is {idx} hops."
    ),
    "dijkstra": lambda idx, spec: (
        f"The weighted shortest path from {spec.query_node} to {spec.target_node} "
        f"has cost {idx}."
    ),
    "labeled_reasoning": lambda idx, spec: {
        0: f"There is a causal relationship between {spec.query_node} and {spec.target_node}.",
        1: f"The relationship between {spec.query_node} and {spec.target_node} is blocked.",
        2: f"{spec.query_node} and {spec.target_node} are independent — no causal link found.",
    }.get(idx, f"Relationship class {idx} between {spec.query_node} and {spec.target_node}."),
    "cycle_detection": lambda idx, spec: {
        0: f"No cycle was found involving {spec.query_node}.",
        1: f"A cycle exists involving {spec.query_node}.",
    }.get(idx, f"Cycle detection result: class {idx}."),
    "graph_completion": lambda idx, spec: {
        0: f"No edge should exist between {spec.query_node} and {spec.target_node}.",
        1: f"An edge between {spec.query_node} and {spec.target_node} should exist.",
    }.get(idx, f"Edge prediction: class {idx}."),
    "path_counting": lambda idx, spec: (
        f"There are {idx} distinct paths from {spec.query_node} to {spec.target_node}."
    ),
    "analogical_transfer": lambda idx, spec: (
        f"{spec.query_node} plays role {idx} in the analogy between the domains."
    ),
    "spectral_gap": lambda idx, spec: (
        f"The graph has connectivity level {idx} out of 7 "
        f"(spectral gap bucket {idx})."
    ),
    "hodge_class": lambda idx, spec: {
        0: "The dominant signal flow is gradient (source-to-sink).",
        1: "The dominant signal flow is curl (rotational/cyclic).",
        2: "The dominant signal is harmonic (globally balanced).",
    }.get(idx, f"Hodge classification: class {idx}."),
    "diverse": lambda idx, spec: (
        f"The graph reasoning result for {spec.query_node} → {spec.target_node} "
        f"is class {idx}."
    ),
}


class AnswerGenerator:
    """Generates NL answers from model predictions.

    Phase 5a: Template-based (no LLM).
    Phase 5b: Llama generation with context.
    """

    def generate(
        self,
        class_idx: int,
        task_route: TaskRoute,
        graph_spec: GraphSpec,
        original_query: str,
    ) -> str:
        """Generate a natural language answer.

        Args:
            class_idx: Model's predicted class index.
            task_route: Routing decision with task_type.
            graph_spec: Extracted graph structure.
            original_query: The original NL question.

        Returns:
            A human-readable explanation string.
        """
        interpreter = CLASS_INTERPRETATIONS.get(task_route.task_type)
        if interpreter is not None:
            base = interpreter(class_idx, graph_spec)
        else:
            base = (
                f"The model predicted class {class_idx} for task "
                f"'{task_route.task_type}' on the query: {original_query}"
            )

        # Prefix with topology context when available
        if graph_spec.topology_hint and graph_spec.topology_confidence >= 0.3:
            desc = TOPOLOGY_DESCRIPTIONS.get(graph_spec.topology_hint, "network")
            return f"In the {desc}: {base}"
        return base
