"""Heuristic task router: maps GraphSpec + NL query to task type."""
from src.benchmarks.benchmark_dataset import get_max_classes
from src.nl_pipeline.data_types import GraphSpec, TaskRoute

_ROUTE_RULES: list[tuple[list[str], str]] = [
    (["shortest", "distance", "how far", "how many hops"], "bfs"),
    (["how many paths", "count paths", "number of paths", "count routes"], "path_counting"),
    (["cycle", "loop", "circular"], "cycle_detection"),
    (["missing", "what connects", "should there be", "predict edge"], "graph_completion"),
    (["similar", "analogy", "analogous", "correspond"], "analogical_transfer"),
    (["connected", "connectivity", "clustered", "spectral", "dense"], "spectral_gap"),
]
_CAUSAL_RELATIONS = {"causes", "prevents", "enables"}


class TaskRouter:
    """Routes a GraphSpec + NL query to a task type using heuristic rules.

    Priority order:
    1. Causal relations in edges -> labeled_reasoning
    2. Keyword matching in query -> specific task type
    3. Fallback -> diverse
    """

    def route(self, spec: GraphSpec, query: str) -> TaskRoute:
        """Classify the query and build a TaskRoute with metadata."""
        task_type = self._classify(spec, query)
        max_classes = get_max_classes(task_type)
        name_to_idx = {n.name: i for i, n in enumerate(spec.nodes)}
        query_idx = name_to_idx.get(spec.query_node, 0)
        target_idx = name_to_idx.get(spec.target_node, min(1, len(spec.nodes) - 1))
        node_labels = {i: n.name for i, n in enumerate(spec.nodes)}
        edge_labels = {f"{e.source}-{e.target}": e.relation for e in spec.edges}
        task_prompt = (
            f"nodes={len(spec.nodes)} edges={len(spec.edges)} "
            f"domain={spec.domain} | task={task_type}"
        )
        return TaskRoute(
            task_type=task_type,
            query_node_idx=query_idx,
            target_node_idx=target_idx,
            max_classes=max_classes,
            metadata={
                "task_type": task_type,
                "task_prompt": task_prompt,
                "node_labels": node_labels,
                "edge_labels": edge_labels,
                "domain": spec.domain,
            },
        )

    def _classify(self, spec: GraphSpec, query: str) -> str:
        """Determine task type from edge relations then query keywords."""
        relations = {e.relation.lower() for e in spec.edges}
        if relations & _CAUSAL_RELATIONS:
            return "labeled_reasoning"
        q_lower = query.lower()
        for keywords, task_type in _ROUTE_RULES:
            if any(kw in q_lower for kw in keywords):
                return task_type
        return "diverse"
