"""Data structures for the NL reasoning pipeline."""
from dataclasses import dataclass, field


@dataclass
class NodeSpec:
    """A node extracted from natural language."""
    name: str
    type: str


@dataclass
class EdgeSpec:
    """An edge/relation extracted from natural language."""
    source: str
    target: str
    relation: str


@dataclass
class TopologyHint:
    """Inferred graph topology from NL query."""
    topology: str | None      # ba, ws, sbm, grid, tree, ladder, caveman, er
    confidence: float         # 0.0 to 1.0
    node_roles: dict[str, str] = field(default_factory=dict)  # name → role
    properties: dict = field(default_factory=dict)


@dataclass
class GraphSpec:
    """Complete graph structure extracted from a natural language query."""
    nodes: list[NodeSpec]
    edges: list[EdgeSpec]
    query_node: str
    target_node: str | None
    domain: str
    topology_hint: str | None = None
    topology_confidence: float = 0.0
    node_roles: dict[str, str] | None = None


@dataclass
class TaskRoute:
    """Routing decision: which task type and model inputs to use."""
    task_type: str
    query_node_idx: int
    target_node_idx: int
    max_classes: int
    metadata: dict = field(default_factory=dict)


@dataclass
class NLResult:
    """Final result of the NL reasoning pipeline."""
    answer: str
    class_idx: int
    task_type: str
    graph_spec: GraphSpec
    confidence: float
