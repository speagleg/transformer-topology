"""LLM-dependent benchmark tasks: graph completion, labeled reasoning, analogical transfer.

These tasks return 5-tuples (cc, src, tgt, answer, metadata) where metadata
carries text labels and task prompts that an LLM can leverage.
"""

import random

import networkx as nx
import torch

from src.benchmarks.graph_generators import random_graph
from src.benchmarks.graph_convert import nx_to_cell_complex


# Vocabulary for labeled reasoning
CAUSAL_LABELS = ["causes", "prevents", "enables"]
NODE_LABEL_POOLS = {
    "weather": ["rain", "drought", "wind", "flood", "sunshine", "frost", "hail"],
    "health": ["fever", "infection", "medicine", "recovery", "fatigue", "rest", "pain"],
    "economy": ["demand", "supply", "price", "profit", "loss", "investment", "growth"],
    "technology": ["server", "database", "network", "cache", "firewall", "router", "latency", "bandwidth"],
    "biology": ["photosynthesis", "mitosis", "enzyme", "membrane", "nucleus", "ribosome", "protein", "gene"],
    "politics": ["legislation", "veto", "coalition", "lobby", "mandate", "referendum", "treaty", "sanction"],
    "sports": ["offense", "defense", "strategy", "stamina", "injury", "recovery", "momentum", "pressure"],
    "music": ["harmony", "melody", "rhythm", "tempo", "resonance", "dissonance", "crescendo", "cadence"],
}

# Analogy domain pairs: (domain_a, domain_b, role_mapping)
ANALOGY_DOMAINS = [
    (
        {"producer": 0, "consumer": 1, "decomposer": 2, "apex": 3, "prey": 4},
        {"supplier": 0, "buyer": 1, "recycler": 2, "monopoly": 3, "commodity": 4},
    ),
    (
        {"teacher": 0, "student": 1, "principal": 2, "advisor": 3, "assistant": 4},
        {"mentor": 0, "apprentice": 1, "director": 2, "coach": 3, "intern": 4},
    ),
    (
        {"sun": 0, "planet": 1, "moon": 2, "asteroid": 3, "comet": 4},
        {"nucleus": 0, "electron": 1, "neutron": 2, "proton": 3, "photon": 4},
    ),
    (
        {"general": 0, "soldier": 1, "medic": 2, "scout": 3, "engineer": 4},
        {"ceo": 0, "employee": 1, "hr": 2, "analyst": 3, "developer": 4},
    ),
    (
        {"predator": 0, "prey": 1, "scavenger": 2, "parasite": 3, "symbiont": 4},
        {"compiler": 0, "source_code": 1, "debugger": 2, "virus": 3, "library": 4},
    ),
    (
        {"river": 0, "tributary": 1, "delta": 2, "dam": 3, "reservoir": 4},
        {"artery": 0, "capillary": 1, "vein": 2, "valve": 3, "heart": 4},
    ),
]


def generate_graph_completion_task(
    n_nodes: int,
    embedding_dim: int,
    topologies: list[str] | None = None,
) -> tuple:
    """Generate a graph completion task.

    Creates an SBM-like graph, removes 10-20% of edges, and asks whether
    a specific candidate edge should exist. Binary classification (0=no, 1=yes).

    Returns:
        (cc, src_node, tgt_node, answer, metadata)
        answer: 0 (edge doesn't exist) or 1 (edge exists in original)
        metadata: dict with 'task_type', 'task_prompt', 'removed_edges'
    """
    topo = random.choice(topologies) if topologies else 'sbm'
    G = random_graph(n_nodes, topology=topo)

    edges = list(G.edges())
    if len(edges) < 5:
        # Too few edges, use a denser graph
        G = random_graph(max(n_nodes, 10), topology='ba')
        edges = list(G.edges())

    # Remove 10-20% of edges
    num_remove = max(1, int(len(edges) * random.uniform(0.10, 0.20)))
    removed = random.sample(edges, num_remove)
    G_incomplete = G.copy()
    G_incomplete.remove_edges_from(removed)

    # Ensure graph stays connected
    if not nx.is_connected(G_incomplete):
        # Add back edges to maintain connectivity
        for e in removed:
            G_incomplete.add_edge(*e)
            if nx.is_connected(G_incomplete):
                removed = [r for r in removed if r != e]
                break

    # Pick a candidate edge: 50% from removed (positive), 50% non-edge (negative)
    nodes = list(G_incomplete.nodes())
    if random.random() < 0.5 and removed:
        # Positive: a removed edge
        u, v = random.choice(removed)
        answer = 1
    else:
        # Negative: a non-edge (in both original and incomplete)
        non_edges = list(nx.non_edges(G))
        if non_edges:
            u, v = random.choice(non_edges)
            answer = 0
        else:
            # Fallback to removed edge
            u, v = random.choice(removed) if removed else random.sample(nodes, 2)
            answer = 1

    cc, node_map = nx_to_cell_complex(
        G_incomplete, embedding_dim,
        source_node=u, target_node=v,
    )

    density = G_incomplete.number_of_edges() / max(1, n_nodes * (n_nodes - 1) / 2)
    metadata = {
        'task_type': 'graph_completion',
        'task_prompt': f'nodes={n_nodes} edges={G_incomplete.number_of_edges()} '
                       f'original_edges={len(edges)} removed={len(removed)} '
                       f'topology={topo} density={density:.3f} | task=graph_completion',
        'removed_edges': len(removed),
        'original_edges': len(edges),
    }

    return cc, node_map[u], node_map[v], answer, metadata


def generate_labeled_reasoning_task(
    n_nodes: int,
    embedding_dim: int,
    topologies: list[str] | None = None,
) -> tuple:
    """Generate a labeled graph reasoning task.

    Creates a causal graph with text labels on nodes and edges.
    Task: classify the relationship type between source and target.
    3 classes: 0=causal_chain, 1=blocked, 2=independent.

    Returns:
        (cc, src_node, tgt_node, answer, metadata)
        metadata: dict with 'task_type', 'task_prompt', 'node_labels', 'edge_labels'
    """
    topo = random.choice(topologies) if topologies else None
    G = random_graph(n_nodes, topology=topo)

    # Assign text labels to nodes and edges
    domain = random.choice(list(NODE_LABEL_POOLS.keys()))
    label_pool = NODE_LABEL_POOLS[domain]
    node_labels = {}
    for node in G.nodes():
        node_labels[node] = random.choice(label_pool)

    edge_labels = {}
    for u, v in G.edges():
        edge_labels[(u, v)] = random.choice(CAUSAL_LABELS)

    nodes = list(G.nodes())
    source, target = random.sample(nodes, 2)

    # Determine relationship type
    if nx.has_path(G, source, target):
        path = nx.shortest_path(G, source, target)
        # Check edge labels along path
        path_labels = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            label = edge_labels.get((u, v)) or edge_labels.get((v, u), "causes")
            path_labels.append(label)

        if "prevents" in path_labels:
            answer = 1  # blocked
        else:
            answer = 0  # causal chain
    else:
        answer = 2  # independent

    cc, node_map = nx_to_cell_complex(
        G, embedding_dim,
        source_node=source, target_node=target,
    )

    # Build prompt with labels
    edge_label_counts = {l: sum(1 for v in edge_labels.values() if v == l) for l in CAUSAL_LABELS}
    path_summary = ""
    if answer != 2:  # connected
        path_summary = f"path_len={len(path)} path_labels={','.join(path_labels)}"
    metadata = {
        'task_type': 'labeled_reasoning',
        'task_prompt': f'domain={domain} nodes={n_nodes} edges={G.number_of_edges()} '
                       f'causes={edge_label_counts["causes"]} '
                       f'prevents={edge_label_counts["prevents"]} '
                       f'enables={edge_label_counts["enables"]} | '
                       f'src={node_labels[source]} tgt={node_labels[target]} '
                       f'{path_summary} | task=labeled_reasoning',
        'node_labels': node_labels,
        'edge_labels': {f"{u}-{v}": l for (u, v), l in edge_labels.items()},
        'domain': domain,
    }

    return cc, node_map[source], node_map[target], answer, metadata


def generate_analogical_transfer_task(
    n_nodes: int,
    embedding_dim: int,
    topologies: list[str] | None = None,
) -> tuple:
    """Generate an analogical transfer task.

    Creates two graphs with analogous semantic structure but different topology.
    Task: given a query node in graph A, identify the analogous node in graph B.
    Uses the first graph as the context (stored in cc) with both query and
    candidate target nodes. The answer is the role index (0-4 → 5 classes).

    Returns:
        (cc, query_node, candidate_node, answer, metadata)
        answer: role index of the query node (0-4)
        metadata: dict with 'task_type', 'task_prompt', 'domain_a', 'domain_b', 'role'
    """
    # Pick a domain pair
    domain_a, domain_b = random.choice(ANALOGY_DOMAINS)
    num_roles = min(len(domain_a), 5)
    roles_a = list(domain_a.keys())[:num_roles]
    roles_b = list(domain_b.keys())[:num_roles]

    # Generate graph with assigned roles
    topo = random.choice(topologies) if topologies else None
    G = random_graph(max(n_nodes, num_roles + 2), topology=topo)
    nodes = list(G.nodes())

    # Assign roles to first num_roles nodes
    role_assignments = {}
    role_nodes = random.sample(nodes, num_roles)
    for i, node in enumerate(role_nodes):
        role_assignments[node] = i

    # Pick query and a random target
    query_role_idx = random.randint(0, num_roles - 1)
    query_node = role_nodes[query_role_idx]

    # Target is any other node (model must learn the role, not just node identity)
    other_nodes = [n for n in nodes if n != query_node]
    target_node = random.choice(other_nodes)

    answer = query_role_idx  # The role of the query node

    cc, node_map = nx_to_cell_complex(
        G, embedding_dim,
        source_node=query_node, target_node=target_node,
    )

    metadata = {
        'task_type': 'analogical_transfer',
        'task_prompt': f'domain_a={",".join(roles_a)} domain_b={",".join(roles_b)} '
                       f'nodes={G.number_of_nodes()} edges={G.number_of_edges()} '
                       f'assigned_roles={num_roles} | '
                       f'query_role={roles_a[query_role_idx]} | task=analogical_transfer',
        'domain_a': roles_a,
        'domain_b': roles_b,
        'role_index': query_role_idx,
        'role_name': roles_a[query_role_idx],
    }

    return cc, node_map[query_node], node_map[target_node], answer, metadata
