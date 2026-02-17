"""Graph parser: extracts graph structure from NL queries."""
import json
import re
from abc import ABC, abstractmethod

from src.nl_pipeline.data_types import NodeSpec, EdgeSpec, GraphSpec

GRAPH_EXTRACTION_PROMPT = '''You extract graph structure from questions. Output valid JSON only.

Example 1:
Question: "Why does the server crash when the database is overloaded?"
{"nodes": [{"name": "server", "type": "component"}, {"name": "database", "type": "component"}, {"name": "overload", "type": "event"}], "edges": [{"source": "database", "target": "overload", "relation": "causes"}, {"source": "overload", "target": "server", "relation": "causes"}], "query_node": "database", "target_node": "server", "domain": "technology"}

Example 2:
Question: "Is there a cycle between photosynthesis, oxygen, and respiration?"
{"nodes": [{"name": "photosynthesis", "type": "process"}, {"name": "oxygen", "type": "substance"}, {"name": "respiration", "type": "process"}], "edges": [{"source": "photosynthesis", "target": "oxygen", "relation": "causes"}, {"source": "oxygen", "target": "respiration", "relation": "enables"}, {"source": "respiration", "target": "photosynthesis", "relation": "enables"}], "query_node": "photosynthesis", "target_node": "respiration", "domain": "biology"}

Example 3:
Question: "How far is the router from the firewall in the network?"
{"nodes": [{"name": "router", "type": "component"}, {"name": "firewall", "type": "component"}, {"name": "switch", "type": "component"}], "edges": [{"source": "router", "target": "switch", "relation": "connects"}, {"source": "switch", "target": "firewall", "relation": "connects"}], "query_node": "router", "target_node": "firewall", "domain": "technology"}

Question: "{query}"
'''


def parse_graph_json(raw: str) -> GraphSpec | None:
    """Parse a JSON string (possibly surrounded by text) into a GraphSpec.

    Returns None if the JSON is invalid or missing required fields.
    """
    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            return None
        data = json.loads(match.group())
    except (json.JSONDecodeError, TypeError):
        return None

    try:
        nodes = [NodeSpec(name=n["name"], type=n["type"]) for n in data["nodes"]]
        edges = [
            EdgeSpec(source=e["source"], target=e["target"], relation=e["relation"])
            for e in data.get("edges", [])
        ]
        query_node = data["query_node"]
        domain = data["domain"]
    except (KeyError, TypeError):
        return None

    if not nodes or not query_node or not domain:
        return None

    return GraphSpec(
        nodes=nodes,
        edges=edges,
        query_node=query_node,
        target_node=data.get("target_node"),
        domain=domain,
    )


class BaseGraphParser(ABC):
    """Abstract base for graph parsers."""

    @abstractmethod
    def parse(self, query: str) -> GraphSpec: ...


class MockGraphParser(BaseGraphParser):
    """Keyword-based graph parser for CPU testing (no LLM required)."""

    _STOP_WORDS = frozenset({
        "the", "this", "that", "from", "with", "does", "how", "what", "why",
        "when", "where", "there", "have", "has", "are", "was", "were", "been",
        "being", "about", "into", "many", "much", "some", "any", "between",
        "tell", "can", "could", "would", "should", "will",
    })

    def parse(self, query: str) -> GraphSpec:
        words = re.findall(r'\b[a-z]{3,}\b', query.lower())
        keywords = [w for w in words if w not in self._STOP_WORDS]
        if len(keywords) < 2:
            keywords = ["node_a", "node_b"]

        # Deduplicate while preserving order, cap at 6 nodes
        seen: set[str] = set()
        unique: list[str] = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)
        keywords = unique[:6]

        nodes = [NodeSpec(name=kw, type="entity") for kw in keywords]

        # Build a chain of edges
        causal = any(w in query.lower() for w in ["cause", "why", "because"])
        edges = [
            EdgeSpec(
                source=nodes[i].name,
                target=nodes[i + 1].name,
                relation="causes" if causal else "connects",
            )
            for i in range(len(nodes) - 1)
        ]

        # Simple domain detection
        tech_words = {"server", "database", "network", "cache", "router", "firewall"}
        bio_words = {"cell", "protein", "gene", "enzyme", "photosynthesis"}
        domain = "general"
        if any(n.name in tech_words for n in nodes):
            domain = "technology"
        elif any(n.name in bio_words for n in nodes):
            domain = "biology"

        return GraphSpec(
            nodes=nodes,
            edges=edges,
            query_node=nodes[0].name,
            target_node=nodes[-1].name if len(nodes) > 1 else None,
            domain=domain,
        )


class GraphParser(BaseGraphParser):
    """Few-shot Llama-based graph parser. Falls back to MockGraphParser on failure."""

    def __init__(
        self,
        model_name: str = "unsloth/Llama-3.2-1B",
        device: str = "cpu",
        tokenizer=None,
        model=None,
    ):
        self._fallback = MockGraphParser()
        self._model_name = model_name
        self._device = device
        self._tokenizer = tokenizer
        self._model = model
        self._loaded = tokenizer is not None and model is not None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name, device_map=None
            ).to(self._device)
            self._model.eval()
            self._loaded = True
        except ImportError:
            self._loaded = False

    def parse(self, query: str) -> GraphSpec:
        self._ensure_loaded()
        if not self._loaded:
            return self._fallback.parse(query)

        import torch

        prompt = GRAPH_EXTRACTION_PROMPT.format(query=query)
        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        response = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        spec = parse_graph_json(response)
        if spec is not None:
            return spec
        return self._fallback.parse(query)
