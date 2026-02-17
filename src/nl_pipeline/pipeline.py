# src/nl_pipeline/pipeline.py
"""NLReasoningPipeline: end-to-end NL -> reasoning -> NL orchestrator."""

import torch
import torch.nn.functional as F

from src.benchmarks.run_benchmark_suite import _build_model
from src.nl_pipeline.data_types import GraphSpec, NLResult
from src.nl_pipeline.graph_parser import MockGraphParser, BaseGraphParser
from src.nl_pipeline.task_router import TaskRouter
from src.nl_pipeline.cell_complex_builder import CellComplexBuilder
from src.nl_pipeline.answer_generator import AnswerGenerator


def load_frozen_model(
    checkpoint_path: str,
    model_config: dict,
    wave_config: dict | None,
    llm_config: dict | None,
    max_classes: int,
    device: torch.device,
):
    """Load a HierarchicalMultiHopModel from checkpoint with all params frozen."""
    model = _build_model(
        "hierarchical_llm",
        model_config,
        max_classes=max_classes,
        device=device,
        wave_config=wave_config,
        llm_config=llm_config,
    )
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


class NLReasoningPipeline:
    """End-to-end NL -> graph reasoning -> NL answer pipeline.

    Uses a frozen Phase 4 model for reasoning. GraphParser and AnswerGenerator
    handle NL input/output via few-shot prompting (Phase 5a) or LoRA (Phase 5b).
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_config: dict,
        wave_config: dict | None = None,
        llm_config: dict | None = None,
        max_classes: int = 16,
        device: str = "cpu",
        parser: BaseGraphParser | None = None,
    ):
        self._device = torch.device(device)

        # Pipeline components
        self.parser = parser or MockGraphParser()
        self.router = TaskRouter()
        self.builder = CellComplexBuilder(
            embedding_dim=model_config.get("embedding_dim", 32),
        )
        self.answerer = AnswerGenerator()

        # Frozen reasoning model
        self.model = load_frozen_model(
            checkpoint_path, model_config, wave_config, llm_config,
            max_classes, self._device,
        )

    def query(self, text: str) -> NLResult:
        """Run the full NL reasoning pipeline.

        Args:
            text: Free-form natural language question.

        Returns:
            NLResult with answer, class prediction, task type, graph, confidence.
        """
        # 1. Parse NL -> graph structure
        graph_spec = self.parser.parse(text)

        # 2. Route to task type
        task_route = self.router.route(graph_spec, text)

        # 3. Build CellComplex
        cc, node_map = self.builder.build(graph_spec)
        cc = cc.clone().to(self._device)

        # 4. Resolve node indices
        query_idx = node_map.get(graph_spec.query_node, 0)
        target_name = graph_spec.target_node or (
            graph_spec.nodes[1].name if len(graph_spec.nodes) > 1
            else graph_spec.nodes[0].name
        )
        target_idx = node_map.get(target_name, min(1, cc.num_cells(0) - 1))

        # 5. Run frozen model
        with torch.no_grad():
            logits = self.model(
                cc, query_idx, target_idx,
                metadata=task_route.metadata,
            )

        # 6. Extract prediction + confidence
        logits = logits.squeeze()
        probs = F.softmax(logits, dim=-1)
        class_idx = probs.argmax().item()
        confidence = probs[class_idx].item()

        # 7. Generate NL answer
        answer = self.answerer.generate(class_idx, task_route, graph_spec, text)

        return NLResult(
            answer=answer,
            class_idx=class_idx,
            task_type=task_route.task_type,
            graph_spec=graph_spec,
            confidence=confidence,
        )
