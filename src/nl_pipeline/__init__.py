"""Phase 5: Natural Language Reasoning Pipeline.

Usage:
    from src.nl_pipeline import NLReasoningPipeline, NLResult

    pipeline = NLReasoningPipeline(
        checkpoint_path="data/llama_checkpoints/phase4c_model.pt",
        model_config=config["model"],
        wave_config=config["wave"],
        llm_config=config["llm"],
    )
    result = pipeline.query("What is the shortest path from server to database?")
    print(result.answer)
"""

from src.nl_pipeline.data_types import (
    NodeSpec,
    EdgeSpec,
    GraphSpec,
    TaskRoute,
    NLResult,
)
from src.nl_pipeline.pipeline import NLReasoningPipeline, load_frozen_model
from src.nl_pipeline.graph_parser import MockGraphParser, GraphParser
from src.nl_pipeline.task_router import TaskRouter
from src.nl_pipeline.cell_complex_builder import CellComplexBuilder
from src.nl_pipeline.answer_generator import AnswerGenerator
