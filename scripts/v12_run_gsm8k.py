"""Full GSM8K evaluation harness for v12 metacognitive reasoning system.

Compares raw, chain-of-thought, and metacognitive reasoning modes.

Usage:
    PYTHONPATH=. python scripts/v12_run_gsm8k.py --mode all --num-problems 100
    PYTHONPATH=. python scripts/v12_run_gsm8k.py --mode metacog --num-problems 50
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def _extract_numeric_answer(text: str) -> str | None:
    """Extract a numeric answer from model output.

    Tries several patterns in priority order:
      1. "#### <number>" (GSM8K answer format)
      2. "answer is <number>" / "answer: <number>"
      3. Last standalone number in the text
    """
    if not text:
        return None

    # 1. GSM8K canonical: "#### 42" or "#### -3.5"
    m = re.search(r"####\s*([-+]?\d[\d,]*\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "")

    # 2. "the answer is X" / "answer is X" / "answer: X"
    m = re.search(
        r"(?:the\s+)?answer\s*(?:is|:)\s*([-+]?\d[\d,]*\.?\d*)",
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).replace(",", "")

    # 3. Last number in text
    numbers = re.findall(r"[-+]?\d[\d,]*\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def _answers_match(predicted: str | None, ground_truth: str) -> bool:
    """Return True if predicted numeric answer matches ground truth."""
    if predicted is None:
        return False
    try:
        pred_val = float(predicted.replace(",", ""))
        true_val = float(ground_truth.replace(",", ""))
        return abs(pred_val - true_val) < 1e-3
    except ValueError:
        return predicted.strip() == ground_truth.strip()


def _extract_gsm8k_answer(solution: str) -> str:
    """Extract ground-truth answer from GSM8K solution string (after ####)."""
    m = re.search(r"####\s*([-+]?\d[\d,]*\.?\d*)", solution)
    if m:
        return m.group(1).replace(",", "")
    # Fallback: last number
    numbers = re.findall(r"[-+]?\d[\d,]*\.?\d*", solution)
    return numbers[-1].replace(",", "") if numbers else ""


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
    """Load Qwen model and tokenizer from HuggingFace Hub."""
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"Loading model: {model_name} (float16, device_map=auto)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.\n")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _generate(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate a response from the model for the given prompt."""
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Evaluation modes
# ---------------------------------------------------------------------------

_RAW_PROMPT = """\
Solve this math problem. Give only the final numeric answer.

Problem: {problem}

Answer:"""

_COT_PROMPT = """\
Solve this math problem step by step. Show your work clearly.
At the end, write your final answer on a line starting with "#### ".

Problem: {problem}

Solution:"""


def eval_raw(model, tokenizer, problem: str) -> dict:
    """Raw mode: single prompt, numeric answer only."""
    t0 = time.time()
    prompt = _RAW_PROMPT.format(problem=problem)
    output = _generate(model, tokenizer, prompt, max_new_tokens=64)
    elapsed = time.time() - t0

    predicted = _extract_numeric_answer(output)
    return {
        "mode": "raw",
        "output": output,
        "predicted_answer": predicted,
        "num_steps": 1,
        "num_interventions": 0,
        "elapsed": elapsed,
    }


def eval_cot(model, tokenizer, problem: str) -> dict:
    """Chain-of-thought mode: step-by-step reasoning."""
    t0 = time.time()
    prompt = _COT_PROMPT.format(problem=problem)
    output = _generate(model, tokenizer, prompt, max_new_tokens=512)
    elapsed = time.time() - t0

    predicted = _extract_numeric_answer(output)
    # Rough step count: count numbered lines
    steps = re.findall(r"^\s*\d+[\.\)]\s+", output, re.MULTILINE)
    num_steps = max(len(steps), 1)

    return {
        "mode": "cot",
        "output": output,
        "predicted_answer": predicted,
        "num_steps": num_steps,
        "num_interventions": 0,
        "elapsed": elapsed,
    }


def eval_metacog(model, tokenizer, problem: str, embedding_dim: int = 128) -> dict:
    """Metacognitive mode: use MetacognitiveReasoner with EmbeddingProjection."""
    from src.metacog.reasoning_loop import MetacognitiveReasoner  # type: ignore
    from src.metacog.embedding_projection import EmbeddingProjection  # type: ignore

    # Qwen2.5-3B hidden dim is 2048
    hidden_dim = model.config.hidden_size if hasattr(model, "config") else 2048
    projection = EmbeddingProjection(llm_dim=hidden_dim, embed_dim=embedding_dim)

    device = next(model.parameters()).device
    projection = projection.to(device).half()  # match model dtype

    reasoner = MetacognitiveReasoner(
        embedding_dim=embedding_dim,
        max_steps=8,
        use_mock=False,
        model=model,
        tokenizer=tokenizer,
        projection=projection,
        device=device,
    )

    t0 = time.time()
    result = reasoner.solve_hybrid(problem)
    elapsed = time.time() - t0

    # Extract numeric answer from the answer string
    predicted = _extract_numeric_answer(result["answer"])
    # Also check all steps for a numeric answer if top-level failed
    if predicted is None:
        for step in reversed(result["steps"]):
            candidate = _extract_numeric_answer(step.get("step", ""))
            if candidate is not None:
                predicted = candidate
                break

    return {
        "mode": "metacog",
        "output": result["answer"],
        "predicted_answer": predicted,
        "num_steps": len(result["steps"]),
        "num_interventions": len(result["interventions"]),
        "problem_type": result.get("problem_type", "unknown"),
        "elapsed": elapsed,
    }


# ---------------------------------------------------------------------------
# Run one mode over all problems
# ---------------------------------------------------------------------------

def run_mode(
    mode: str,
    model,
    tokenizer,
    problems: list[dict[str, str]],
) -> dict[str, Any]:
    """Evaluate all problems in the given mode.

    Args:
        mode: "raw", "cot", or "metacog"
        model: loaded HuggingFace model
        tokenizer: corresponding tokenizer
        problems: list of dicts with "question" and "ground_truth" keys

    Returns:
        dict with per-problem results and aggregate statistics
    """
    eval_fn = {"raw": eval_raw, "cot": eval_cot, "metacog": eval_metacog}[mode]

    per_problem = []
    correct = 0
    total_steps = 0
    total_interventions = 0
    total_elapsed = 0.0

    print(f"\n{'=' * 60}")
    print(f"MODE: {mode.upper()}  |  {len(problems)} problems")
    print("=" * 60)

    for i, ex in enumerate(problems):
        if i % 10 == 0:
            print(f"  [{mode}] Problem {i}/{len(problems)}")

        question = ex["question"]
        ground_truth = ex["ground_truth"]

        result = eval_fn(model, tokenizer, question)

        is_correct = _answers_match(result["predicted_answer"], ground_truth)
        if is_correct:
            correct += 1
        total_steps += result["num_steps"]
        total_interventions += result["num_interventions"]
        total_elapsed += result["elapsed"]

        per_problem.append({
            "index": i,
            "question": question,
            "ground_truth": ground_truth,
            "predicted_answer": result["predicted_answer"],
            "correct": is_correct,
            **result,
        })

    n = len(problems)
    accuracy = correct / n if n > 0 else 0.0
    avg_steps = total_steps / n if n > 0 else 0.0
    avg_interventions = total_interventions / n if n > 0 else 0.0
    avg_elapsed = total_elapsed / n if n > 0 else 0.0

    print(f"\n  [{mode}] Accuracy    : {accuracy:.1%}  ({correct}/{n})")
    print(f"  [{mode}] Avg steps   : {avg_steps:.1f}")
    print(f"  [{mode}] Avg interv. : {avg_interventions:.1f}")
    print(f"  [{mode}] Avg time/q  : {avg_elapsed:.1f}s")

    return {
        "mode": mode,
        "num_problems": n,
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "avg_steps": round(avg_steps, 2),
        "avg_interventions": round(avg_interventions, 2),
        "avg_elapsed_seconds": round(avg_elapsed, 2),
        "total_elapsed_seconds": round(total_elapsed, 1),
        "per_problem": per_problem,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GSM8K evaluation: raw vs CoT vs metacognitive reasoning."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["raw", "cot", "metacog", "all"],
        help="Evaluation mode (default: all).",
    )
    parser.add_argument(
        "--num-problems", type=int, default=100,
        help="Number of GSM8K test problems to evaluate (default: 100).",
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace model name (default: Qwen/Qwen2.5-3B-Instruct).",
    )
    parser.add_argument(
        "--output", type=str, default="data/v12_gsm8k_results.json",
        help="Output path for results JSON.",
    )
    args = parser.parse_args()

    # Load dataset
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required. Install with: pip install datasets"
        )

    print("Loading GSM8K dataset (test split)...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    raw_problems = dataset.select(range(min(args.num_problems, len(dataset))))

    problems = [
        {
            "question": ex["question"],
            "ground_truth": _extract_gsm8k_answer(ex["answer"]),
        }
        for ex in raw_problems
    ]
    print(f"Loaded {len(problems)} problems from GSM8K test split.\n")

    # Load model
    model, tokenizer = load_model(args.model)

    # Run selected modes
    modes_to_run = ["raw", "cot", "metacog"] if args.mode == "all" else [args.mode]
    all_mode_results = []

    for mode in modes_to_run:
        mode_result = run_mode(mode, model, tokenizer, problems)
        all_mode_results.append(mode_result)

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    header = f"{'Mode':<12} {'Accuracy':>10} {'Avg Steps':>10} {'Avg Interv':>11} {'Avg Time':>10}"
    print(header)
    print("-" * 60)
    for r in all_mode_results:
        print(
            f"{r['mode']:<12} "
            f"{r['accuracy']:>9.1%} "
            f"{r['avg_steps']:>10.1f} "
            f"{r['avg_interventions']:>11.1f} "
            f"{r['avg_elapsed_seconds']:>9.1f}s"
        )
    print("=" * 60)

    # Save results
    output = {
        "model": args.model,
        "num_problems": len(problems),
        "modes_evaluated": modes_to_run,
        "results": {r["mode"]: r for r in all_mode_results},
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
