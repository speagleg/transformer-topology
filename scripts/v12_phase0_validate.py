"""Phase 0 validation gate for v12 metacognitive reasoning system.

Tests whether Qwen 2.5-3B can reliably produce structured JSON reasoning steps
with depends_on fields. Pass criteria: parse_rate >= 80%, depends_on_rate >= 60%.

Usage:
    PYTHONPATH=. python scripts/v12_phase0_validate.py --num-problems 50
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_STEP_PROMPT = """\
You are a careful, step-by-step mathematical reasoner.

Problem: {problem}

{prior_section}\
Produce the next reasoning step as a JSON object with EXACTLY these keys:
  "step"       - one-sentence natural-language reasoning step
  "depends_on" - list of 1-indexed step numbers this step relies on ([] if none)
  "confidence" - float in [0, 1]
  "type"       - one of: "observation", "deduction", "computation", "answer"

If you have reached a final answer, set "type" to "answer".

Respond with valid JSON only. No markdown fences, no extra text.
"""


def _format_prior(steps: list[dict]) -> str:
    if not steps:
        return ""
    lines = ["Prior steps:"]
    for i, s in enumerate(steps, start=1):
        lines.append(f"  {i}. [{s.get('type', '?')}] {s.get('step', '')}")
    return "\n".join(lines) + "\n\n"


# ---------------------------------------------------------------------------
# JSON parsing with fallbacks
# ---------------------------------------------------------------------------

def _parse_step_json(text: str) -> dict | None:
    """Try to extract a JSON step dict from LLM output.

    Returns dict on success, None on total failure.
    """
    text = text.strip()

    # 1. Direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "step" in obj:
            return obj
    except json.JSONDecodeError:
        pass

    # 2. Regex: first {...} block (possibly multiline)
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict) and "step" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    # 3. Fallback: treat raw text as the step text (parse failure)
    return None


def _parse_step_json_with_fallback(text: str) -> tuple[dict, bool]:
    """Parse step JSON, returning (step_dict, parse_success)."""
    result = _parse_step_json(text)
    if result is not None:
        return result, True
    # Hard fallback — record as parse failure
    return {
        "step": text.strip(),
        "depends_on": [],
        "confidence": 0.5,
        "type": "deduction",
    }, False


# ---------------------------------------------------------------------------
# depends_on validation
# ---------------------------------------------------------------------------

def _validate_depends_on(step: dict, step_index: int) -> bool:
    """Check that all depends_on refs are valid 1-indexed step numbers.

    step_index is 0-based index of this step in the steps list so far,
    meaning valid refs are 1..step_index (i.e., prior steps only).
    """
    deps = step.get("depends_on", [])
    if not isinstance(deps, list):
        return False
    if len(deps) == 0:
        return True  # empty is always valid
    for ref in deps:
        if not isinstance(ref, int):
            return False
        if ref < 1 or ref > step_index:  # step_index == number of prior steps
            return False
    return True


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
    """Load Qwen model and tokenizer. Auto-downloads from HF Hub if needed."""
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
    print("Model loaded.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Single-problem evaluation
# ---------------------------------------------------------------------------

def generate_step(
    model,
    tokenizer,
    problem: str,
    prior_steps: list[dict],
    max_new_tokens: int = 256,
) -> str:
    """Generate a single reasoning step from the model."""
    prior_section = _format_prior(prior_steps)
    prompt = _STEP_PROMPT.format(problem=problem, prior_section=prior_section)

    # Use chat template if available (Instruct models)
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


def evaluate_problem(
    model,
    tokenizer,
    problem: str,
    max_steps: int = 8,
) -> dict:
    """Run up to max_steps reasoning steps on a single problem.

    Returns:
        dict with:
            steps        - list of step dicts
            parse_results  - list of bool (True = parsed as JSON)
            dep_results    - list of bool (True = depends_on valid)
            raw_outputs    - list of raw model output strings
    """
    steps: list[dict] = []
    parse_results: list[bool] = []
    dep_results: list[bool] = []
    raw_outputs: list[str] = []

    for step_idx in range(max_steps):
        raw = generate_step(model, tokenizer, problem, steps)
        raw_outputs.append(raw)

        parsed, parse_ok = _parse_step_json_with_fallback(raw)
        parse_results.append(parse_ok)

        # Validate depends_on (only meaningful if parsed)
        dep_ok = _validate_depends_on(parsed, step_index=step_idx) if parse_ok else False
        dep_results.append(dep_ok)

        steps.append(parsed)

        # Stop at answer step
        if parsed.get("type") == "answer":
            break

    return {
        "steps": steps,
        "parse_results": parse_results,
        "dep_results": dep_results,
        "raw_outputs": raw_outputs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 0 validation: test Qwen structured JSON reasoning on GSM8K."
    )
    parser.add_argument(
        "--num-problems", type=int, default=50,
        help="Number of GSM8K problems to evaluate (default: 50).",
    )
    parser.add_argument(
        "--max-steps", type=int, default=8,
        help="Maximum reasoning steps per problem (default: 8).",
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace model name (default: Qwen/Qwen2.5-3B-Instruct).",
    )
    parser.add_argument(
        "--output", type=str, default="data/v12_phase0_results.json",
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

    print("Loading GSM8K dataset (train split)...")
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    problems = [ex["question"] for ex in dataset.select(range(args.num_problems))]
    print(f"Loaded {len(problems)} problems.")

    # Load model
    model, tokenizer = load_model(args.model)

    # Evaluate
    all_results = []
    total_parse_ok = 0
    total_parse_total = 0
    total_dep_ok = 0
    total_dep_total = 0

    start_time = time.time()

    for i, problem in enumerate(problems):
        if i % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Problem {i}/{len(problems)} | elapsed: {elapsed:.1f}s")

        result = evaluate_problem(model, tokenizer, problem, max_steps=args.max_steps)

        parse_ok_count = sum(result["parse_results"])
        dep_ok_count = sum(r for r, p in zip(result["dep_results"], result["parse_results"]) if p)
        dep_eligible = sum(result["parse_results"])

        total_parse_ok += parse_ok_count
        total_parse_total += len(result["parse_results"])
        total_dep_ok += dep_ok_count
        total_dep_total += dep_eligible

        all_results.append({
            "problem_index": i,
            "problem": problem,
            "num_steps": len(result["steps"]),
            "parse_ok_count": parse_ok_count,
            "dep_ok_count": dep_ok_count,
            "steps": result["steps"],
            "parse_results": result["parse_results"],
            "dep_results": result["dep_results"],
            "raw_outputs": result["raw_outputs"],
        })

    elapsed_total = time.time() - start_time

    # Compute rates
    parse_rate = total_parse_ok / total_parse_total if total_parse_total > 0 else 0.0
    dep_rate = total_dep_ok / total_dep_total if total_dep_total > 0 else 0.0

    passed = parse_rate >= 0.80 and dep_rate >= 0.60

    summary = {
        "model": args.model,
        "num_problems": len(problems),
        "max_steps": args.max_steps,
        "total_steps_generated": total_parse_total,
        "parse_rate": round(parse_rate, 4),
        "depends_on_rate": round(dep_rate, 4),
        "pass_criteria": {"parse_rate": 0.80, "depends_on_rate": 0.60},
        "passed": passed,
        "elapsed_seconds": round(elapsed_total, 1),
    }

    print("\n" + "=" * 60)
    print("PHASE 0 VALIDATION RESULTS")
    print("=" * 60)
    print(f"  Problems evaluated : {len(problems)}")
    print(f"  Total steps        : {total_parse_total}")
    print(f"  JSON parse rate    : {parse_rate:.1%}  (threshold: 80%)")
    print(f"  depends_on rate    : {dep_rate:.1%}  (threshold: 60%)")
    print(f"  Elapsed            : {elapsed_total:.1f}s")
    print(f"  VERDICT            : {'PASS' if passed else 'FAIL'}")
    print("=" * 60)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"summary": summary, "results": all_results}, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
