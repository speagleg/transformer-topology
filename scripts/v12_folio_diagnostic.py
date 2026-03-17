#!/usr/bin/env python
"""v12 Diagnostic: Does topology predict reasoning failures?

Runs ProntoQA (logical reasoning chains) through CoT, builds reasoning
graphs, computes topology features, and correlates: do unhealthy
topology signals predict wrong answers?

This is the validation experiment for the metacognitive thesis:
if topology CAN diagnose reasoning failures, interventions have a target.

Usage:
    PYTHONPATH=. python scripts/v12_folio_diagnostic.py --num-problems 200
"""

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.metacog.reasoning_loop import MetacognitiveReasoner
from src.metacog.embedding_projection import EmbeddingProjection
from src.metacog.graph_constructor import ReasoningGraphConstructor
from src.metacog.topology_analyzer import TopologyAnalyzer
from src.metacog.health_report import decide_action, Action


def load_prontoqa(max_problems=200):
    ds = load_dataset("renma/ProntoQA", split="validation")
    problems = []
    for item in ds:
        if len(problems) >= max_problems:
            break
        gold = item["answer"].strip().upper()  # "A" (True) or "B" (False)
        problems.append({
            "id": item["id"],
            "context": item["context"],
            "question": item["question"],
            "options": item["options"],
            "gold": gold,
            "gold_label": "True" if gold == "A" else "False",
        })
    return problems


def generate_cot(model, tokenizer, problem, device, max_new_tokens=512):
    """Generate free-form CoT for a ProntoQA problem."""
    prompt = (
        f"Given the following facts and rules, determine if the statement "
        f"is true or false. Think step by step through the logical chain.\n\n"
        f"Facts and rules:\n{problem['context']}\n\n"
        f"Question: {problem['question']}\n"
        f"Options: {', '.join(problem['options'])}\n\n"
        f"Let me reason step by step:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def extract_answer(cot_text, options):
    """Extract True/False from CoT output."""
    text = cot_text.lower()
    # Look for explicit answer patterns
    if "the answer is b" in text or "the answer is false" in text or "statement is false" in text:
        return "B"
    if "the answer is a" in text or "the answer is true" in text or "statement is true" in text:
        return "A"
    # Look for last occurrence of true/false
    last_true = text.rfind("true")
    last_false = text.rfind("false")
    if last_false > last_true:
        return "B"
    if last_true > last_false:
        return "A"
    return "?"


def parse_cot_to_steps(cot_text):
    """Parse CoT into reasoning steps."""
    lines = re.split(r'\n+|(?<=\.)\s+(?=[A-Z0-9])', cot_text.strip())
    steps = []
    for line in lines:
        line = line.strip()
        if len(line) < 10:
            continue
        steps.append({"step": line, "type": "deduction", "depends_on": []})
    return steps if steps else [{"step": cot_text[:500], "type": "deduction", "depends_on": []}]


def build_graph_and_analyze(steps, model, tokenizer, projection, device, embedding_dim=128):
    """Build reasoning graph from steps and compute topology features."""
    gc = ReasoningGraphConstructor(embedding_dim=embedding_dim)
    analyzer = TopologyAnalyzer()

    for i, step in enumerate(steps):
        # Get embedding
        text = step["step"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][0, -1, :]
        embedding = projection(hidden).float().cpu()

        if embedding.shape[0] > embedding_dim:
            embedding = embedding[:embedding_dim]
        elif embedding.shape[0] < embedding_dim:
            pad = torch.zeros(embedding_dim - embedding.shape[0])
            embedding = torch.cat([embedding, pad])

        # Dependencies: each step depends on the previous 1-2
        deps = list(range(max(0, i - 2), i))
        gc.add_step(
            step_text=text,
            embedding=embedding,
            depends_on=[d + 1 for d in deps],
            step_type="deduction",
        )

    if gc.num_steps < 2:
        return None

    snapshot = gc.get_snapshot()
    report = analyzer.analyze(snapshot)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-problems", type=int, default=200)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="auto",
    )
    projection = EmbeddingProjection(
        llm_dim=model.config.hidden_size, embed_dim=128,
    ).to(device).half()

    problems = load_prontoqa(args.num_problems)
    print(f"Loaded {len(problems)} ProntoQA problems")

    # Track results
    results = []
    correct_topology = defaultdict(list)  # topology feature → values when correct
    wrong_topology = defaultdict(list)    # topology feature → values when wrong

    total_correct = 0
    total = 0

    print(f"\n{'='*72}")
    print("ProntoQA Diagnostic: Topology vs Reasoning Correctness")
    print(f"{'='*72}\n")

    for i, problem in enumerate(problems):
        t0 = time.time()

        # Generate CoT
        cot_text = generate_cot(model, tokenizer, problem, device)

        # Extract answer
        predicted = extract_answer(cot_text, problem["options"])
        is_correct = predicted == problem["gold"]
        if is_correct:
            total_correct += 1
        total += 1

        # Parse and analyze topology
        steps = parse_cot_to_steps(cot_text)
        report = build_graph_and_analyze(
            steps, model, tokenizer, projection, device,
        )

        elapsed = time.time() - t0

        result = {
            "id": problem["id"],
            "correct": is_correct,
            "predicted": predicted,
            "gold": problem["gold"],
            "num_steps": len(steps),
            "elapsed": elapsed,
        }

        if report is not None:
            result["curl"] = report.curl_energy
            result["gradient"] = report.gradient_energy
            result["harmonic"] = report.harmonic_energy
            result["spectral_gap"] = report.spectral_gap
            result["components"] = report.num_components
            result["betti_1"] = report.betti_1
            result["density"] = report.graph_density
            result["action"] = report.action.value if hasattr(report.action, 'value') else str(report.action)

            bucket = correct_topology if is_correct else wrong_topology
            bucket["curl"].append(report.curl_energy)
            bucket["gradient"].append(report.gradient_energy)
            bucket["harmonic"].append(report.harmonic_energy)
            bucket["spectral_gap"].append(report.spectral_gap)
            bucket["components"].append(report.num_components)
            bucket["betti_1"].append(report.betti_1)

        results.append(result)

        if (i + 1) % 20 == 0:
            acc = total_correct / total
            print(f"  [{i+1}/{len(problems)}] acc={acc:.1%} elapsed={elapsed:.1f}s")

    # === ANALYSIS ===
    print(f"\n{'='*72}")
    print("RESULTS")
    print(f"{'='*72}")
    print(f"  Accuracy: {total_correct}/{total} ({total_correct/total:.1%})")

    print(f"\n{'='*72}")
    print("TOPOLOGY SIGNAL CORRELATION")
    print(f"{'='*72}")
    print(f"{'Feature':<18} {'Correct (mean±std)':<22} {'Wrong (mean±std)':<22} {'Delta':>8}")
    print("-" * 72)

    for feat in ["curl", "gradient", "harmonic", "spectral_gap", "components", "betti_1"]:
        c_vals = correct_topology[feat]
        w_vals = wrong_topology[feat]
        if not c_vals or not w_vals:
            continue

        c_mean = sum(c_vals) / len(c_vals)
        w_mean = sum(w_vals) / len(w_vals)
        c_std = (sum((x - c_mean)**2 for x in c_vals) / len(c_vals)) ** 0.5
        w_std = (sum((x - w_mean)**2 for x in w_vals) / len(w_vals)) ** 0.5
        delta = w_mean - c_mean

        print(f"  {feat:<16} {c_mean:>7.4f}±{c_std:<7.4f}    {w_mean:>7.4f}±{w_std:<7.4f}    {delta:>+.4f}")

    # Predictive power: does unhealthy topology predict wrong answers?
    print(f"\n{'='*72}")
    print("INTERVENTION TARGETING")
    print(f"{'='*72}")

    intervene_correct = sum(1 for r in results if r.get("action") == "INTERVENE" and r["correct"])
    intervene_wrong = sum(1 for r in results if r.get("action") == "INTERVENE" and not r["correct"])
    continue_correct = sum(1 for r in results if r.get("action") == "CONTINUE" and r["correct"])
    continue_wrong = sum(1 for r in results if r.get("action") == "CONTINUE" and not r["correct"])

    print(f"  INTERVENE flagged: {intervene_correct + intervene_wrong} problems")
    print(f"    Correct answers flagged: {intervene_correct} (false alarms)")
    print(f"    Wrong answers flagged:   {intervene_wrong} (true catches)")
    print(f"  CONTINUE (healthy): {continue_correct + continue_wrong} problems")
    print(f"    Correct + healthy: {continue_correct}")
    print(f"    Wrong + healthy:   {continue_wrong} (missed failures)")

    if intervene_correct + intervene_wrong > 0:
        precision = intervene_wrong / (intervene_correct + intervene_wrong)
        print(f"\n  Intervention precision: {precision:.1%} (of flagged problems, how many were actually wrong)")
    if intervene_wrong + continue_wrong > 0:
        recall = intervene_wrong / (intervene_wrong + continue_wrong)
        print(f"  Intervention recall:    {recall:.1%} (of wrong answers, how many were flagged)")

    # Save
    out_path = Path("data/v12_prontoqa_diagnostic.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "accuracy": total_correct / total,
            "total": total,
            "correct": total_correct,
            "topology_correlation": {
                feat: {
                    "correct_mean": sum(correct_topology[feat]) / max(len(correct_topology[feat]), 1),
                    "wrong_mean": sum(wrong_topology[feat]) / max(len(wrong_topology[feat]), 1),
                }
                for feat in ["curl", "gradient", "harmonic", "spectral_gap", "components", "betti_1"]
            },
            "intervention_targeting": {
                "intervene_correct": intervene_correct,
                "intervene_wrong": intervene_wrong,
                "continue_correct": continue_correct,
                "continue_wrong": continue_wrong,
            },
            "per_problem": results,
        }, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
