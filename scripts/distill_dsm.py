#!/usr/bin/env python3
"""Distill Llama 3.2 1B into DSM 250M.

One-time operation requiring transformers + sentencepiece.

Loss:
    0.5 * MSE(student_hidden[layer_8], PCA(teacher_hidden[layer_8]))
    + 0.5 * KL(student_logits, teacher_logits, temperature=2.0)

Usage:
    python scripts/distill_dsm.py --teacher meta-llama/Llama-3.2-1B \
        --output data/dsm_distilled.pt --epochs 10
"""

import argparse
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_graph_description_texts(n=10000):
    """Generate graph-description texts from task prompt templates.

    These match the format used in training tasks, e.g.:
    - "Graph with 20 nodes and 35 edges, topology=ba"
    - "query_role=producer, domain_a=supply_chain, task=analogical_transfer"
    - "Find shortest path from node 3 to node 12 in a grid graph"
    """
    templates = [
        "Graph with {n} nodes and {e} edges, topology={topo}",
        "Find shortest path from node {s} to node {t} in a {topo} graph with {n} nodes",
        "Classify the edge signal: gradient, curl, or harmonic component",
        "Does edge ({s},{t}) exist in this {topo} graph?",
        "query_role={role} domain_a={d1},{d2} domain_b={d3},{d4} task=analogical_transfer",
        "Count paths of length at most {k} from node {s} to node {t}",
        "Estimate spectral gap of {topo} graph with {n} nodes",
        "Determine if node {s} can reach node {t} through causal chain, or if blocked",
        "This is a {topo} graph with {n} nodes used for {task}",
        "The graph structure is {topo} with average degree {d}",
    ]
    topos = ["ba", "ws", "grid", "tree", "ladder", "sbm", "er", "caveman"]
    roles = ["producer", "consumer", "hub"]
    tasks = [
        "bfs",
        "path_counting",
        "diverse",
        "graph_completion",
        "labeled_reasoning",
    ]

    texts = []
    for _ in range(n):
        tmpl = random.choice(templates)
        text = tmpl.format(
            n=random.randint(10, 80),
            e=random.randint(15, 200),
            topo=random.choice(topos),
            s=random.randint(0, 19),
            t=random.randint(0, 19),
            k=random.randint(1, 5),
            role=random.choice(roles),
            d1="supply",
            d2="chain",
            d3="data",
            d4="pipeline",
            task=random.choice(tasks),
            d=random.uniform(2, 6),
        )
        texts.append(text)
    return texts


def compute_pca_projection(teacher_hidden_dim, student_hidden_dim, teacher_embeddings):
    """Compute PCA projection matrix from teacher to student dim.

    Args:
        teacher_hidden_dim: Teacher's hidden dimension (e.g. 2048).
        student_hidden_dim: Student's hidden dimension (e.g. 1024).
        teacher_embeddings: (vocab_size, teacher_hidden_dim) embedding matrix.

    Returns:
        projection: (teacher_hidden_dim, student_hidden_dim) matrix.
        mean: (teacher_hidden_dim,) centering vector.
    """
    # Center the embeddings
    mean = teacher_embeddings.mean(dim=0)
    centered = teacher_embeddings - mean

    # SVD for PCA
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)

    # Take top student_hidden_dim components
    projection = Vh[:student_hidden_dim].T  # (teacher_dim, student_dim)
    return projection, mean


class DistillationWrapper(nn.Module):
    """Wraps DSM for distillation (adds a projection head for logits)."""

    def __init__(self, dsm, vocab_size, hidden_dim):
        super().__init__()
        self.dsm = dsm
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_embeds):
        """Forward pass treating input_embeds as prefix tokens.

        For distillation, we pass text embeddings directly as prefix,
        with dummy topo_memory.

        Args:
            input_embeds: (seq_len, hidden_dim) projected teacher embeddings.

        Returns:
            hidden: (seq_len, hidden_dim) DSM hidden states.
            logits: (seq_len, vocab_size) language model logits.
        """
        # Create dummy topo_memory (1 token of zeros)
        dummy_topo = torch.zeros(
            1, self.dsm.hidden_dim, device=input_embeds.device
        )
        # Use input_embeds as prefix
        hidden = self.dsm(input_embeds, dummy_topo)
        # Only take the prefix positions (skip the 1 dummy topo position)
        prefix_hidden = hidden[: input_embeds.shape[0]]
        logits = self.lm_head(prefix_hidden)
        return hidden, logits


def distill(
    teacher_name,
    output_path,
    epochs=10,
    batch_size=8,
    max_seq_len=128,
    lr=3e-4,
    temperature=2.0,
    device=None,
):
    """Run distillation from Llama teacher to DSM student."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: transformers package required for distillation.")
        print("Install with: pip install transformers sentencepiece")
        return

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Teacher: {teacher_name}")

    # Load teacher
    print("Loading teacher model...")
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name, torch_dtype=torch.bfloat16, device_map=device
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(teacher_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher_dim = teacher.config.hidden_size  # 2048 for Llama 1B
    vocab_size = teacher.config.vocab_size

    # Compute PCA projection
    print("Computing PCA projection...")
    teacher_embeds = teacher.model.embed_tokens.weight.float()
    pca_proj, pca_mean = compute_pca_projection(teacher_dim, 1024, teacher_embeds)
    pca_proj = pca_proj.to(device)
    pca_mean = pca_mean.to(device)

    # Create student
    print("Creating student DSM...")
    from src.llm.dsm import DistilledSemanticModel

    student_dsm = DistilledSemanticModel(
        hidden_dim=1024,
        num_heads=16,
        ff_dim=4096,
        num_layers=16,
        cross_attn_layer=4,
        dropout=0.1,
    )
    student = DistillationWrapper(student_dsm, vocab_size, 1024).to(device)

    # Initialize student lm_head from PCA'd teacher embeddings
    pca_embeds = (teacher_embeds.to(device) - pca_mean) @ pca_proj
    student.lm_head.weight.data = pca_embeds[:vocab_size].float()

    total_params = sum(p.numel() for p in student.parameters())
    print(f"Student params: {total_params:,}")

    # Generate training data
    print("Generating training texts...")
    graph_texts = create_graph_description_texts(10000)

    # Simple sentences for general text (no SlimPajama download needed)
    general_texts = [
        f"The {adj} {noun} {verb} {adv}."
        for adj in ["large", "small", "complex", "simple", "dense", "sparse"]
        for noun in ["graph", "network", "structure", "system", "model", "path"]
        for verb in ["connects", "separates", "links", "processes", "transforms"]
        for adv in ["efficiently", "quickly", "reliably", "correctly", "robustly"]
    ]  # 900 simple sentences

    all_texts = graph_texts + general_texts
    random.shuffle(all_texts)

    # Tokenize
    print(f"Tokenizing {len(all_texts)} texts...")

    # Optimizer
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    teacher_extract_layer = 8  # Middle layer of 16-layer Llama

    print(f"\nStarting distillation for {epochs} epochs...")
    for epoch in range(epochs):
        t0 = time.time()
        random.shuffle(all_texts)
        total_loss = 0
        n_batches = 0

        for batch_start in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[batch_start : batch_start + batch_size]

            # Tokenize batch
            tokens = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
            ).to(device)

            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_out = teacher(
                    **tokens,
                    output_hidden_states=True,
                )
                teacher_hidden = teacher_out.hidden_states[teacher_extract_layer].float()
                teacher_logits = teacher_out.logits.float()

            # Project teacher hidden to student dim via PCA
            teacher_projected = (teacher_hidden - pca_mean) @ pca_proj

            # Student forward -- process each sequence in the batch
            # The DSM expects (seq_len, hidden_dim) not (batch, seq, dim)
            batch_loss = torch.tensor(0.0, device=device)
            for seq_idx in range(len(batch_texts)):
                seq_len = tokens["attention_mask"][seq_idx].sum().item()
                # Get teacher embeddings for this sequence as student input
                input_embeds = teacher_projected[seq_idx, :seq_len]  # (seq, 1024)

                student_hidden, student_logits = student(input_embeds)

                # MSE loss on hidden states (at corresponding layer)
                # Student layer 8 ~ teacher layer 8 (both 16 layers)
                target_hidden = teacher_projected[seq_idx, :seq_len]
                mse_loss = F.mse_loss(student_hidden[:seq_len], target_hidden)

                # KL divergence on logits
                t_logits = teacher_logits[seq_idx, :seq_len] / temperature
                s_logits = student_logits[:seq_len] / temperature
                kl_loss = F.kl_div(
                    F.log_softmax(s_logits, dim=-1),
                    F.softmax(t_logits, dim=-1),
                    reduction="batchmean",
                ) * (temperature**2)

                batch_loss = batch_loss + 0.5 * mse_loss + 0.5 * kl_loss

            batch_loss = batch_loss / len(batch_texts)

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += batch_loss.item()
            n_batches += 1

        scheduler.step()
        elapsed = time.time() - t0
        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch + 1}/{epochs} | loss={avg_loss:.4f} | {elapsed:.0f}s")

    # Save only the DSM state_dict (not the wrapper)
    print(f"\nSaving DSM to {output_path}...")
    import os

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(student.dsm.state_dict(), output_path)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distill Llama 1B into DSM 250M")
    parser.add_argument(
        "--teacher",
        default="meta-llama/Llama-3.2-1B",
        help="Teacher model name/path",
    )
    parser.add_argument(
        "--output",
        default="data/dsm_distilled.pt",
        help="Output path for DSM state_dict",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=2.0)
    args = parser.parse_args()

    distill(
        teacher_name=args.teacher,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_seq_len=args.max_seq_len,
        temperature=args.temperature,
    )
