"""Pre-train DSM on WikiText-103 as a causal language model.

Saves only transformer layer weights (discards text embedding + LM head).
These weights serve as warm initialization for the graph pipeline.

Usage:
    python scripts/pretrain_dsm.py [--epochs 5] [--batch_size 32] [--output data/dsm_pretrained.pt]
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.llm.dsm import DistilledSemanticModel


class DSMForLM(nn.Module):
    """Wrap DSM with text embedding + LM head for language modeling."""

    def __init__(self, vocab_size, dsm_dim, num_heads, ff_dim, num_layers,
                 cross_attn_layer):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dsm_dim)
        self.dsm = DistilledSemanticModel(
            hidden_dim=dsm_dim, num_heads=num_heads, ff_dim=ff_dim,
            num_layers=num_layers, cross_attn_layer=cross_attn_layer,
        )
        self.lm_head = nn.Linear(dsm_dim, vocab_size, bias=False)
        # Tie weights
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids):
        # input_ids: (batch, seq_len)
        x = self.embedding(input_ids)  # (batch, seq, dim)
        # DSM expects (seq, hidden_dim) per sample, run per-sample
        outputs = []
        for i in range(x.shape[0]):
            prefix = x[i]  # (seq, dim)
            topo_memory = torch.zeros(1, prefix.shape[-1], device=x.device)
            out = self.dsm(prefix, topo_memory)  # (seq, dim)
            outputs.append(out)
        hidden = torch.stack(outputs, dim=0)  # (batch, seq, dim)
        logits = self.lm_head(hidden)  # (batch, seq, vocab)
        return logits


def load_wikitext(seq_len=512, split="train"):
    """Load WikiText-103 using HuggingFace datasets."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

    # Concatenate all text, tokenize, chunk into seq_len blocks
    all_text = "\n\n".join(ds["text"])
    tokens = tokenizer.encode(all_text)
    tokens = torch.tensor(tokens, dtype=torch.long)

    # Chunk
    n_chunks = len(tokens) // seq_len
    tokens = tokens[:n_chunks * seq_len].view(n_chunks, seq_len)
    return tokens, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dsm_dim", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--ff_dim", type=int, default=4096)
    parser.add_argument("--num_layers", type=int, default=16)
    parser.add_argument("--cross_attn_layer", type=int, default=4)
    parser.add_argument("--output", type=str, default="data/dsm_pretrained.pt")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )

    print("Loading WikiText-103...")
    tokens, tokenizer = load_wikitext(args.seq_len)
    vocab_size = tokenizer.vocab_size
    print(f"  {len(tokens)} chunks of {args.seq_len} tokens, vocab={vocab_size}")

    loader = DataLoader(tokens, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = DSMForLM(
        vocab_size=vocab_size, dsm_dim=args.dsm_dim, num_heads=args.num_heads,
        ff_dim=args.ff_dim, num_layers=args.num_layers,
        cross_attn_layer=args.cross_attn_layer,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params/1e6:.1f}M params on {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader),
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        t0 = time.time()
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]

            logits = model(input_ids)
            loss = criterion(logits.reshape(-1, vocab_size), target_ids.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            if (batch_idx + 1) % 100 == 0:
                avg = total_loss / (batch_idx + 1)
                print(f"  Epoch {epoch+1} batch {batch_idx+1}/{len(loader)} loss={avg:.4f}")

        elapsed = time.time() - t0
        avg_loss = total_loss / len(loader)
        ppl = torch.exp(torch.tensor(avg_loss)).item()
        print(f"Epoch {epoch+1}/{args.epochs}  loss={avg_loss:.4f}  ppl={ppl:.1f}  time={elapsed:.0f}s")

    # Save only DSM transformer layers (not embedding or LM head)
    dsm_state = model.dsm.state_dict()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dsm_state, out_path)
    print(f"\nSaved DSM weights ({len(dsm_state)} tensors) to {out_path}")


if __name__ == "__main__":
    main()
