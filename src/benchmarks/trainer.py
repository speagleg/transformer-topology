import torch
import torch.nn as nn
from src.benchmarks.model import MultiHopReasoningModel
from src.benchmarks.multi_hop import MultiHopDataset


def train_epoch(model: MultiHopReasoningModel, dataset: MultiHopDataset,
                optimizer: torch.optim.Optimizer) -> float:
    model.train()
    total_loss = 0.0
    for i in range(len(dataset)):
        cc, query, target, answer = dataset[i]
        optimizer.zero_grad()
        logits = model(cc, query, target)
        loss = nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([answer]))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataset)


@torch.no_grad()
def evaluate(model: MultiHopReasoningModel, dataset: MultiHopDataset) -> tuple[float, float]:
    model.eval()
    correct = 0
    total_loss = 0.0
    for i in range(len(dataset)):
        cc, query, target, answer = dataset[i]
        logits = model(cc, query, target)
        loss = nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([answer]))
        total_loss += loss.item()
        pred = logits.argmax().item()
        if pred == answer:
            correct += 1
    accuracy = correct / len(dataset)
    avg_loss = total_loss / len(dataset)
    return accuracy, avg_loss
