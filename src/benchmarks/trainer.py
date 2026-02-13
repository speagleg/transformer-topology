import torch
import torch.nn as nn
from src.benchmarks.model import MultiHopReasoningModel
from src.benchmarks.multi_hop import MultiHopDataset


def train_epoch(model: MultiHopReasoningModel, dataset: MultiHopDataset,
                optimizer: torch.optim.Optimizer, max_norm: float = 5.0,
                accumulation_steps: int = 4, label_smoothing: float = 0.0,
                device: torch.device | str | None = None) -> tuple[float, float]:
    if device is None:
        device = torch.device('cpu')
    model.train()
    if hasattr(dataset, 'shuffle'):
        dataset.shuffle()
    total_loss = 0.0
    grad_norms = []
    optimizer.zero_grad()
    for i in range(len(dataset)):
        cc, query, target, answer = dataset[i]
        cc = cc.clone().to(device)
        logits = model(cc, query, target)
        loss = nn.functional.cross_entropy(logits.unsqueeze(0),
                                           torch.tensor([answer], device=device),
                                           label_smoothing=label_smoothing)
        loss = loss / accumulation_steps
        loss.backward()
        total_loss += loss.item() * accumulation_steps
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataset):
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            grad_norms.append(gn.item() if isinstance(gn, torch.Tensor) else gn)
            optimizer.step()
            optimizer.zero_grad()
    avg_loss = total_loss / len(dataset)
    avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
    return avg_loss, avg_grad_norm


@torch.no_grad()
def evaluate(model: MultiHopReasoningModel, dataset: MultiHopDataset,
             label_smoothing: float = 0.0,
             device: torch.device | str | None = None) -> tuple[float, float]:
    if device is None:
        device = torch.device('cpu')
    model.eval()
    correct = 0
    total_loss = 0.0
    for i in range(len(dataset)):
        cc, query, target, answer = dataset[i]
        cc = cc.clone().to(device)
        logits = model(cc, query, target)
        loss = nn.functional.cross_entropy(logits.unsqueeze(0),
                                           torch.tensor([answer], device=device),
                                           label_smoothing=label_smoothing)
        total_loss += loss.item()
        pred = logits.argmax().item()
        if pred == answer:
            correct += 1
    accuracy = correct / len(dataset)
    avg_loss = total_loss / len(dataset)
    return accuracy, avg_loss
