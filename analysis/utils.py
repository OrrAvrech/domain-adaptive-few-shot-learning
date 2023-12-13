from pathlib import Path

import torch
import yaml


def read_yaml(filepath: Path) -> dict:
    with open(str(filepath), "r") as stream:
        data = yaml.safe_load(stream)
    return data


def per_class_acc(
    predictions: torch.Tensor, labels: torch.Tensor, idx2class: dict
) -> dict[str]:
    per_class_accuracy = dict()
    for i in range(labels.max() + 1):
        class_predictions = predictions[labels == i]
        class_labels = labels[labels == i]
        class_accuracy = (
            torch.Tensor(class_predictions == class_labels).float().mean().item()
            * 100.0
        )
        per_class_accuracy[idx2class[i]] = class_accuracy
    return per_class_accuracy
