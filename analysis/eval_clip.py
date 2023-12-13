import torch
import pyrallis
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from dataclasses import dataclass
from datasets import BaseImageDataset, ExtractRGB
import pandas as pd


@dataclass
class EvalConfig:
    images_dir: Path
    imagenet_mapping_path: Path
    report_path: Path
    batch_size: int

    def __post_init__(self):
        self.report_path.parent.mkdir(exist_ok=True, parents=True)


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


@pyrallis.wrap()
def main(cfg: EvalConfig):
    images_dir = cfg.images_dir
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image_processor = processor.image_processor
    size = (
        (image_processor.size["shortest_edge"], image_processor.size["shortest_edge"])
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )
    transform = v2.Compose([ExtractRGB(), v2.ToImage(), v2.Resize(size=size)])

    ds = BaseImageDataset(images_dir, transform=transform)
    test_loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

    tot_pred, tot_labels = None, None
    texts = [f"a photo of a {cls}" for cls in ds.classes]
    for i, batch in enumerate(test_loader):
        images, labels = batch
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(**inputs)

        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        predicted_labels = probs.argmax(-1)

        if tot_pred is None:
            tot_pred = predicted_labels
            tot_labels = labels
        else:
            tot_pred = torch.cat([tot_pred, predicted_labels], dim=0)
            tot_labels = torch.cat([tot_labels, labels], dim=0)

    accuracy = torch.Tensor(tot_pred == tot_labels).float().mean() * 100
    cls_accuracy = per_class_acc(tot_pred, tot_labels, ds.idx2class)
    print(f"Accuracy: {accuracy}")
    print(f"Per-Class Accuracy: {cls_accuracy}")

    # report
    cls_accuracy["overall"] = accuracy
    pd.DataFrame(cls_accuracy, index=[0]).to_csv(cfg.report_path)


if __name__ == "__main__":
    main()
