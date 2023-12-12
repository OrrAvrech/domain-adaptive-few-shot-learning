import yaml
import torch
from pathlib import Path
from transformers import AutoImageProcessor, ResNetForImageClassification
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from datasets import MapImageDataset, ExtractRGB


def read_yaml(filepath: Path) -> dict:
    with open(str(filepath), "r") as stream:
        data = yaml.safe_load(stream)
    return data


def main():
    images_dir = Path(
        "/Users/orrav/Documents/Data/domain-adaptive-few-shot-learning/images"
    )
    imagenet_mapping_path = Path("./imagenet_mapping.yaml")
    mapping = read_yaml(imagenet_mapping_path)

    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

    size = (
        (processor.size["shortest_edge"], processor.size["shortest_edge"])
        if "shortest_edge" in processor.size
        else (processor.size["height"], processor.size["width"])
    )
    transform = v2.Compose([ExtractRGB(), v2.ToImage(), v2.Resize(size=size)])

    ds = MapImageDataset(images_dir, mapping=mapping, transform=transform)
    test_loader = DataLoader(ds, batch_size=2, shuffle=False)

    acc_sum = 0
    for i, batch in enumerate(test_loader):
        if i > 3:
            break
        images, labels = batch
        inputs = processor(images, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_in_labels = logits.argmax(-1)
        predicted_labels = ds.map_predictions(predicted_in_labels)

        batch_acc = torch.Tensor(predicted_labels == labels).float().sum()
        acc_sum += batch_acc
        for i, (pred_in, pred) in enumerate(zip(predicted_in_labels, predicted_labels)):
            cls_pred_in = model.config.id2label[pred_in.item()]
            cls_pred = ds.idx2class.get(pred.item(), "other")
            print(f"ImageNet pred: {cls_pred_in}, pred: {cls_pred}, label: {labels[i]}")
    accuracy = acc_sum / len(ds)
    print(f"Accuracy: {accuracy*100}")


if __name__ == "__main__":
    main()
