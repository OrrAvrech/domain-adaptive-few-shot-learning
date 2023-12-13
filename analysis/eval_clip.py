import torch
import pyrallis
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from eval_config import EvalConfig
from utils import read_yaml, per_class_acc
from imagenet_2012_labels import IMAGENET_2012_LABELS
from datasets import MapImageDataset, ExtractRGB
import pandas as pd


@pyrallis.wrap()
def main(cfg: EvalConfig):
    images_dir = cfg.images_dir
    imagenet_mapping_path = cfg.imagenet_mapping_path
    mapping = read_yaml(imagenet_mapping_path)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image_processor = processor.image_processor
    size = (
        (image_processor.size["shortest_edge"], image_processor.size["shortest_edge"])
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )
    transform = v2.Compose([ExtractRGB(), v2.ToImage(), v2.Resize(size=size)])

    ds = MapImageDataset(images_dir, mapping=mapping, transform=transform)
    test_loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

    tot_pred, tot_labels = None, None
    if cfg.use_imagenet_labels:
        texts = [f"a photo of a {cls}" for idx, cls in sorted(IMAGENET_2012_LABELS.items())]
    else:
        texts = [f"a photo of a {cls}" for cls in ds.classes]

    for i, batch in enumerate(test_loader):
        images, labels = batch
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(**inputs)

        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        predicted_labels = probs.argmax(-1)
        if cfg.use_imagenet_labels:
            predicted_labels = ds.map_predictions(predicted_labels)

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
