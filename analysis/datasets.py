from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class ExtractRGB:
    def __call__(self, img):
        return img[:3, ...]


class BaseImageDataset(Dataset):
    def __init__(
        self, root_dir: Path, classes: Optional[list] = None, transform: Optional = None
    ):
        self.root_dir = root_dir
        self.transform = transform
        if classes is None:
            self.classes = sorted(
                [d.name for d in self.root_dir.iterdir() if d.is_dir()]
            )
        else:
            self.classes = classes
        self.class2idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx2class = {i: cls_name for i, cls_name in enumerate(self.classes)}
        self.images = self._load_images()

    def _load_images(self):
        images = []
        for cls_name in self.classes:
            class_dir = self.root_dir / cls_name
            for img_path in class_dir.iterdir():
                images.append((str(img_path), self.class2idx[cls_name]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = read_image(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label


class MapImageDataset(BaseImageDataset):
    def __init__(
        self, root_dir: Path, transform: Optional = None, mapping: Optional[dict] = None
    ):
        super().__init__(root_dir=root_dir, transform=transform)
        self.mapping = dict() if mapping is None else mapping
        self.in2idx = dict()
        for cls in mapping.keys():
            self.in2idx[self.class2idx[cls]] = self.mapping[cls]

    def find_pred(self, in_pred: torch.Tensor) -> int:
        pred = -1
        for folder_idx, in_indices in self.in2idx.items():
            if in_pred in in_indices:
                pred = folder_idx
        return pred

    def map_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        converted = torch.Tensor([self.find_pred(in_pred) for in_pred in predictions]).to(torch.int)
        return converted
