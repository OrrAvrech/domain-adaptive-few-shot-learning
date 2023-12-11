from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset
from torchvision.io import read_image


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
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = self._load_images()

    def _load_images(self):
        images = []
        for cls_name in self.classes:
            class_dir = self.root_dir / cls_name
            for img_path in class_dir.iterdir():
                images.append((str(img_path), self.class_to_idx[cls_name]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = read_image(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label
