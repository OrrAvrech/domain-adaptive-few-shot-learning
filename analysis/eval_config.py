from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class EvalConfig:
    images_dir: Path
    imagenet_mapping_path: Path
    report_path: Path
    batch_size: int
    use_imagenet_labels: Optional[bool]

    def __post_init__(self):
        self.report_path.parent.mkdir(exist_ok=True, parents=True)
