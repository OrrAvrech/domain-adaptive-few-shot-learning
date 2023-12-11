from pathlib import Path
import pandas as pd
from shutil import copy


def main():
    raw_images_dir = Path("/Users/orrav/Documents/Data/domain-adaptive-few-shot-learning/raw_images")
    dst_dir = Path("/Users/orrav/Documents/Data/domain-adaptive-few-shot-learning/images")
    annotations_path = Path("/Users/orrav/Documents/Data/domain-adaptive-few-shot-learning/filtered_annotations.csv")
    df = pd.read_csv(annotations_path)
    df["filename"] = df.apply(lambda x: f'{x["folder"]}/{x["img_name"]}', axis=1)
    filtered_names = df["filename"].to_list()
    filtered_paths = [raw_images_dir / p for p in filtered_names]
    for img_path in filtered_paths:
        img_dir = dst_dir / img_path.parts[-2]
        img_dir.mkdir(exist_ok=True, parents=True)
        dst_path = img_dir / img_path.name
        copy(img_path, dst_path)


if __name__ == "__main__":
    main()
