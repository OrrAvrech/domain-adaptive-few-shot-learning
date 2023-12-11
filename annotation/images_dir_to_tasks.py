import json
from pathlib import Path
from common import IMAGE_EXT

images_dir = Path("/Users/orrav/Documents/Data/domain-adaptive-few-shot-learning")
num_annotators = 4
output_dir = Path("./data")
output_dir.mkdir(exist_ok=True)

image_list = []
for ext in IMAGE_EXT:
    image_list += [img for img in images_dir.rglob(ext)]

split_size = len(image_list) // num_annotators
bucket = "meta-dda"
prefix = f"gs://{bucket}/domain-adaptive-few-shot-learning"

for i in range(num_annotators):
    idx = i * split_size
    split_list = image_list[idx : idx + split_size]
    task = [
        {"id": j, "data": {"image": f"{prefix}/{img_path.parts[-2]}/{img_path.name}"}}
        for j, img_path in enumerate(split_list)
    ]

    with open(output_dir / f"data_{i}.json", "w") as fp:
        json.dump(task, fp)
