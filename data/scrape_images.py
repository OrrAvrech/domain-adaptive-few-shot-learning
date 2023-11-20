import os
import pyrallis
from pathlib import Path
from dataclasses import dataclass
from google_images_search import GoogleImagesSearch


@dataclass
class ImageSearchConfig:
    classes: list[str]
    prefix: list[str]
    limit_per_class: int
    output_dir: Path


def download_images(search_params: dict, output_dir: Path):
    # you can provide API key and CX using arguments,
    # or you can set environment variables: GCS_DEVELOPER_KEY, GCS_CX
    dev_api_key = os.getenv("GCS_DEVELOPER_KEY")
    project_cx = os.getenv("GCS_CX")
    gis = GoogleImagesSearch(dev_api_key, project_cx)

    # this will search and download:
    gis.search(search_params=search_params, path_to_dir=str(output_dir))
    for image in gis.results():
        print(image)


@pyrallis.wrap()
def main(cfg: ImageSearchConfig):
    for cls in cfg.classes:
        cls_dir = cfg.output_dir / cls
        num = cfg.limit_per_class // len(cfg.prefix)
        for prefix in cfg.prefix:
            query = f"{prefix} {cls}"
            print(f"{query}")

            search_params = {
                'q': query,
                'num': num
            }

            download_images(search_params=search_params, output_dir=cls_dir)


if __name__ == '__main__':
    main()
