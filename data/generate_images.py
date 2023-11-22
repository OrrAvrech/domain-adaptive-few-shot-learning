import torch
import random
import pyrallis
from pathlib import Path
from diffusers import DiffusionPipeline
from dataclasses import dataclass


@dataclass
class ImageGeneratorConfig:
    classes: list[str]
    prompts: list[str]
    images_per_prompt: int
    output_dir: Path

    def __post_init__(self):
        class_prompts = dict()
        for cls in self.classes:
            cls_pair = random.choice([c for c in self.classes if c != cls])
            class_prompts[cls] = [p.replace("_", cls).replace("+", cls_pair) for p in self.prompts]
        self.class_prompts = class_prompts

        self.output_dir.mkdir(exist_ok=True, parents=True)


@pyrallis.wrap()
def main(cfg: ImageGeneratorConfig):
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
                                             use_safetensors=True, variant="fp16")
    pipe.to("cuda")

    for cls, prompts in cfg.class_prompts.items():
        for prompt in prompts:
            print(prompt)
            images = pipe(prompt=prompt).images[:cfg.images_per_prompt]
            filename = str(prompt).replace(" ", "_")
            cls_dir = cfg.output_dir / cls
            cls_dir.mkdir(exist_ok=True, parents=True)
            [im.save(cls_dir / f"{filename}_{i}.png") for i, im in enumerate(images)]


if __name__ == '__main__':
    main()
