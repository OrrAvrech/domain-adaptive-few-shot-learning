import torch
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
        class_prompts = []
        for cls in self.classes:
            class_prompts += [p.replace("_", cls) for p in self.prompts]
        self.class_prompts = class_prompts

        self.output_dir.mkdir(exist_ok=True, parents=True)


@pyrallis.wrap()
def main(cfg: ImageGeneratorConfig):
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
                                             use_safetensors=True, variant="fp16")
    pipe.to("cuda")

    for prompt in cfg.class_prompts:
        print(prompt)
        images = pipe(prompt=prompt).images[:cfg.images_per_prompt]
        filename = str(prompt).replace(" ", "_")
        [im.save(cfg.output_dir / f"{filename}_{i}.png") for i, im in enumerate(images)]


if __name__ == '__main__':
    main()
