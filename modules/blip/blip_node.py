import os

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import folder_paths
from comfy.model_management import text_encoder_device, soft_empty_cache

from ..model_downloader import load_models
from ..utils import is_junction

blip = None
blip_size = 384
blip_device = text_encoder_device()
blip_current_device = None
cpu = torch.device("cpu")
model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth"


# Freeze PIP modules
def packages(versions=False):
    import subprocess
    import sys

    return [
        (r.decode().split("==")[0] if not versions else r.decode())
        for r in subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"]
        ).split()
    ]


# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def transformImage_legacy(input_image):
    raw_image = input_image.convert("RGB")
    raw_image = raw_image.resize((blip_size, blip_size))
    transform = transforms.Compose(
        [
            transforms.Resize(raw_image.size, interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    image = transform(raw_image).unsqueeze(0).to(blip_device)
    return image


def transformImage(input_image):
    raw_image = input_image.convert("RGB")
    raw_image = raw_image.resize((blip_size, blip_size))
    transform = transforms.Compose(
        [
            transforms.Resize(raw_image.size, interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    image = transform(raw_image).unsqueeze(0).to(blip_device)
    return image.view(
        1, -1, blip_size, blip_size
    )  # Change the shape of the output tensor


def load_blip():
    global blip, blip_current_device
    if blip is None:
        blip_dir = os.path.join(folder_paths.models_dir, "blip")
        if not os.path.exists(blip_dir) and not is_junction(blip_dir):
            os.makedirs(blip_dir, exist_ok=True)

        files = load_models(
            model_path=blip_dir,
            model_url="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth",
            ext_filter=[".pth"],
            download_name="model_base_caption_capfilt_large.pth",
        )

        from .models.blip import blip_decoder

        current_dir = os.path.dirname(os.path.realpath(__file__))
        med_config = os.path.join(current_dir, "configs", "med_config.json")
        blip = blip_decoder(
            pretrained=files[0],
            image_size=blip_size,
            vit="base",
            med_config=med_config,
        )
        blip.eval()

    if blip_current_device != blip_device:
        blip_current_device = blip_device
        blip = blip.to(blip_current_device)

    return blip


def unload_blip():
    global blip, blip_current_device
    if blip is not None:
        blip_current_device = cpu
        blip = blip.to(blip_current_device)

    soft_empty_cache()


class BlipCaption:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "min_length": (
                    "INT",
                    {
                        "default": 24,
                        "min": 0,  # minimum value
                        "max": 200,  # maximum value
                        "step": 1,  # slider's step
                    },
                ),
                "max_length": (
                    "INT",
                    {
                        "default": 48,
                        "min": 0,  # minimum value
                        "max": 200,  # maximum value
                        "step": 1,  # slider's step
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)

    FUNCTION = "blip_caption"

    CATEGORY = "Art Venture/Utils"

    def blip_caption(self, image, min_length, max_length):
        print(f"\033[34mStarting BLIP...\033[0m")

        model = load_blip()
        image = tensor2pil(image)

        if "transformers==4.26.1" in packages(True):
            print("Using Legacy `transformImaage()`")
            tensor = transformImage_legacy(image)
        else:
            tensor = transformImage(image)

        with torch.no_grad():
            caption = model.generate(
                tensor,
                sample=False,
                num_beams=1,
                min_length=min_length,
                max_length=max_length,
            )

        unload_blip()

        return (caption[0],)


NODE_CLASS_MAPPINGS = {"BLIPCaption": BlipCaption}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BLIPCaption": "BLIP Caption",
}
