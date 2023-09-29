import io
import os
import json
import torch
import base64
import random
import requests
from typing import Dict, Tuple

from PIL import Image, ImageOps, ImageFilter
import numpy as np

import folder_paths

from .utils import pil2tensor, tensor2pil, ensure_package, get_dict_attribute


MAX_RESOLUTION = 8192


def prepare_image_for_preview(image: Image.Image, output_dir: str, prefix=None):
    if prefix is None:
        prefix = "preview_" + "".join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))

    # save image to temp folder
    (
        outdir,
        filename,
        counter,
        subfolder,
        _,
    ) = folder_paths.get_save_image_path(prefix, output_dir, image.width, image.height)
    file = f"{filename}_{counter:05}.png"
    image.save(os.path.join(outdir, file), format="PNG", compress_level=4)

    return {
        "filename": file,
        "subfolder": subfolder,
        "type": "temp",
    }


class UtilLoadImageFromUrl:
    def __init__(self) -> None:
        self.output_dir = folder_paths.get_temp_directory()
        self.filename_prefix = "TempImageFromUrl"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {"default": ""}),
            },
            "optional": {
                "keep_alpha_channel": (
                    "BOOLEAN",
                    {"default": False, "label_on": "enabled", "label_off": "disabled"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOOLEAN")
    RETURN_NAMES = ("image", "mask", "has_alpha_channel")
    CATEGORY = "Art Venture/Image"
    FUNCTION = "load_image"

    def load_image_from_url(self, url: str, keep_alpha_channel=False):
        if url.startswith("data:image/"):
            i = Image.open(io.BytesIO(base64.b64decode(url.split(",")[1])))
        else:
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                raise Exception(response.text)

            i = Image.open(io.BytesIO(response.content))

        i = ImageOps.exif_transpose(i)
        mask = None
        if "A" in i.getbands():
            mask = i.getchannel("A")

        if i.mode == "RGBA":
            # recreate image to fix weird RGB image
            alpha = i.split()[-1]
            image = Image.new("RGB", i.size, (0, 0, 0))
            image.paste(i, mask=alpha)
            image.putalpha(alpha)

        if not keep_alpha_channel:
            image = i.convert("RGB")

        return (image, mask)

    def load_image(self, url: str, keep_alpha_channel=False):
        image, mask = self.load_image_from_url(url, keep_alpha_channel)

        # save image to temp folder
        preview = prepare_image_for_preview(image, self.output_dir, self.filename_prefix)
        image = pil2tensor(image)

        if mask:
            mask = np.array(mask).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

        return {"ui": {"images": [preview]}, "result": (image, mask)}


class UtilLoadImageAsMaskFromUrl(UtilLoadImageFromUrl):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {"default": ""}),
                "channel": (["alpha", "red", "green", "blue"],),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)

    def load_image(self, url: str, channel: str):
        image, alpha = self.load_image_from_url(url, False)

        if channel == "alpha":
            mask = alpha
        elif channel == "red":
            mask = image.getchannel("R")
        elif channel == "green":
            mask = image.getchannel("G")
        elif channel == "blue":
            mask = image.getchannel("B")

        mask = np.array(mask).astype(np.float32) / 255.0
        mask = 1.0 - torch.from_numpy(mask)

        return (mask,)


class UtilLoadJsonFromUrl:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {"default": ""}),
            },
            "optional": {
                "print_to_console": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("JSON",)
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "load_json"

    def load_json(self, url: str, print_to_console=False):
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            raise Exception(response.text)

        res = response.json()
        if print_to_console:
            print("JSON content:", json.dumps(res))

        return (res,)


class UtilGetObjectFromJson:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json": ("JSON",),
                "key": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("JSON",)
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "get_objects_from_json"
    OUTPUT_NODE = True

    def get_objects_from_json(self, json: Dict, key: str):
        return (get_dict_attribute(json, key, {}),)


class UtilGetTextFromJson:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json": ("JSON",),
                "key": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "get_string_from_json"
    OUTPUT_NODE = True

    def get_string_from_json(self, json: Dict, key: str):
        return (get_dict_attribute(json, key, ""),)


class UtilGetFloatFromJson:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json": ("JSON",),
                "key": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "get_float_from_json"
    OUTPUT_NODE = True

    def get_float_from_json(self, json: Dict, key: str):
        return (get_dict_attribute(json, key, 0.0),)


class UtilGetIntFromJson:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json": ("JSON",),
                "key": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("INT",)
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "get_int_from_json"
    OUTPUT_NODE = True

    def get_int_from_json(self, json: Dict, key: str):
        return (get_dict_attribute(json, key, 0),)


class UtilGetBoolFromJson:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json": ("JSON",),
                "key": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "get_bool_from_json"
    OUTPUT_NODE = True

    def get_bool_from_json(self, json: Dict, key: str):
        return (get_dict_attribute(json, key, False),)


class UtilRandomInt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "min": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "max": ("INT", {"default": 100, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("INT", "STRING")
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "random_int"

    def random_int(self, min: int, max: int):
        num = torch.randint(min, max, (1,)).item()
        return (num, str(num))


class UtilRandomFloat:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0xFFFFFFFFFFFFFFFF}),
                "max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "random_float"

    def random_float(self, min: float, max: float):
        num = torch.rand(1).item() * (max - min) + min
        return (num, str(num))


class UtilStringToInt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"string": ("STRING", {"default": "0"})},
        }

    RETURN_TYPES = ("INT",)
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "string_to_int"

    def string_to_int(self, string: str):
        return (int(string),)


class UtilNumberScaler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "min": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 0xFFFFFFFFFFFFFFFF}),
                "max": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 0xFFFFFFFFFFFFFFFF}),
                "scale_to_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0xFFFFFFFFFFFFFFFF}),
                "scale_to_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 0xFFFFFFFFFFFFFFFF}),
                "value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "scale_number"

    def scale_number(self, min: float, max: float, scale_to_min: float, scale_to_max: float, value: float):
        num = (value - min) / (max - min) * (scale_to_max - scale_to_min) + scale_to_min
        return (num,)


class UtilImageMuxer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "input_selector": ("INT", {"default": 0}),
            },
            "optional": {"image_3": ("IMAGE",), "image_4": ("IMAGE",)},
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "image_muxer"

    def image_muxer(self, image_1, image_2, input_selector, image_3=None, image_4=None):
        images = [image_1, image_2, image_3, image_4]
        return (images[input_selector],)


class UtilSDXLAspectRatioSelector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "aspect_ratio": (
                    [
                        "1:1",
                        "2:3",
                        "3:4",
                        "5:8",
                        "9:16",
                        "9:19",
                        "9:21",
                        "3:2",
                        "4:3",
                        "8:5",
                        "16:9",
                        "19:9",
                        "21:9",
                    ],
                ),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("ratio", "width", "height")
    FUNCTION = "get_aspect_ratio"
    CATEGORY = "Art Venture/Utils"

    def get_aspect_ratio(self, aspect_ratio):
        width, height = 1024, 1024

        if aspect_ratio == "1:1":
            width, height = 1024, 1024
        elif aspect_ratio == "2:3":
            width, height = 832, 1216
        elif aspect_ratio == "3:4":
            width, height = 896, 1152
        elif aspect_ratio == "5:8":
            width, height = 768, 1216
        elif aspect_ratio == "9:16":
            width, height = 768, 1344
        elif aspect_ratio == "9:19":
            width, height = 704, 1472
        elif aspect_ratio == "9:21":
            width, height = 640, 1536
        elif aspect_ratio == "3:2":
            width, height = 1216, 832
        elif aspect_ratio == "4:3":
            width, height = 1152, 896
        elif aspect_ratio == "8:5":
            width, height = 1216, 768
        elif aspect_ratio == "16:9":
            width, height = 1344, 768
        elif aspect_ratio == "19:9":
            width, height = 1472, 704
        elif aspect_ratio == "21:9":
            width, height = 1536, 640

        return (aspect_ratio, width, height)


class UtilAspectRatioSelector(UtilSDXLAspectRatioSelector):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "aspect_ratio": (
                    [
                        "1:1",
                        "2:3",
                        "3:4",
                        "9:16",
                        "3:2",
                        "4:3",
                        "16:9",
                    ],
                ),
            }
        }

    def get_aspect_ratio(self, aspect_ratio):
        ratio, width, height = super().get_aspect_ratio(aspect_ratio)

        scale_ratio = 512 / min(width, height)

        width = int(scale_ratio * width / 8) * 8
        height = int(scale_ratio * height / 8) * 8

        return (ratio, width, height)


class UtilDependenciesEdit:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dependencies": ("DEPENDENCIES",),
            },
            "optional": {
                "ckpt_name": (
                    [
                        "Original",
                    ]
                    + folder_paths.get_filename_list("checkpoints"),
                ),
                "vae_name": (["Original", "Baked VAE"] + folder_paths.get_filename_list("vae"),),
                "clip": ("CLIP",),
                "clip_skip": (
                    "INT",
                    {"default": 0, "min": -24, "max": 0, "step": 1},
                ),
                "positive": ("STRING", {"default": "Original", "multiline": True}),
                "negative": ("STRING", {"default": "Original", "multiline": True}),
                "lora_stack": ("LORA_STACK",),
                "cnet_stack": ("CONTROL_NET_STACK",),
            },
        }

    RETURN_TYPES = ("DEPENDENCIES",)
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "edit_dependencies"

    def edit_dependencies(
        self,
        dependencies: Tuple,
        vae_name="Original",
        ckpt_name="Original",
        clip=None,
        clip_skip=0,
        positive="Original",
        negative="Original",
        lora_stack=None,
        cnet_stack=None,
    ):
        (
            _vae_name,
            _ckpt_name,
            _clip,
            _clip_skip,
            _positive_prompt,
            _negative_prompt,
            _lora_stack,
            _cnet_stack,
        ) = dependencies

        if vae_name != "Original":
            _vae_name = vae_name
        if ckpt_name != "Original":
            _ckpt_name = ckpt_name
        if clip is not None:
            _clip = clip
        if clip_skip < 0:
            _clip_skip = clip_skip
        if positive != "Original":
            _positive_prompt = positive
        if negative != "Original":
            _negative_prompt = negative
        if lora_stack is not None:
            _lora_stack = lora_stack
        if cnet_stack is not None:
            _cnet_stack = cnet_stack

        dependencies = (
            _vae_name,
            _ckpt_name,
            _clip,
            _clip_skip,
            _positive_prompt,
            _negative_prompt,
            _lora_stack,
            _cnet_stack,
        )

        print("Dependencies:", dependencies)

        return (dependencies,)


class UtilImageScaleDown:
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "width": (
                    "INT",
                    {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1},
                ),
                "height": (
                    "INT",
                    {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1},
                ),
                "crop": (s.crop_methods,),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "image_scale_down"

    def image_scale_down(self, images, width, height, crop):
        if crop == "center":
            old_width = images.shape[2]
            old_height = images.shape[1]
            old_aspect = old_width / old_height
            new_aspect = width / height
            x = 0
            y = 0
            if old_aspect > new_aspect:
                x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
            elif old_aspect < new_aspect:
                y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
            s = images[:, y : old_height - y, x : old_width - x, :]
        else:
            s = images

        results = []
        for image in s:
            img = tensor2pil(image).convert("RGB")
            img = img.resize((width, height), Image.LANCZOS)
            results.append(pil2tensor(img))

        return (torch.cat(results, dim=0),)


class UtilImageScaleDownBy(UtilImageScaleDown):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "scale_by": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "image_scale_down_by"

    def image_scale_down_by(self, images, scale_by):
        width = images.shape[2]
        height = images.shape[1]
        new_width = int(width * scale_by)
        new_height = int(height * scale_by)
        return self.image_scale_down(images, new_width, new_height, "center")


class UtilImageScaleDownToSize(UtilImageScaleDownBy):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "size": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "mode": (["min", "max"], {"default": "max"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "image_scale_down_to_size"

    def image_scale_down_to_size(self, images, size, mode):
        width = images.shape[2]
        height = images.shape[1]

        if mode == "min":
            scale_by = size / min(width, height)
        else:
            scale_by = size / max(width, height)

        scale_by = min(scale_by, 1.0)
        return self.image_scale_down_by(images, scale_by)


class UtilImageScaleDownToTotalPixels(UtilImageScaleDownBy):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "megapixels": ("INT", {"default": 1, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "image_scale_down_to_total_pixels"

    def image_scale_down_to_total_pixels(self, images, megapixels):
        width = images.shape[2]
        height = images.shape[1]
        scale_by = np.sqrt((megapixels * 1024 * 1024) / (width * height))
        scale_by = min(scale_by, 1.0)

        return self.image_scale_down_by(images, scale_by)


class UtilImageAlphaComposite:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "image_alpha_composite"

    def image_alpha_composite(self, image_1: torch.Tensor, image_2: torch.Tensor, alpha):
        if image_1.shape[0] != image_2.shape[0]:
            raise Exception("Images must have the same amount")

        if image_1.shape[1] != image_2.shape[1] or image_1.shape[2] != image_2.shape[2]:
            raise Exception("Images must have the same size")

        composited_images = []
        for i, im1 in enumerate(image_1):
            composited = Image.alpha_composite(
                im1.convert("RGBA"),
                image_2[i].convert("RGBA"),
            )
            composited_images.append(pil2tensor(composited))

        return (torch.cat(composited_images, dim=0),)


class UtilImageGaussianBlur:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "radius": ("INT", {"default": 1, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "image_gaussian_blur"

    def image_gaussian_blur(self, images, radius):
        blured_images = []
        for image in images:
            img = tensor2pil(image)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
            blured_images.append(pil2tensor(img))

        return (torch.cat(blured_images, dim=0),)


class UtillQRCodeGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "qr_version": ("INT", {"default": 1, "min": 1, "max": 40, "step": 1}),
                "error_correction": (["L", "M", "Q", "H"], {"default": "H"}),
                "box_size": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
                "border": ("INT", {"default": 4, "min": 0, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "create_qr_code"
    CATEGORY = "Art Venture/Utils"

    def create_qr_code(self, text, size, qr_version, error_correction, box_size, border):
        ensure_package("qrcode", "qrcode[pil]")
        import qrcode

        if error_correction == "L":
            error_level = qrcode.constants.ERROR_CORRECT_L
        elif error_correction == "M":
            error_level = qrcode.constants.ERROR_CORRECT_M
        elif error_correction == "Q":
            error_level = qrcode.constants.ERROR_CORRECT_Q
        else:
            error_level = qrcode.constants.ERROR_CORRECT_H

        qr = qrcode.QRCode(version=qr_version, error_correction=error_level, box_size=box_size, border=border)
        qr.add_data(text)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        img = img.resize((size, size)).convert("RGB")

        return (pil2tensor(img),)


class UtilSeedSelector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": (["random", "fixed"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "fixed_seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "get_seed"

    def get_seed(self, mode, seed, fixed_seed):
        return (seed if mode == "random" else fixed_seed,)


NODE_CLASS_MAPPINGS = {
    "LoadImageFromUrl": UtilLoadImageFromUrl,
    "LoadImageAsMaskFromUrl": UtilLoadImageAsMaskFromUrl,
    "StringToInt": UtilStringToInt,
    "ImageMuxer": UtilImageMuxer,
    "ImageScaleDown": UtilImageScaleDown,
    "ImageScaleDownBy": UtilImageScaleDownBy,
    "ImageScaleDownToSize": UtilImageScaleDownToSize,
    "ImageScaleDownToTotalPixels": UtilImageScaleDownToTotalPixels,
    "ImageAlphaComposite": UtilImageAlphaComposite,
    "ImageGaussianBlur": UtilImageGaussianBlur,
    "QRCodeGenerator": UtillQRCodeGenerator,
    "DependenciesEdit": UtilDependenciesEdit,
    "AspectRatioSelector": UtilAspectRatioSelector,
    "SDXLAspectRatioSelector": UtilSDXLAspectRatioSelector,
    "SeedSelector": UtilSeedSelector,
    "LoadJsonFromUrl": UtilLoadJsonFromUrl,
    "GetObjectFromJson": UtilGetObjectFromJson,
    "GetTextFromJson": UtilGetTextFromJson,
    "GetFloatFromJson": UtilGetFloatFromJson,
    "GetIntFromJson": UtilGetIntFromJson,
    "GetBoolFromJson": UtilGetBoolFromJson,
    "RandomInt": UtilRandomInt,
    "RandomFloat": UtilRandomFloat,
    "NumberScaler": UtilNumberScaler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageFromUrl": "Load Image From URL",
    "LoadImageAsMaskFromUrl": "Load Image (as Mask) From URL",
    "StringToInt": "String to Int",
    "ImageMuxer": "Image Muxer",
    "ImageScaleDown": "Scale Down",
    "ImageScaleDownBy": "Scale Down By",
    "ImageScaleDownToSize": "Scale Down To Size",
    "ImageScaleDownToTotalPixels": "Scale Down To Megapixels",
    "ImageAlphaComposite": "Image Alpha Composite",
    "ImageGaussianBlur": "Image Gaussian Blur",
    "QRCodeGenerator": "QR Code Generator",
    "DependenciesEdit": "Dependencies Edit",
    "AspectRatioSelector": "Aspect Ratio",
    "SDXLAspectRatioSelector": "SDXL Aspect Ratio",
    "SeedSelector": "Seed Selector",
    "LoadJsonFromUrl": "Load JSON From URL",
    "GetObjectFromJson": "Get Object From JSON",
    "GetTextFromJson": "Get Text From JSON",
    "GetFloatFromJson": "Get Float From JSON",
    "GetIntFromJson": "Get Int From JSON",
    "GetBoolFromJson": "Get Bool From JSON",
    "RandomInt": "Random Int",
    "RandomFloat": "Random Float",
    "NumberScaler": "Number Scaler",
}
