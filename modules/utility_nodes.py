import io
import os
import json
import torch
import base64
import random
import requests
from typing import List, Dict, Tuple, Optional

from PIL import Image, ImageOps, ImageFilter
import numpy as np

import folder_paths
import comfy.utils
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel

from .utils import pil2tensor, tensor2pil, ensure_package, get_dict_attribute


MAX_RESOLUTION = 8192


class AnyType(str):
    """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

    def __ne__(self, __value: object) -> bool:
        return False


class FlexibleOptionalInputType(dict):
    """A special class to make flexible nodes that pass data to our python handlers.

    Enables both flexible/dynamic input types (like for Any Switch) or a dynamic number of inputs
    (like for Any Switch, Context Switch, Context Merge, Power Lora Loader, etc).

    Note, for ComfyUI, all that's needed is the `__contains__` override below, which tells ComfyUI
    that our node will handle the input, regardless of what it is.

    However, with https://github.com/comfyanonymous/ComfyUI/pull/2666 a large change would occur
    requiring more details on the input itself. There, we need to return a list/tuple where the first
    item is the type. This can be a real type, or use the AnyType for additional flexibility.

    This should be forwards compatible unless more changes occur in the PR.
    """

    def __init__(self, type):
        self.type = type

    def __getitem__(self, key):
        return (self.type,)

    def __contains__(self, key):
        return True


any_type = AnyType("*")


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
    file = f"{filename}_{counter:05}_.png"
    image.save(os.path.join(outdir, file), format="PNG", compress_level=4)

    return {
        "filename": file,
        "subfolder": subfolder,
        "type": "temp",
    }


def load_images_from_url(urls: List[str], keep_alpha_channel=False):
    images: List[Image.Image] = []
    masks: List[Optional[Image.Image]] = []

    for url in urls:
        if url.startswith("data:image/"):
            i = Image.open(io.BytesIO(base64.b64decode(url.split(",")[1])))
        elif url.startswith("file://"):
            url = url[7:]
            if not os.path.isfile(url):
                raise Exception(f"File {url} does not exist")

            i = Image.open(url)
        elif url.startswith("http://") or url.startswith("https://"):
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                raise Exception(response.text)

            i = Image.open(io.BytesIO(response.content))
        elif url.startswith(("/view?", "/api/view?")):
            from urllib.parse import parse_qs

            qs_idx = url.find("?")
            qs = parse_qs(url[qs_idx + 1 :])
            filename = qs.get("name", qs.get("filename", None))
            if filename is None:
                raise Exception(f"Invalid url: {url}")

            filename = filename[0]
            subfolder = qs.get("subfolder", None)
            if subfolder is not None:
                filename = os.path.join(subfolder[0], filename)

            dirtype = qs.get("type", ["input"])
            if dirtype[0] == "input":
                url = os.path.join(folder_paths.get_input_directory(), filename)
            elif dirtype[0] == "output":
                url = os.path.join(folder_paths.get_output_directory(), filename)
            elif dirtype[0] == "temp":
                url = os.path.join(folder_paths.get_temp_directory(), filename)
            else:
                raise Exception(f"Invalid url: {url}")

            i = Image.open(url)
        elif url == "":
            continue
        else:
            url = folder_paths.get_annotated_filepath(url)
            if not os.path.isfile(url):
                raise Exception(f"Invalid url: {url}")

            i = Image.open(url)

        i = ImageOps.exif_transpose(i)
        has_alpha = "A" in i.getbands()
        mask = None

        if "RGB" not in i.mode:
            i = i.convert("RGBA") if has_alpha else i.convert("RGB")

        if has_alpha:
            mask = i.getchannel("A")

        if not keep_alpha_channel:
            image = i.convert("RGB")
        else:
            image = i

        images.append(image)
        masks.append(mask)

    return (images, masks)


class UtilLoadImageFromUrl:
    def __init__(self) -> None:
        self.output_dir = folder_paths.get_temp_directory()
        self.filename_prefix = "TempImageFromUrl"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("STRING", {
                    "default": "",
                    "placeholder": "Input image paths or URLS one per line. Eg:\nhttps://example.com/image.png\nfile:///path/to/local/image.jpg\ndata:image/png;base64,...",
                    "multiline": True,
                    "dynamicPrompts": False,
                }),
            },
            "optional": {
                "keep_alpha_channel": (
                    "BOOLEAN",
                    {"default": False, "label_on": "enabled", "label_off": "disabled"},
                ),
                "output_mode": (
                    "BOOLEAN",
                    {"default": False, "label_on": "list", "label_off": "batch"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOOLEAN")
    OUTPUT_IS_LIST = (True, True, False)
    RETURN_NAMES = ("images", "masks", "has_image")
    CATEGORY = "Art Venture/Image"
    FUNCTION = "load_image"

    def load_image(self, image: str, keep_alpha_channel=False, output_mode=False):
        urls = image.strip().split("\n")
        pil_images, pil_masks = load_images_from_url(urls, keep_alpha_channel)
        has_image = len(pil_images) > 0
        if not has_image:
            i = torch.zeros((1, 64, 64, 3), dtype=torch.float32, device="cpu")
            m = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            pil_images = [tensor2pil(i)]
            pil_masks = [tensor2pil(m, mode="L")]

        previews = []
        np_images: list[torch.Tensor] = []
        np_masks: list[torch.Tensor] = []

        for pil_image, pil_mask in zip(pil_images, pil_masks):
            if pil_mask is not None:
                preview_image = Image.new("RGB", pil_image.size)
                preview_image.paste(pil_image, (0, 0))
                preview_image.putalpha(pil_mask)
            else:
                preview_image = pil_image

            previews.append(prepare_image_for_preview(preview_image, self.output_dir, self.filename_prefix))

            np_image = pil2tensor(pil_image)
            if pil_mask:
                np_mask = np.array(pil_mask).astype(np.float32) / 255.0
                np_mask = 1.0 - torch.from_numpy(np_mask)
            else:
                np_mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

            np_images.append(np_image)
            np_masks.append(np_mask.unsqueeze(0))

        if output_mode:
            result = (np_images, np_masks, has_image)
        else:
            has_size_mismatch = False
            if len(np_images) > 1:
                for np_image in np_images[1:]:
                    if np_image.shape[1] != np_images[0].shape[1] or np_image.shape[2] != np_images[0].shape[2]:
                        has_size_mismatch = True
                        break

            if has_size_mismatch:
                raise Exception("To output as batch, images must have the same size. Use list output mode instead.")

            result = ([torch.cat(np_images)], [torch.cat(np_masks)], has_image)

        return {"ui": {"images": previews}, "result": result}


class UtilLoadImageAsMaskFromUrl(UtilLoadImageFromUrl):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("STRING", {
                    "default": "",
                    "placeholder": "Input image paths or URLS one per line. Eg:\nhttps://example.com/image.png\nfile:///path/to/local/image.jpg\ndata:image/png;base64,...",
                    "multiline": True,
                    "dynamicPrompts": False,
                }),
                "channel": (["alpha", "red", "green", "blue"],),
            },
            "optional": {
                "output_mode": (
                    "BOOLEAN",
                    {"default": False, "label_on": "list", "label_off": "batch"},
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)
    OUTPUT_IS_LIST = (True,)

    def load_image(self, image: str, channel: str, output_mode=False, url=""):
        if not image or image == "":
            image = url

        urls = image.strip().split("\n")
        pil_images, pil_alphas = load_images_from_url(urls, True)

        masks: List[torch.Tensor] = []

        for img, alpha in zip(pil_images, pil_alphas):
            if channel == "alpha":
                mask = alpha
            elif channel == "red":
                mask = img.getchannel("R")
            elif channel == "green":
                mask = img.getchannel("G")
            elif channel == "blue":
                mask = img.getchannel("B")

            if mask:
                mask = np.array(mask, dtype=np.float32) / 255.0
                mask = torch.from_numpy(mask)
                if channel == "alpha":
                    mask = 1.0 - mask
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

            masks.append(mask.unsqueeze(0))

        if output_mode:
            return (masks,)

        if len(masks) > 1:
            for mask in masks[1:]:
                if mask.shape[0] != masks[0].shape[0] or mask.shape[1] != masks[0].shape[1]:
                    raise Exception("To output as batch, masks must have the same size. Use list output mode instead.")

        return ([torch.cat(masks)],)


class UtilLoadJsonFromText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": (
                    "STRING",
                    {"multiline": True, "dynamicPrompts": False, "placeholder": "JSON object. Eg: {'key': 'value'}"},
                ),
            }
        }

    RETURN_TYPES = ("JSON",)
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "load_json"

    def load_json(self, data: str):
        return (json.loads(data),)


class UtilLoadJsonFromUrl:
    @classmethod
    def INPUT_TYPES(cls):
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
    def INPUT_TYPES(cls):
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
    def INPUT_TYPES(cls):
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
        return (str(get_dict_attribute(json, key, "")),)


class UtilGetFloatFromJson:
    @classmethod
    def INPUT_TYPES(cls):
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
        return (float(get_dict_attribute(json, key, 0.0)),)


class UtilGetIntFromJson:
    @classmethod
    def INPUT_TYPES(cls):
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
        return (int(get_dict_attribute(json, key, 0)),)


class UtilGetBoolFromJson:
    @classmethod
    def INPUT_TYPES(cls):
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
    def INPUT_TYPES(cls):
        return {
            "required": {
                "min": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "max": ("INT", {"default": 100, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("INT", "STRING")
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "random_int"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return torch.rand(1).item()

    def random_int(self, min: int, max: int):
        num = torch.randint(min, max, (1,)).item()
        return (num, str(num))


class UtilRandomFloat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0xFFFFFFFFFFFFFFFF}),
                "max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "random_float"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return torch.rand(1).item()

    def random_float(self, min: float, max: float):
        num = torch.rand(1).item() * (max - min) + min
        return (num, str(num))


class UtilStringToInt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"string": ("STRING", {"default": "0"})},
        }

    RETURN_TYPES = ("INT",)
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "string_to_int"

    def string_to_int(self, string: str):
        return (int(string),)


class UtilStringToNumber:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"default": "0"}),
                "rounding": (["round", "floor", "ceil"], {"default": "round"}),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT")
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "string_to_numbers"

    def string_to_numbers(self, string: str, rounding):
        f = float(string)

        if rounding == "floor":
            return (int(np.floor(f)), f)
        elif rounding == "ceil":
            return (int(np.ceil(f)), f)
        else:
            return (int(round(f)), f)


class UtilNumberScaler:
    @classmethod
    def INPUT_TYPES(cls):
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


class UtilBooleanPrimitive:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("BOOLEAN", {"default": False}),
                "reverse": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "STRING")
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "boolean_primitive"

    def boolean_primitive(self, value: bool, reverse: bool):
        if reverse:
            value = not value

        return (value, str(value))


class UtilTextSwitchCase:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "switch_cases": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "placeholder": "case_1:output_1\ncase_2:output_2\nthat span multiple lines\ncase_3:output_3",
                    },
                ),
                "condition": ("STRING", {"default": ""}),
                "default_value": ("STRING", {"default": ""}),
                "delimiter": ("STRING", {"default": ":"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "text_switch_case"

    def text_switch_case(self, switch_cases: str, condition: str, default_value: str, delimiter: str = ":"):
        # Split into cases first
        cases = switch_cases.split("\n")
        current_case = None
        current_output = []

        for line in cases:
            if delimiter in line:
                # Process previous case if exists
                if current_case is not None and condition == current_case:
                    return ("\n".join(current_output),)

                # Start new case
                current_case, output = line.split(delimiter, 1)
                current_output = [output]
            elif current_case is not None:
                current_output.append(line)

        # Check last case
        if current_case is not None and condition == current_case:
            return ("\n".join(current_output),)

        return (default_value,)


class UtilImageMuxer:
    @classmethod
    def INPUT_TYPES(cls):
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
    def INPUT_TYPES(cls):
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
    def INPUT_TYPES(cls):
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

        scale_ratio = 768 / max(width, height)

        width = int(scale_ratio * width / 8) * 8
        height = int(scale_ratio * height / 8) * 8

        return (ratio, width, height)


class UtilDependenciesEdit:
    @classmethod
    def INPUT_TYPES(cls):
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
    def INPUT_TYPES(cls):
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
                "crop": (cls.crop_methods,),
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
            img = img.resize((width, height), Image.Resampling.LANCZOS)
            results.append(pil2tensor(img))

        return (torch.cat(results, dim=0),)


class UtilImageScaleDownBy(UtilImageScaleDown):
    @classmethod
    def INPUT_TYPES(cls):
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
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "size": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "mode": ("BOOLEAN", {"default": True, "label_on": "max", "label_off": "min"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "image_scale_down_to_size"

    def image_scale_down_to_size(self, images, size, mode):
        width = images.shape[2]
        height = images.shape[1]

        if mode:
            scale_by = size / max(width, height)
        else:
            scale_by = size / min(width, height)

        scale_by = min(scale_by, 1.0)
        return self.image_scale_down_by(images, scale_by)


class UtilImageScaleToTotalPixels(UtilImageScaleDownBy, ImageUpscaleWithModel):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "megapixels": ("FLOAT", {"default": 1, "min": 0.1, "max": 100, "step": 0.05}),
            },
            "optional": {
                "upscale_model_opt": ("UPSCALE_MODEL",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "image_scale_down_to_total_pixels"

    def image_scale_up_by(self, images: torch.Tensor, scale_by, upscale_model_opt):
        width = round(images.shape[2] * scale_by)
        height = round(images.shape[1] * scale_by)

        if scale_by < 1.2 or upscale_model_opt is None:
            s = images.movedim(-1, 1)
            s = comfy.utils.common_upscale(s, width, height, "bicubic", "disabled")
            s = s.movedim(1, -1)
            return (s,)
        else:
            s = self.upscale(upscale_model_opt, images)[0]
            return self.image_scale_down(s, width, height, "center")

    def image_scale_down_to_total_pixels(self, images, megapixels, upscale_model_opt=None):
        width = images.shape[2]
        height = images.shape[1]
        scale_by = np.sqrt((megapixels * 1024 * 1024) / (width * height))

        if scale_by <= 1.0:
            return self.image_scale_down_by(images, scale_by)
        else:
            return self.image_scale_up_by(images, scale_by, upscale_model_opt)


class UtilImageAlphaComposite:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "image_alpha_composite"

    def image_alpha_composite(self, image_1: torch.Tensor, image_2: torch.Tensor):
        if image_1.shape[0] != image_2.shape[0]:
            raise Exception("Images must have the same amount")

        if image_1.shape[1] != image_2.shape[1] or image_1.shape[2] != image_2.shape[2]:
            raise Exception("Images must have the same size")

        composited_images = []
        for i, im1 in enumerate(image_1):
            composited = Image.alpha_composite(
                tensor2pil(im1).convert("RGBA"),
                tensor2pil(image_2[i]).convert("RGBA"),
            )
            composited_images.append(pil2tensor(composited))

        return (torch.cat(composited_images, dim=0),)


class UtilImageGaussianBlur:
    @classmethod
    def INPUT_TYPES(cls):
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


class UtilImageExtractChannel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "channel": (["R", "G", "B", "A"],),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("channel_data",)
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "image_extract_alpha"

    def image_extract_alpha(self, images: torch.Tensor, channel):
        # images in shape (N, H, W, C)

        if len(images.shape) < 4:
            images = images.unsqueeze(3).repeat(1, 1, 1, 3)

        if channel == "A" and images.shape[3] < 4:
            raise Exception("Image does not have an alpha channel")

        channel_index = ["R", "G", "B", "A"].index(channel)
        mask = images[:, :, :, channel_index].cpu().clone()

        return (mask,)


class UtilImageApplyChannel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "channel_data": ("MASK",),
                "channel": (["R", "G", "B", "A"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "image_apply_channel"

    def image_apply_channel(self, images: torch.Tensor, channel_data: torch.Tensor, channel):
        merged_images = []

        for image in images:
            image = image.cpu().clone()

            if channel == "A":
                if image.shape[2] < 4:
                    image = torch.cat([image, torch.ones((image.shape[0], image.shape[1], 1))], dim=2)

                image[:, :, 3] = channel_data
            elif channel == "R":
                image[:, :, 0] = channel_data
            elif channel == "G":
                image[:, :, 1] = channel_data
            else:
                image[:, :, 2] = channel_data

            merged_images.append(image)

        return (torch.stack(merged_images),)


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
        ensure_package("qrcode", install_package_name="qrcode[pil]")
        import qrcode

        if error_correction == "L":
            error_level = qrcode.ERROR_CORRECT_L
        elif error_correction == "M":
            error_level = qrcode.ERROR_CORRECT_M
        elif error_correction == "Q":
            error_level = qrcode.ERROR_CORRECT_Q
        else:
            error_level = qrcode.ERROR_CORRECT_H

        qr = qrcode.QRCode(version=qr_version, error_correction=error_level, box_size=box_size, border=border)
        qr.add_data(text)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        img = img.resize((size, size)).convert("RGB")

        return (pil2tensor(img),)


class UtilRepeatImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "amount": ("INT", {"default": 1, "min": 1, "max": 1024}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "rebatch"

    def rebatch(self, images: torch.Tensor, amount):
        return (images.repeat(amount, 1, 1, 1),)


class UtilSeedSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": ("BOOLEAN", {"default": True, "label_on": "random", "label_off": "fixed"}),
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
        return (fixed_seed if not mode else seed,)


class UtilCheckpointSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            }
        }

    RETURN_TYPES = (folder_paths.get_filename_list("checkpoints"), "STRING")
    RETURN_NAMES = ("ckpt_name", "ckpt_name_str")
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "get_ckpt_name"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return torch.rand(1).item()

    def get_ckpt_name(self, ckpt_name):
        return (ckpt_name, ckpt_name)


class UtilModelMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model1": ("MODEL",),
                "model2": ("MODEL",),
                "ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "merge_models"

    def merge_models(self, model1, model2, ratio=1.0):
        m = model1.clone()
        kp = model2.get_key_patches("diffusion_model.")

        for k in kp:
            k_unet = k[len("diffusion_model.") :]
            if k_unet == "input_blocks.0.0.weight":
                w = kp[k][0]
                if w.shape[1] == 9:
                    w = w[:, 0:4, :, :]
                m.add_patches({k: (w,)}, 1.0 - ratio, ratio)
            else:
                m.add_patches({k: kp[k]}, 1.0 - ratio, ratio)

        return (m,)


class UtilTextRandomMultiline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "amount": ("INT", {"default": 1, "min": 1, "max": 1024}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lines",)
    OUTPUT_IS_LIST = (True,)
    CATEGORY = "Art Venture/Utils"
    FUNCTION = "random_multiline"

    def random_multiline(self, text: str, amount=1, seed=0):
        lines = text.strip().split("\n")
        lines = [line.strip() for line in lines if line.strip()]

        custom_random = random.Random(seed)
        custom_random.shuffle(lines)
        return (lines[:amount],)


NODE_CLASS_MAPPINGS = {
    "LoadImageFromUrl": UtilLoadImageFromUrl,
    "LoadImageAsMaskFromUrl": UtilLoadImageAsMaskFromUrl,
    "StringToInt": UtilStringToInt,
    "StringToNumber": UtilStringToNumber,
    "BooleanPrimitive": UtilBooleanPrimitive,
    "ImageMuxer": UtilImageMuxer,
    "ImageScaleDown": UtilImageScaleDown,
    "ImageScaleDownBy": UtilImageScaleDownBy,
    "ImageScaleDownToSize": UtilImageScaleDownToSize,
    "ImageScaleToMegapixels": UtilImageScaleToTotalPixels,
    "ImageAlphaComposite": UtilImageAlphaComposite,
    "ImageGaussianBlur": UtilImageGaussianBlur,
    "ImageRepeat": UtilRepeatImages,
    "ImageExtractChannel": UtilImageExtractChannel,
    "ImageApplyChannel": UtilImageApplyChannel,
    "QRCodeGenerator": UtillQRCodeGenerator,
    "DependenciesEdit": UtilDependenciesEdit,
    "AspectRatioSelector": UtilAspectRatioSelector,
    "SDXLAspectRatioSelector": UtilSDXLAspectRatioSelector,
    "SeedSelector": UtilSeedSelector,
    "CheckpointNameSelector": UtilCheckpointSelector,
    "LoadJsonFromUrl": UtilLoadJsonFromUrl,
    "LoadJsonFromText": UtilLoadJsonFromText,
    "GetObjectFromJson": UtilGetObjectFromJson,
    "GetTextFromJson": UtilGetTextFromJson,
    "GetFloatFromJson": UtilGetFloatFromJson,
    "GetIntFromJson": UtilGetIntFromJson,
    "GetBoolFromJson": UtilGetBoolFromJson,
    "RandomInt": UtilRandomInt,
    "RandomFloat": UtilRandomFloat,
    "NumberScaler": UtilNumberScaler,
    "MergeModels": UtilModelMerge,
    "TextRandomMultiline": UtilTextRandomMultiline,
    "TextSwitchCase": UtilTextSwitchCase,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageFromUrl": "Load Image From URL",
    "LoadImageAsMaskFromUrl": "Load Image (as Mask) From URL",
    "StringToInt": "String to Int",
    "StringToNumber": "String to Number",
    "BooleanPrimitive": "Boolean",
    "ImageMuxer": "Image Muxer",
    "ImageScaleDown": "Scale Down",
    "ImageScaleDownBy": "Scale Down By",
    "ImageScaleDownToSize": "Scale Down To Size",
    "ImageScaleToMegapixels": "Scale To Megapixels",
    "ImageAlphaComposite": "Image Alpha Composite",
    "ImageGaussianBlur": "Image Gaussian Blur",
    "ImageRepeat": "Repeat Images",
    "ImageExtractChannel": "Image Extract Channel",
    "ImageApplyChannel": "Image Apply Channel",
    "QRCodeGenerator": "QR Code Generator",
    "DependenciesEdit": "Dependencies Edit",
    "AspectRatioSelector": "Aspect Ratio",
    "SDXLAspectRatioSelector": "SDXL Aspect Ratio",
    "SeedSelector": "Seed Selector",
    "CheckpointNameSelector": "Checkpoint Name Selector",
    "LoadJsonFromUrl": "Load JSON From URL",
    "LoadJsonFromText": "Load JSON From Text",
    "GetObjectFromJson": "Get Object From JSON",
    "GetTextFromJson": "Get Text From JSON",
    "GetFloatFromJson": "Get Float From JSON",
    "GetIntFromJson": "Get Int From JSON",
    "GetBoolFromJson": "Get Bool From JSON",
    "RandomInt": "Random Int",
    "RandomFloat": "Random Float",
    "NumberScaler": "Number Scaler",
    "MergeModels": "Merge Models",
    "TextRandomMultiline": "Text Random Multiline",
    "TextSwitchCase": "Text Switch Case",
}
