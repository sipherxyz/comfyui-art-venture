import io
import os
import json
import torch
import base64
import requests
from typing import Dict, Tuple

from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
import numpy as np

import folder_paths
from nodes import LoraLoader, VAELoader

from .logger import logger
from .utils import upload_to_av
from .sdxl_prompt_styler import (
    NODE_CLASS_MAPPINGS as SDXL_STYLER_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as SDXL_STYLER_NODE_DISPLAY_NAME_MAPPINGS,
)
from .blip import (
    NODE_CLASS_MAPPINGS as BLIP_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as BLIP_NODE_DISPLAY_NAME_MAPPINGS,
)
from .fooocus import (
    NODE_CLASS_MAPPINGS as FOOOCUS_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as FOOOCUS_NODE_DISPLAY_NAME_MAPPINGS,
)
from .postprocessing import (
    NODE_CLASS_MAPPINGS as PP_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as PP_NODE_DISPLAY_NAME_MAPPINGS,
)
from .controlnet_nodes import (
    NODE_CLASS_MAPPINGS as CONTROLNET_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as CONTROLNET_NODE_DISPLAY_NAME_MAPPINGS,
)


MAX_RESOLUTION = 4096


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

    RETURN_TYPES = ("IMAGE", "MASK")
    CATEGORY = "Art Venture/Image"
    FUNCTION = "load_image_from_url"

    def load_image_from_url(self, url: str, keep_alpha_channel=False):
        if url.startswith("data:image/"):
            i = Image.open(io.BytesIO(base64.b64decode(url.split(",")[1])))
        else:
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                raise Exception(response.text)

            i = Image.open(io.BytesIO(response.content))

        i = ImageOps.exif_transpose(i)
        if not keep_alpha_channel:
            image = i.convert("RGB")
        else:
            if i.mode != "RGBA":
                i = i.convert("RGBA")

            # recreate image to fix weird RGB image
            alpha = i.split()[-1]
            image = Image.new("RGB", i.size, (0, 0, 0))
            image.paste(i, mask=alpha)
            image.putalpha(alpha)

        # save image to temp folder
        (
            outdir,
            filename,
            counter,
            subfolder,
            _,
        ) = folder_paths.get_save_image_path(
            self.filename_prefix, self.output_dir, image.width, image.height
        )
        file = f"{filename}_{counter:05}.png"
        image.save(os.path.join(outdir, file), format="PNG", compress_level=4)
        preview = {
            "filename": file,
            "subfolder": subfolder,
            "type": "temp",
        }

        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if "A" in i.getbands():
            mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

        return {"ui": {"images": [preview]}, "result": (image, mask)}


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
                "vae_name": (
                    ["Original", "Baked VAE"] + folder_paths.get_filename_list("vae"),
                ),
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

        pil_images = []
        for idx, image in enumerate(s):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img = img.convert("RGB")
            img = img.resize((width, height), Image.LANCZOS)
            pil_images.append(img)

        results = torch.cat(
            [
                torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None,]
                for img in pil_images
            ],
            dim=0,
        )

        return (results,)


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
        print("images", images.shape)
        width = images.shape[2]
        height = images.shape[1]
        new_width = int(width * scale_by)
        new_height = int(height * scale_by)
        return self.image_scale_down(images, new_width, new_height, "center")


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


class AVVAELoader(VAELoader):
    @classmethod
    def INPUT_TYPES(s):
        inputs = VAELoader.INPUT_TYPES()
        inputs["optional"] = {"vae_override": ("STRING", {"default": "None"})}
        return inputs

    CATEGORY = "Art Venture/Loaders"

    def load_vae(self, vae_name, vae_override="None"):
        if vae_override != "None":
            if vae_override not in folder_paths.get_filename_list("vae"):
                print(
                    f"Warning: Not found VAE model {vae_override}. Use {vae_name} instead."
                )
            else:
                vae_name = vae_override

        return super().load_vae(vae_name)


class AVLoraLoader(LoraLoader):
    @classmethod
    def INPUT_TYPES(s):
        inputs = LoraLoader.INPUT_TYPES()
        inputs["optional"] = {"lora_override": ("STRING", {"default": "None"})}
        return inputs

    CATEGORY = "Art Venture/Loaders"

    def load_lora(self, lora_name, lora_override="None"):
        if lora_override != "None":
            if lora_override not in folder_paths.get_filename_list("loras"):
                print(
                    f"Warning: Not found Lora model {lora_override}. Use {lora_name} instead."
                )
            else:
                lora_name = lora_override

        return super().load_lora(lora_name)


class AVOutputUploadImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"images": ("IMAGE",)},
            "optional": {
                "folder_id": ("STRING", {"multiline": False}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    CATEGORY = "Art Venture"
    FUNCTION = "upload_images"

    def upload_images(
        self,
        images,
        folder_id: str = None,
        prompt=None,
        extra_pnginfo=None,
    ):
        files = list()
        for idx, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    logger.debug(f"Adding {x} to pnginfo: {extra_pnginfo[x]}")
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            buffer = io.BytesIO()
            buffer.seek(0)
            files.append(
                (
                    "files",
                    (f"image-{idx}.png", buffer, "image/png"),
                )
            )

        additional_data = {"success": "true"}
        if folder_id is not None:
            additional_data["folderId"] = folder_id

        upload_to_av(files, additional_data=additional_data)
        return ("Uploaded to ArtVenture!",)


class AVCheckpointModelsToParametersPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            },
            "optional": {
                "pipe": ("PIPE",),
                "secondary_ckpt_name": (
                    ["None"] + folder_paths.get_filename_list("checkpoints"),
                ),
                "vae_name": (["None"] + folder_paths.get_filename_list("vae"),),
                "upscaler_name": (
                    ["None"] + folder_paths.get_filename_list("upscale_models"),
                ),
                "secondary_upscaler_name": (
                    ["None"] + folder_paths.get_filename_list("upscale_models"),
                ),
                "lora_1_name": (["None"] + folder_paths.get_filename_list("loras"),),
                "lora_2_name": (["None"] + folder_paths.get_filename_list("loras"),),
                "lora_3_name": (["None"] + folder_paths.get_filename_list("loras"),),
            },
        }

    RETURN_TYPES = ("PIPE",)
    CATEGORY = "Art Venture/Parameters"
    FUNCTION = "checkpoint_models_to_parameter_pipe"

    def checkpoint_models_to_parameter_pipe(
        self,
        ckpt_name,
        pipe: Dict = {},
        secondary_ckpt_name="None",
        vae_name="None",
        upscaler_name="None",
        secondary_upscaler_name="None",
        lora_1_name="None",
        lora_2_name="None",
        lora_3_name="None",
    ):
        pipe["ckpt_name"] = ckpt_name if ckpt_name != "None" else None
        pipe["secondary_ckpt_name"] = (
            secondary_ckpt_name if secondary_ckpt_name != "None" else None
        )
        pipe["vae_name"] = vae_name if vae_name != "None" else None
        pipe["upscaler_name"] = upscaler_name if upscaler_name != "None" else None
        pipe["secondary_upscaler_name"] = (
            secondary_upscaler_name if secondary_upscaler_name != "None" else None
        )
        pipe["lora_1_name"] = lora_1_name if lora_1_name != "None" else None
        pipe["lora_2_name"] = lora_2_name if lora_2_name != "None" else None
        pipe["lora_3_name"] = lora_3_name if lora_3_name != "None" else None
        return (pipe,)


class AVPromptsToParametersPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("STRING", {"multiline": True, "default": "Positive"}),
                "negative": ("STRING", {"multiline": True, "default": "Negative"}),
            },
            "optional": {
                "pipe": ("PIPE",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("PIPE",)
    CATEGORY = "Art Venture/Parameters"
    FUNCTION = "prompt_to_parameter_pipe"

    def prompt_to_parameter_pipe(
        self, positive, negative, pipe: Dict = {}, image=None, mask=None
    ):
        pipe["positive"] = positive
        pipe["negative"] = negative
        pipe["image"] = image
        pipe["mask"] = mask
        return (pipe,)


class AVParametersPipeToCheckpointModels:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE",),
            },
        }

    RETURN_TYPES = (
        "PIPE",
        "CHECKPOINT_NAME",
        "CHECKPOINT_NAME",
        "VAE_NAME",
        "UPSCALER_NAME",
        "UPSCALER_NAME",
        "LORA_NAME",
        "LORA_NAME",
        "LORA_NAME",
    )
    RETURN_NAMES = (
        "pipe",
        "ckpt_name",
        "secondary_ckpt_name",
        "vae_name",
        "upscaler_name",
        "secondary_upscaler_name",
        "lora_1_name",
        "lora_2_name",
        "lora_3_name",
    )
    CATEGORY = "Art Venture/Parameters"
    FUNCTION = "parameter_pipe_to_checkpoint_models"

    def parameter_pipe_to_checkpoint_models(self, pipe: Dict = {}):
        ckpt_name = pipe.get("ckpt_name", None)
        secondary_ckpt_name = pipe.get("secondary_ckpt_name", None)
        vae_name = pipe.get("vae_name", None)
        upscaler_name = pipe.get("upscaler_name", None)
        secondary_upscaler_name = pipe.get("secondary_upscaler_name", None)
        lora_1_name = pipe.get("lora_1_name", None)
        lora_2_name = pipe.get("lora_2_name", None)
        lora_3_name = pipe.get("lora_3_name", None)

        return (
            pipe,
            ckpt_name,
            secondary_ckpt_name,
            vae_name,
            upscaler_name,
            secondary_upscaler_name,
            lora_1_name,
            lora_2_name,
            lora_3_name,
        )


class AVParametersPipeToPrompts:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE",),
            },
        }

    RETURN_TYPES = (
        "PIPE",
        "STRING",
        "STRING",
        "IMAGE",
        "MASK",
    )
    RETURN_NAMES = (
        "pipe",
        "positive",
        "negative",
        "image",
        "mask",
    )
    CATEGORY = "Art Venture/Parameters"
    FUNCTION = "parameter_pipe_to_prompt"

    def parameter_pipe_to_prompt(self, pipe: Dict = {}):
        positive = pipe.get("positive", None)
        negative = pipe.get("negative", None)
        image = pipe.get("image", None)
        mask = pipe.get("mask", None)

        return (
            pipe,
            positive,
            negative,
            image,
            mask,
        )


NODE_CLASS_MAPPINGS = {
    "LoadImageFromUrl": UtilLoadImageFromUrl,
    "StringToInt": UtilStringToInt,
    "ImageMuxer": UtilImageMuxer,
    "ImageScaleDown": UtilImageScaleDown,
    "ImageScaleDownBy": UtilImageScaleDownBy,
    "DependenciesEdit": UtilDependenciesEdit,
    "AspectRatioSelector": UtilAspectRatioSelector,
    "SDXLAspectRatioSelector": UtilSDXLAspectRatioSelector,
    "SeedSelector": UtilSeedSelector,
    "AV_UploadImage": AVOutputUploadImage,
    "AV_CheckpointModelsToParametersPipe": AVCheckpointModelsToParametersPipe,
    "AV_PromptsToParametersPipe": AVPromptsToParametersPipe,
    "AV_ParametersPipeToCheckpointModels": AVParametersPipeToCheckpointModels,
    "AV_ParametersPipeToPrompts": AVParametersPipeToPrompts,
    "AV_VAELoader": AVVAELoader,
    "AV_LoraLoader": AVLoraLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageFromUrl": "Load Image From URL",
    "StringToInt": "String to Int",
    "ImageMuxer": "Image Muxer",
    "ImageScaleDown": "Image Scale Down",
    "ImageScaleDownBy": "Image Scale Down By",
    "DependenciesEdit": "Dependencies Edit",
    "AspectRatioSelector": "Aspect Ratio",
    "SDXLAspectRatioSelector": "SDXL Aspect Ratio",
    "SeedSelector": "Seed Selector",
    "AV_UploadImage": "Upload to Art Venture",
    "AV_CheckpointModelsToParametersPipe": "Checkpoint Models to Pipe",
    "AV_PromptsToParametersPipe": "Prompts to Pipe",
    "AV_ParametersPipeToCheckpointModels": "Pipe to Checkpoint Models",
    "AV_ParametersPipeToPrompts": "Pipe to Prompts",
    "AV_VAELoader": "VAE Loader",
    "AV_LoraLoader": "Lora Loader",
}


NODE_CLASS_MAPPINGS.update(SDXL_STYLER_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(SDXL_STYLER_NODE_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(BLIP_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(BLIP_NODE_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(FOOOCUS_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(FOOOCUS_NODE_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(PP_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(PP_NODE_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(CONTROLNET_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(CONTROLNET_NODE_DISPLAY_NAME_MAPPINGS)
