import io
import json
from typing import Dict

from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np

import folder_paths
from nodes import LoraLoader, VAELoader

from .logger import logger
from .utils import upload_to_av

from .utility_nodes import (
    NODE_CLASS_MAPPINGS as UTIL_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as UTIL_NODE_DISPLAY_NAME_MAPPINGS,
)
from .sdxl_prompt_styler import (
    NODE_CLASS_MAPPINGS as SDXL_STYLER_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as SDXL_STYLER_NODE_DISPLAY_NAME_MAPPINGS,
)
from .interrogate import (
    NODE_CLASS_MAPPINGS as INTERROGATE_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as INTERROGATE_NODE_DISPLAY_NAME_MAPPINGS,
)
from .fooocus import (
    NODE_CLASS_MAPPINGS as FOOOCUS_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as FOOOCUS_NODE_DISPLAY_NAME_MAPPINGS,
)
from .postprocessing import (
    NODE_CLASS_MAPPINGS as PP_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as PP_NODE_DISPLAY_NAME_MAPPINGS,
)
from .controlnet import (
    NODE_CLASS_MAPPINGS as CONTROLNET_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as CONTROLNET_NODE_DISPLAY_NAME_MAPPINGS,
)
from .animatediff import (
    NODE_CLASS_MAPPINGS as ANIMATEDIFF_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as ANIMATEDIFF_NODE_DISPLAY_NAME_MAPPINGS,
)
from .ip_adapter_nodes import (
    NODE_CLASS_MAPPINGS as IP_ADAPTER_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as IP_ADAPTER_NODE_DISPLAY_NAME_MAPPINGS,
)
from .isnet import (
    NODE_CLASS_MAPPINGS as ISNET_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as ISNET_NODE_DISPLAY_NAME_MAPPINGS,
)
from .inpaint import (
    NODE_CLASS_MAPPINGS as INPAINT_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as INPAINT_NODE_DISPLAY_NAME_MAPPINGS,
)


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
                print(f"Warning: Not found VAE model {vae_override}. Use {vae_name} instead.")
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
                print(f"Warning: Not found Lora model {lora_override}. Use {lora_name} instead.")
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
                "secondary_ckpt_name": (["None"] + folder_paths.get_filename_list("checkpoints"),),
                "vae_name": (["None"] + folder_paths.get_filename_list("vae"),),
                "upscaler_name": (["None"] + folder_paths.get_filename_list("upscale_models"),),
                "secondary_upscaler_name": (["None"] + folder_paths.get_filename_list("upscale_models"),),
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
        pipe["secondary_ckpt_name"] = secondary_ckpt_name if secondary_ckpt_name != "None" else None
        pipe["vae_name"] = vae_name if vae_name != "None" else None
        pipe["upscaler_name"] = upscaler_name if upscaler_name != "None" else None
        pipe["secondary_upscaler_name"] = secondary_upscaler_name if secondary_upscaler_name != "None" else None
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

    def prompt_to_parameter_pipe(self, positive, negative, pipe: Dict = {}, image=None, mask=None):
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
    "AV_UploadImage": AVOutputUploadImage,
    "AV_CheckpointModelsToParametersPipe": AVCheckpointModelsToParametersPipe,
    "AV_PromptsToParametersPipe": AVPromptsToParametersPipe,
    "AV_ParametersPipeToCheckpointModels": AVParametersPipeToCheckpointModels,
    "AV_ParametersPipeToPrompts": AVParametersPipeToPrompts,
    "AV_VAELoader": AVVAELoader,
    "AV_LoraLoader": AVLoraLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AV_UploadImage": "Upload to Art Venture",
    "AV_CheckpointModelsToParametersPipe": "Checkpoint Models to Pipe",
    "AV_PromptsToParametersPipe": "Prompts to Pipe",
    "AV_ParametersPipeToCheckpointModels": "Pipe to Checkpoint Models",
    "AV_ParametersPipeToPrompts": "Pipe to Prompts",
    "AV_VAELoader": "VAE Loader",
    "AV_LoraLoader": "Lora Loader",
}


NODE_CLASS_MAPPINGS.update(UTIL_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(UTIL_NODE_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(SDXL_STYLER_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(SDXL_STYLER_NODE_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(INTERROGATE_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(INTERROGATE_NODE_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(FOOOCUS_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(FOOOCUS_NODE_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(PP_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(PP_NODE_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(CONTROLNET_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(CONTROLNET_NODE_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(ANIMATEDIFF_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(ANIMATEDIFF_NODE_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(IP_ADAPTER_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(IP_ADAPTER_NODE_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(ISNET_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(ISNET_NODE_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(INPAINT_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(INPAINT_NODE_DISPLAY_NAME_MAPPINGS)
