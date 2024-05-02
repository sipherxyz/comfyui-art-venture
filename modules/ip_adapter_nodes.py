import os
import json
import torch
from typing import Dict, Tuple, List
from pydantic import BaseModel

import folder_paths
import comfy.clip_vision
import comfy.controlnet
import comfy.utils
import comfy.model_management

from .utils import load_module, pil2tensor
from .utility_nodes import load_images_from_url

custom_nodes = folder_paths.get_folder_paths("custom_nodes")
ip_adapter_dir_names = ["IPAdapter", "ComfyUI_IPAdapter_plus"]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    module_path = None

    for custom_node in custom_nodes:
        custom_node = custom_node if not os.path.islink(custom_node) else os.readlink(custom_node)
        for module_dir in ip_adapter_dir_names:
            if module_dir in os.listdir(custom_node):
                module_path = os.path.abspath(os.path.join(custom_node, module_dir))
                break

    if module_path is None:
        raise Exception("Could not find IPAdapter nodes")

    module = load_module(module_path)
    print("Loaded IPAdapter nodes from", module_path)

    nodes: Dict = getattr(module, "NODE_CLASS_MAPPINGS")
    IPAdapterUnifiedLoader = nodes.get("IPAdapterUnifiedLoader")
    IPAdapterModelLoader = nodes.get("IPAdapterModelLoader")
    IPAdapterApply = nodes.get("IPAdapter")
    IPAdapterEncoder = nodes.get("IPAdapterEncoder")
    IPAdapterEmbeds = nodes.get("IPAdapterEmbeds")
    IPAdapterCombineEmbeds = nodes.get("IPAdapterCombineEmbeds")

    loader = IPAdapterModelLoader()
    unifyLoader = IPAdapterUnifiedLoader()
    apply = IPAdapterApply()
    encoder = IPAdapterEncoder()
    combiner = IPAdapterCombineEmbeds()
    embedder = IPAdapterEmbeds()

    WEIGHT_TYPES = [
        "linear",
        "ease in",
        "ease out",
        "ease in-out",
        "reverse in-out",
        "weak input",
        "weak output",
        "weak middle",
        "strong middle",
        "style transfer (SDXL)",
        "composition (SDXL)",
    ]

    PRESETS = [
        "LIGHT - SD1.5 only (low strength)",
        "STANDARD (medium strength)",
        "VIT-G (medium strength)",
        "PLUS (high strength)",
        "PLUS FACE (portraits)",
        "FULL FACE - SD1.5 only (portraits stronger)",
    ]

    class AV_IPAdapterPipe:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "ip_adapter_name": (folder_paths.get_filename_list("ipadapter"),),
                    "clip_name": (folder_paths.get_filename_list("clip_vision"),),
                }
            }

        RETURN_TYPES = ("IPADAPTER",)
        RETURN_NAMES = "pipeline"
        CATEGORY = "Art Venture/IP Adapter"
        FUNCTION = "load_ip_adapter"

        def load_ip_adapter(self, ip_adapter_name, clip_name):
            ip_adapter = loader.load_ipadapter_model(ip_adapter_name)[0]

            clip_path = folder_paths.get_full_path("clip_vision", clip_name)
            clip_vision = comfy.clip_vision.load(clip_path)

            pipeline = {"ipadapter": {"model": ip_adapter}, "clipvision": {"model": clip_vision}}
            return (pipeline,)

    class AV_IPAdapter(IPAdapterModelLoader, IPAdapterApply):
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "ip_adapter_name": (["None"] + folder_paths.get_filename_list("ipadapter"),),
                    "clip_name": (["None"] + folder_paths.get_filename_list("clip_vision"),),
                    "model": ("MODEL",),
                    "image": ("IMAGE",),
                    "weight": ("FLOAT", {"default": 1.0, "min": -1, "max": 3, "step": 0.05}),
                    "noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                },
                "optional": {
                    "ip_adapter_opt": ("IPADAPTER",),
                    "clip_vision_opt": ("CLIP_VISION",),
                    "attn_mask": ("MASK",),
                    "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                    "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                    "weight_type": (
                        ["standard", "prompt is more important", "style transfer (SDXL only)"],
                        {"default": "standard"},
                    ),
                    "enabled": ("BOOLEAN", {"default": True}),
                },
            }

        RETURN_TYPES = ("MODEL", "IPADAPTER", "CLIP_VISION")
        RETURN_NAMES = ("model", "pipeline", "clip_vision")
        CATEGORY = "Art Venture/IP Adapter"
        FUNCTION = "apply_ip_adapter"

        def apply_ip_adapter(
            self,
            ip_adapter_name,
            clip_name,
            model,
            image,
            weight,
            noise,
            ip_adapter_opt=None,
            clip_vision_opt=None,
            enabled=True,
            **kwargs,
        ):
            if not enabled:
                return (model, None, None)

            if ip_adapter_opt:
                if "ipadapter" in ip_adapter_opt:
                    ip_adapter = ip_adapter_opt["ipadapter"]["model"]
                else:
                    ip_adapter = ip_adapter_opt
            else:
                assert ip_adapter_name != "None", "IP Adapter name must be specified"
                ip_adapter = loader.load_ipadapter_model(ip_adapter_name)[0]

            if clip_vision_opt:
                clip_vision = clip_vision_opt
            elif ip_adapter_opt and "clipvision" in ip_adapter_opt:
                clip_vision = ip_adapter_opt["clipvision"]["model"]
            else:
                assert clip_name != "None", "Clip vision name must be specified"
                clip_path = folder_paths.get_full_path("clip_vision", clip_name)
                clip_vision = comfy.clip_vision.load(clip_path)

            pipeline = {"ipadapter": {"model": ip_adapter}, "clipvision": {"model": clip_vision}}

            res: Tuple = apply.apply_ipadapter(model, pipeline, image=image, weight=weight, **kwargs)
            res += (pipeline, clip_vision)

            return res

    class IPAdapterImage(BaseModel):
        url: str
        weight: float

    class IPAdapterData(BaseModel):
        images: List[IPAdapterImage]

    class AV_StyleApply:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "model": ("MODEL",),
                    "preset": (PRESETS,),
                    "data": (
                        "STRING",
                        {
                            "placeholder": '[{"url": "http://domain/path/image.png", "weight": 1}]',
                            "multiline": True,
                            "dynamicPrompts": False,
                        },
                    ),
                    "weight": ("FLOAT", {"default": 0.5, "min": -1, "max": 3, "step": 0.05}),
                    "weight_type": (WEIGHT_TYPES,),
                    "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                    "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                },
                "optional": {
                    "mask": ("MASK",),
                    "enabled": ("BOOLEAN", {"default": True}),
                },
            }

        RETURN_TYPES = ("MODEL", "IMAGE")
        CATEGORY = "Art Venture/Style"
        FUNCTION = "apply_style"

        def apply_style(self, model, preset: str, data: str, mask=None, enabled=True, **kwargs):
            data = json.loads(data or "[]")
            data: IPAdapterData = IPAdapterData(images=data)  # validate

            if len(data.images) == 0:
                images = torch.zeros((1, 64, 64, 3))
                return (model, images)

            (model, pipeline) = unifyLoader.load_models(model, preset)

            urls = [image.url for image in data.images]
            pils, _ = load_images_from_url(urls)

            embeds_avg = None
            neg_embeds_avg = None
            images = []

            for i, pil in enumerate(pils):
                weight = data.images[i].weight
                image = pil2tensor(pil)
                if i > 0 and image.shape[1:] != images[0].shape[1:]:
                    image = comfy.utils.common_upscale(
                        image.movedim(-1, 1), images[0].shape[2], images[0].shape[1], "bilinear", "center"
                    ).movedim(1, -1)
                images.append(image)

                embeds = encoder.encode(pipeline, image, weight, mask=mask)
                if embeds_avg is None:
                    embeds_avg = embeds[0]
                    neg_embeds_avg = embeds[1]
                else:
                    embeds_avg = combiner.batch(embeds_avg, method="average", embed2=embeds[0])[0]
                    neg_embeds_avg = combiner.batch(neg_embeds_avg, method="average", embed2=embeds[1])[0]

            images = torch.cat(images)

            model = embedder.apply_ipadapter(model, pipeline, embeds_avg, neg_embed=neg_embeds_avg, **kwargs)[0]

            return (model, images)

    NODE_CLASS_MAPPINGS.update(
        {"AV_IPAdapter": AV_IPAdapter, "AV_IPAdapterPipe": AV_IPAdapterPipe, "AV_StyleApply": AV_StyleApply}
    )
    NODE_DISPLAY_NAME_MAPPINGS.update(
        {
            "AV_IPAdapter": "IP Adapter Apply",
            "AV_IPAdapterPipe": "IP Adapter Pipe",
            "AV_StyleApply": "AV Style Apply",
        }
    )

except Exception as e:
    print(e)
