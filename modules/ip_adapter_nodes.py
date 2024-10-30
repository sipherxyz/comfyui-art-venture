import os
from typing import Dict, Tuple

import folder_paths
import comfy.clip_vision
import comfy.controlnet
import comfy.utils
import comfy.model_management

from .utils import load_module

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
    IPAdapterModelLoader = nodes.get("IPAdapterModelLoader")
    IPAdapterSimple = nodes.get("IPAdapter")

    loader = IPAdapterModelLoader()
    apply = IPAdapterSimple()

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
        RETURN_NAMES = ("pipeline",)
        CATEGORY = "Art Venture/IP Adapter"
        FUNCTION = "load_ip_adapter"

        def load_ip_adapter(self, ip_adapter_name, clip_name):
            ip_adapter = loader.load_ipadapter_model(ip_adapter_name)[0]

            clip_path = folder_paths.get_full_path("clip_vision", clip_name)
            clip_vision = comfy.clip_vision.load(clip_path)

            pipeline = {"ipadapter": {"model": ip_adapter}, "clipvision": {"model": clip_vision}}
            return (pipeline,)

    class AV_IPAdapter:
        @classmethod
        def INPUT_TYPES(cls):
            inputs = IPAdapterSimple.INPUT_TYPES()

            return {
                "required": {
                    "ip_adapter_name": (["None"] + folder_paths.get_filename_list("ipadapter"),),
                    "clip_name": (["None"] + folder_paths.get_filename_list("clip_vision"),),
                    "model": ("MODEL",),
                    "image": ("IMAGE",),
                    "weight": ("FLOAT", {"default": 1.0, "min": -1, "max": 3, "step": 0.05}),
                },
                "optional": {
                    "ip_adapter_opt": ("IPADAPTER",),
                    "clip_vision_opt": ("CLIP_VISION",),
                    "attn_mask": ("MASK",),
                    "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                    "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                    "weight_type": inputs["required"]["weight_type"],
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

    NODE_CLASS_MAPPINGS.update(
        {
            "AV_IPAdapter": AV_IPAdapter,
            "AV_IPAdapterPipe": AV_IPAdapterPipe,
        }
    )
    NODE_DISPLAY_NAME_MAPPINGS.update(
        {
            "AV_IPAdapter": "IP Adapter Apply",
            "AV_IPAdapterPipe": "IP Adapter Pipe",
        }
    )

except Exception as e:
    print(e)
