import os
from typing import Dict, Tuple

import folder_paths
import comfy.clip_vision
import comfy.controlnet
import comfy.utils

from .utils import load_module

custom_nodes = folder_paths.get_folder_paths("custom_nodes")
ip_adapter_dir_names = ["IPAdapter", "ComfyUI_IPAdapter_plus"]

model_dir = os.path.join(folder_paths.models_dir, "ip_adapter")
folder_paths.folder_names_and_paths["ip_adapter"] = ([model_dir], folder_paths.supported_pt_extensions)

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
    IPAdapter = nodes.get("IPAdapterApply")

    class AV_IPAdapter(IPAdapter):
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "ip_adapter_name": (["None"] + folder_paths.get_filename_list("ip_adapter"),),
                    "clip_name": (["None"] + folder_paths.get_filename_list("clip_vision"),),
                    "model": ("MODEL",),
                    "image": ("IMAGE",),
                    "weight": ("FLOAT", {"default": 1.0, "min": -1, "max": 3, "step": 0.05}),
                    "noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                },
                "optional": {
                    "ip_adapter_opt": ("IPADAPTER",),
                    "clip_vision_opt": ("CLIP_VISION",),
                    "enabled": ("BOOLEAN", {"default": True}),
                },
            }

        RETURN_TYPES = ("MODEL", "IPADAPTER", "CLIP_VISION")
        CATEGORY = "Art Venture/Loaders"
        FUNCTION = "load_id_adapter"

        def load_id_adapter(
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
        ):
            if not enabled:
                return (model, None, None)

            if ip_adapter_opt:
                ip_adapter = ip_adapter_opt
            else:
                assert ip_adapter_name != "None", "IP Adapter name must be specified"
                ip_adapter_path = folder_paths.get_full_path("ip_adapter", ip_adapter_name)
                ip_adapter = comfy.utils.load_torch_file(ip_adapter_path, safe_load=True)

            if clip_vision_opt:
                clip_vision = clip_vision_opt
            else:
                assert clip_name != "None", "Clip vision name must be specified"
                clip_path = folder_paths.get_full_path("clip_vision", clip_name)
                clip_vision = comfy.clip_vision.load(clip_path)

            res: Tuple = super().apply_ipadapter(
                ip_adapter, model, weight, clip_vision=clip_vision, image=image, noise=noise
            )
            res += (ip_adapter, clip_vision)

            return res

    NODE_CLASS_MAPPINGS["AV_IPAdapter"] = AV_IPAdapter
    NODE_DISPLAY_NAME_MAPPINGS["AV_IPAdapter"] = "IP Adapter"

except Exception as e:
    print(e)
