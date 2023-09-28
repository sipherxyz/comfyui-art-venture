import os
import sys
from typing import Dict

import folder_paths
import comfy.clip_vision
import comfy.controlnet

from .utils import load_module

custom_nodes = folder_paths.get_folder_paths("custom_nodes")
ip_adapter_dir_names = ["IPAdapter", "IPAdapter-ComfyUI"]

folder_paths.folder_names_and_paths["ip_adapter"] = (
    [os.path.join(folder_paths.models_dir, "ip_adapter")],
    folder_paths.supported_pt_extensions,
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    module_path = None

    for custom_node in custom_nodes:
        custom_node = custom_node if not os.path.islink(custom_node) else os.readlink(custom_node)
        for module_dir in ip_adapter_dir_names:
            if module_dir in os.listdir(custom_node):
                module_path = os.path.abspath(os.path.join(custom_node, module_dir))
                if custom_node not in sys.path:
                    sys.path.append(custom_node)
                break

    if module_path is None:
        raise Exception("Could not find IPAdapter nodes")

    module = load_module(module_path)
    print("Loaded IPAdapter nodes from", module_path)

    nodes: Dict = getattr(module, "NODE_CLASS_MAPPINGS")
    IPAdapter = nodes["IPAdapter"]

    class AV_IPAdapter(IPAdapter):
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "model": ("MODEL",),
                    "image": ("IMAGE",),
                    "weight": (
                        "FLOAT",
                        {"default": 1, "min": -1, "max": 3, "step": 0.05},
                    ),
                    "model_name": (folder_paths.get_filename_list("ip_adapter"),),
                    "clip_name": (folder_paths.get_filename_list("clip_vision"),),
                    "dtype": (["fp16", "fp32"],),
                },
                "optional": {
                    "enabled": ("BOOLEAN", {"default": True}),
                    "mask": ("MASK",),
                },
            }

        CATEGORY = "Art Venture/Loaders"
        FUNCTION = "load_id_adapter"

        def load_id_adapter(self, model, image, *args, apply=True, clip_name="None", **kwargs):
            if not apply:
                return (model, None)

            clip_path = folder_paths.get_full_path("clip_vision", clip_name)
            clip_vision = comfy.clip_vision.load(clip_path)

            return super().load_id_adapter(model, image, clip_vision, *args, **kwargs)

    NODE_CLASS_MAPPINGS["AV_IPAdapter"] = AV_IPAdapter
    NODE_DISPLAY_NAME_MAPPINGS["AV_IPAdapter"] = "IP Adapter"

except Exception as e:
    print(e)
