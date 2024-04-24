import os
import inspect
from typing import Dict

import folder_paths

from ..utils import load_module

custom_nodes = folder_paths.get_folder_paths("custom_nodes")
efficieny_dir_names = ["ImpactPack", "ComfyUI-Impact-Pack"]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    module_path = None

    for custom_node in custom_nodes:
        custom_node = custom_node if not os.path.islink(custom_node) else os.readlink(custom_node)
        for module_dir in efficieny_dir_names:
            if module_dir in os.listdir(custom_node):
                module_path = os.path.abspath(os.path.join(custom_node, module_dir))
                break

    if module_path is None:
        raise Exception("Could not find ImpactPack nodes")

    module = load_module(module_path)
    print("Loaded ImpactPack nodes from", module_path)

    nodes: Dict = getattr(module, "NODE_CLASS_MAPPINGS")
    FaceDetailer = nodes["FaceDetailer"]
    FaceDetailerPipe = nodes["FaceDetailerPipe"]

    class AV_FaceDetailer(FaceDetailer):
        @classmethod
        def INPUT_TYPES(s):
            inputs = FaceDetailer.INPUT_TYPES()
            inputs["optional"]["enabled"] = (
                "BOOLEAN",
                {"default": True, "label_on": "enabled", "label_off": "disabled"},
            )
            return inputs

        CATEGORY = "ArtVenture/Detailer"

        def args_to_pipe(self, args: dict):
            hook_args = [
                "model",
                "clip",
                "vae",
                "positive",
                "negative",
                "wildcard",
                "bbox_detector",
                "segm_detector_opt",
                "sam_model_opt",
                "detailer_hook",
            ]

            pipe_args = []
            for arg in hook_args:
                pipe_args.append(args.get(arg, None))

            return tuple(pipe_args + [None, None, None, None])

        def doit(self, image, *args, enabled=True, **kwargs):
            if enabled:
                return super().doit(image, *args, **kwargs)
            else:
                pipe = self.args_to_pipe(kwargs)
                return (image, [], [], None, pipe, [])

    class AV_FaceDetailerPipe(FaceDetailerPipe):
        @classmethod
        def INPUT_TYPES(s):
            inputs = FaceDetailerPipe.INPUT_TYPES()
            inputs["optional"]["enabled"] = (
                "BOOLEAN",
                {"default": True, "label_on": "enabled", "label_off": "disabled"},
            )
            return inputs

        CATEGORY = "ArtVenture/Detailer"

        def doit(self, image, detailer_pipe, *args, enabled=True, **kwargs):
            if enabled:
                return super().doit(image, detailer_pipe, *args, **kwargs)
            else:
                return (image, [], [], None, detailer_pipe, [])

    NODE_CLASS_MAPPINGS.update(
        {
            "AV_FaceDetailer": AV_FaceDetailer,
            "AV_FaceDetailerPipe": AV_FaceDetailerPipe,
        }
    )
    NODE_DISPLAY_NAME_MAPPINGS.update(
        {"AV_FaceDetailer": "FaceDetailer (AV)", "AV_FaceDetailerPipe": "FaceDetailerPipe (AV)"}
    )

except Exception as e:
    print("Could not load ImpactPack nodes", e)
