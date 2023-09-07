import os
import sys
import math
from typing import Dict, List

import folder_paths
import comfy.sd
import comfy.controlnet
from nodes import ControlNetLoader, ControlNetApply

from .utils import load_module

custom_nodes = folder_paths.get_folder_paths("custom_nodes")
preprocessors_dir_names = ["ControlNetPreprocessors", "comfyui_controlnet_aux"]

control_net_preprocessors = {}

try:
    module_path = None

    for custom_node in custom_nodes:
        custom_node = (
            custom_node if not os.path.islink(custom_node) else os.readlink(custom_node)
        )
        for module_dir in preprocessors_dir_names:
            if module_dir in os.listdir(custom_node):
                module_path = os.path.abspath(os.path.join(custom_node, module_dir))
                if custom_node not in sys.path:
                    sys.path.append(custom_node)
                break

    if module_path is None:
        raise Exception("Could not find ControlNetPreprocessors nodes")

    module = load_module(module_path)
    print("Loaded ControlNetPreprocessors nodes from", module_path)

    nodes: Dict = getattr(module, "NODE_CLASS_MAPPINGS")

    if "CannyEdgePreprocessor" in nodes:
        control_net_preprocessors["canny"] = (
            nodes["CannyEdgePreprocessor"],
            [100, 200],
        )
    if "LineArtPreprocessor" in nodes:
        control_net_preprocessors["lineart"] = (
            nodes["LineArtPreprocessor"],
            ["disable"],
        )
    if "AnimeLineArtPreprocessor" in nodes:
        control_net_preprocessors["lineart-anime"] = (
            nodes["AnimeLineArtPreprocessor"],
            [],
        )
    if "ScribblePreprocessor" in nodes:
        control_net_preprocessors["scribble"] = (nodes["ScribblePreprocessor"], [])
    if "HEDPreprocessor" in nodes:
        control_net_preprocessors["hed"] = (nodes["HEDPreprocessor"], ["enable"])
    if "PiDiNetPreprocessor" in nodes:
        control_net_preprocessors["pidinet"] = (
            nodes["PiDiNetPreprocessor"],
            ["enable"],
        )
    if "M-LSDPreprocessor" in nodes:
        control_net_preprocessors["mlsd"] = (nodes["M-LSDPreprocessor"], [0.1, 0.1])
    if "OpenposePreprocessor" in nodes:
        control_net_preprocessors["openpose"] = (
            nodes["OpenposePreprocessor"],
            ["enable", "enable", "enable"],
        )
    if "DWPreprocessor" in nodes:
        control_net_preprocessors["dwpose"] = (
            nodes["DWPreprocessor"],
            ["enable", "enable", "enable"],
        )
    if "BAE-NormalMapPreprocessor" in nodes:
        control_net_preprocessors["normalmap-bae"] = (
            nodes["BAE-NormalMapPreprocessor"],
            [],
        )
    if "MiDaS-NormalMapPreprocessor" in nodes:
        control_net_preprocessors["normalmap-midas"] = (
            nodes["MiDaS-NormalMapPreprocessor"],
            [math.pi * 2.0, 0.1],
        )
    if "MiDaS-DepthMapPreprocessor" in nodes:
        control_net_preprocessors["depth-midas"] = (
            nodes["MiDaS-DepthMapPreprocessor"],
            [math.pi * 2.0, 0.4],
        )
    if "Zoe-DepthMapPreprocessor" in nodes:
        control_net_preprocessors["depth"] = (nodes["Zoe-DepthMapPreprocessor"], [])

except Exception as e:
    print(e)


def load_controlnet(control_net_name, control_net_override="None"):
    if control_net_override != "None":
        if control_net_override not in folder_paths.get_filename_list("controlnet"):
            print(
                f"Warning: Not found ControlNet model {control_net_override}. Use {control_net_name} instead."
            )
        else:
            control_net_name = control_net_override

    if control_net_name == "None":
        return None

    controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
    return comfy.controlnet.load_controlnet(controlnet_path)


def apply_preprocessor(image, preprocessor):
    if preprocessor == "None":
        return image

    if preprocessor not in control_net_preprocessors:
        raise Exception(f"Preprocessor {preprocessor} not found")

    preprocessor_class, default_args = control_net_preprocessors[preprocessor]
    default_args: List = default_args.copy()
    default_args.insert(0, image)
    preprocessor_args = {
        key: default_args[i]
        for i, key in enumerate(preprocessor_class.INPUT_TYPES()["required"].keys())
    }
    function_name = preprocessor_class.FUNCTION
    image = getattr(preprocessor_class(), function_name)(**preprocessor_args)[0]

    return image


class AVControlNetLoader(ControlNetLoader):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control_net_name": (folder_paths.get_filename_list("controlnet"),)
            },
            "optional": {"control_net_override": ("STRING", {"default": "None"})},
        }

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_controlnet"
    CATEGORY = "Art Venture/Loaders"

    def load_controlnet(self, control_net_name, control_net_override="None"):
        return load_controlnet(control_net_name, control_net_override)


class AVControlNetEfficientStacker:
    controlnets = folder_paths.get_filename_list("controlnet")
    preprocessors = list(control_net_preprocessors.keys())

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control_net_name": (["None"] + s.controlnets,),
                "image": ("IMAGE",),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
                "preprocessor": (["None"] + s.preprocessors,),
            },
            "optional": {
                "cnet_stack": ("CONTROL_NET_STACK",),
                "control_net_override": ("STRING", {"default": "None"}),
            },
        }

    RETURN_TYPES = ("CONTROL_NET_STACK",)
    RETURN_NAMES = ("CNET_STACK",)
    FUNCTION = "control_net_stacker"
    CATEGORY = "Art Venture/Loaders"

    def control_net_stacker(
        self,
        control_net_name,
        image,
        strength,
        preprocessor,
        cnet_stack=None,
        control_net_override="None",
    ):
        # If control_net_stack is None, initialize as an empty list
        if cnet_stack is None:
            cnet_stack = []

        control_net = load_controlnet(control_net_name, control_net_override)

        # Extend the control_net_stack with the new tuple
        if control_net is not None:
            image = apply_preprocessor(image, preprocessor)
            cnet_stack.extend([(control_net, image, strength)])

        return (cnet_stack,)


class AVControlNetEfficientLoader(ControlNetApply):
    controlnets = folder_paths.get_filename_list("controlnet")
    preprocessors = list(control_net_preprocessors.keys())

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control_net_name": (["None"] + s.controlnets,),
                "conditioning": ("CONDITIONING",),
                "image": ("IMAGE",),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
                "preprocessor": (["None"] + s.preprocessors,),
            },
            "optional": {"control_net_override": ("STRING", {"default": "None"})},
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "load_controlnet"
    CATEGORY = "Art Venture/Loaders"

    def load_controlnet(
        self,
        control_net_name,
        conditioning,
        image,
        strength,
        preprocessor,
        control_net_override="None",
    ):
        control_net = load_controlnet(control_net_name, control_net_override)
        if control_net is None:
            return (conditioning,)

        image = apply_preprocessor(image, preprocessor)

        return super().apply_controlnet(conditioning, control_net, image, strength)


NODE_CLASS_MAPPINGS = {
    "AV_ControlNetLoader": AVControlNetLoader,
    "AV_ControlNetEfficientLoader": AVControlNetEfficientLoader,
    "AV_ControlNetEfficientStacker": AVControlNetEfficientStacker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AV_ControlNetLoader": "ControlNet Loader",
    "AV_ControlNetEfficientLoader": "ControlNet Loader (Efficient)",
    "AV_ControlNetEfficientStacker": "ControlNet Stacker (Efficient)",
}
