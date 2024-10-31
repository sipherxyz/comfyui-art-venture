from typing import List

import folder_paths
from nodes import ControlNetLoader, ControlNetApply, ControlNetApplyAdvanced

from .preprocessor import preprocessors, apply_preprocessor
from .advanced import comfy_load_controlnet


def load_controlnet(control_net_name, control_net_override="None", timestep_keyframe=None):
    if control_net_override != "None":
        if control_net_override not in folder_paths.get_filename_list("controlnet"):
            print(f"Warning: Not found ControlNet model {control_net_override}. Use {control_net_name} instead.")
        else:
            control_net_name = control_net_override

    if control_net_name == "None":
        return None

    return comfy_load_controlnet(control_net_name, timestep_keyframe=timestep_keyframe)


def detect_controlnet(preprocessor: str, sd_version: str):
    controlnets = folder_paths.get_filename_list("controlnet")
    controlnets = filter(lambda x: sd_version in x, controlnets)
    if sd_version == "sdxl":
        controlnets = filter(lambda x: "t2i" not in x, controlnets)
        controlnets = filter(lambda x: "lllite" not in x, controlnets)

    control_net_name = "None"
    if preprocessor in {"canny", "scribble", "mlsd"}:
        control_net_name = next((c for c in controlnets if preprocessor in c), "None")
    if preprocessor in {"scribble", "scribble_hed"}:
        control_net_name = next((c for c in controlnets if "scribble" in c), "None")
    if preprocessor in {"lineart", "lineart_coarse"}:
        control_net_name = next((c for c in controlnets if "lineart." in c), "None")
    if preprocessor in {"lineart_anime", "lineart_manga"}:
        control_net_name = next((c for c in controlnets if "lineart_anime" in c), "None")
    if preprocessor in {"hed", "hed_safe", "pidi", "pidi_safe"}:
        control_net_name = next((c for c in controlnets if "softedge" in c), "None")
    if preprocessor in {"pose", "openpose", "dwpose"}:
        control_net_name = next((c for c in controlnets if "openpose" in c), "None")
    if preprocessor in {"normalmap_bae", "normalmap_midas"}:
        control_net_name = next((c for c in controlnets if "normalbae" in c), "None")
    if preprocessor in {"depth", "depth_midas", "depth_zoe"}:
        control_net_name = next((c for c in controlnets if "depth" in c), "None")
    if preprocessor in {"seg_ofcoco", "seg_ofade20k", "seg_ufade20k"}:
        control_net_name = next((c for c in controlnets if "seg" in c), "None")

    if preprocessor in {"tile"}:
        control_net_name = next((c for c in controlnets if "tile" in c), "None")

    return control_net_name


class AVControlNetLoader(ControlNetLoader):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"control_net_name": (folder_paths.get_filename_list("controlnet"),)},
            "optional": {
                "control_net_override": ("STRING", {"default": "None"}),
                "timestep_keyframe": ("TIMESTEP_KEYFRAME",),
            },
        }

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_controlnet"
    CATEGORY = "Art Venture/Loaders"

    def load_controlnet(self, control_net_name, control_net_override="None", timestep_keyframe=None):
        return load_controlnet(control_net_name, control_net_override, timestep_keyframe=timestep_keyframe)


class AV_ControlNetPreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "preprocessor": (["None"] + preprocessors,),
                "sd_version": (["sd15", "sdxl"],),
            },
            "optional": {
                "resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "preprocessor_override": ("STRING", {"default": "None"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("IMAGE", "CNET_NAME")
    FUNCTION = "detect_controlnet"
    CATEGORY = "Art Venture/Loaders"

    def detect_controlnet(self, image, preprocessor, sd_version, resolution=512, preprocessor_override="None"):
        if preprocessor_override != "None":
            if preprocessor_override not in preprocessors:
                print(
                    f"Warning: Not found ControlNet preprocessor {preprocessor_override}. Use {preprocessor} instead."
                )
            else:
                preprocessor = preprocessor_override

        image = apply_preprocessor(image, preprocessor, resolution=resolution)
        control_net_name = detect_controlnet(preprocessor, sd_version)

        return (image, control_net_name)


class AVControlNetEfficientStacker:
    controlnets = folder_paths.get_filename_list("controlnet")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control_net_name": (["None", "Auto: sd15", "Auto: sdxl", "Auto: sdxl_t2i"] + s.controlnets,),
                "image": ("IMAGE",),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "preprocessor": (["None"] + preprocessors,),
            },
            "optional": {
                "cnet_stack": ("CONTROL_NET_STACK",),
                "control_net_override": ("STRING", {"default": "None"}),
                "timestep_keyframe": ("TIMESTEP_KEYFRAME",),
                "resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "enabled": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("CONTROL_NET_STACK",)
    RETURN_NAMES = ("CNET_STACK",)
    FUNCTION = "control_net_stacker"
    CATEGORY = "Art Venture/Loaders"

    def control_net_stacker(
        self,
        control_net_name: str,
        image,
        strength,
        start_percent,
        end_percent,
        preprocessor,
        cnet_stack=None,
        control_net_override="None",
        timestep_keyframe=None,
        resolution=512,
        enabled=True,
    ):
        if not enabled:
            return (cnet_stack,)

        # If control_net_stack is None, initialize as an empty list
        if cnet_stack is None:
            cnet_stack = []

        if control_net_name.startswith("Auto: "):
            assert preprocessor != "None", "preprocessor must be set when using Auto mode"

            sd_version = control_net_name[len("Auto: ") :]
            control_net_name = detect_controlnet(preprocessor, sd_version)

        control_net = load_controlnet(control_net_name, control_net_override, timestep_keyframe=timestep_keyframe)

        # Extend the control_net_stack with the new tuple
        if control_net is not None:
            image = apply_preprocessor(image, preprocessor, resolution=resolution)
            cnet_stack.extend([(control_net, image, strength, start_percent, end_percent)])

        return (cnet_stack,)


class AVControlNetEfficientStackerSimple(AVControlNetEfficientStacker):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control_net_name": (["None", "Auto: sd15", "Auto: sdxl", "Auto: sdxl_t2i"] + s.controlnets,),
                "image": ("IMAGE",),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
                "preprocessor": (["None"] + preprocessors,),
            },
            "optional": {
                "cnet_stack": ("CONTROL_NET_STACK",),
                "control_net_override": ("STRING", {"default": "None"}),
                "timestep_keyframe": ("TIMESTEP_KEYFRAME",),
                "resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "enabled": ("BOOLEAN", {"default": True}),
            },
        }

    FUNCTION = "control_net_stacker_simple"

    def control_net_stacker_simple(
        self,
        *args,
        **kwargs,
    ):
        return self.control_net_stacker(*args, start_percent=0.0, end_percent=1.0, **kwargs)


class AVControlNetEfficientLoader(ControlNetApply):
    controlnets = folder_paths.get_filename_list("controlnet")

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
                "preprocessor": (["None"] + preprocessors,),
            },
            "optional": {
                "control_net_override": ("STRING", {"default": "None"}),
                "timestep_keyframe": ("TIMESTEP_KEYFRAME",),
                "resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "enabled": ("BOOLEAN", {"default": True}),
            },
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
        timestep_keyframe=None,
        resolution=512,
        enabled=True,
    ):
        if not enabled:
            return (conditioning,)

        control_net = load_controlnet(control_net_name, control_net_override, timestep_keyframe=timestep_keyframe)
        if control_net is None:
            return (conditioning,)

        image = apply_preprocessor(image, preprocessor, resolution=resolution)

        return super().apply_controlnet(conditioning, control_net, image, strength)


class AVControlNetEfficientLoaderAdvanced(ControlNetApplyAdvanced):
    controlnets = folder_paths.get_filename_list("controlnet")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control_net_name": (["None"] + s.controlnets,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "image": ("IMAGE",),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "preprocessor": (["None"] + preprocessors,),
            },
            "optional": {
                "control_net_override": ("STRING", {"default": "None"}),
                "timestep_keyframe": ("TIMESTEP_KEYFRAME",),
                "resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "enabled": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "load_controlnet"
    CATEGORY = "Art Venture/Loaders"

    def load_controlnet(
        self,
        control_net_name,
        positive,
        negative,
        image,
        strength,
        start_percent,
        end_percent,
        preprocessor,
        control_net_override="None",
        timestep_keyframe=None,
        resolution=512,
        enabled=True,
    ):
        if not enabled:
            return (positive, negative)

        control_net = load_controlnet(control_net_name, control_net_override, timestep_keyframe=timestep_keyframe)
        if control_net is None:
            return (positive, negative)

        image = apply_preprocessor(image, preprocessor, resolution=resolution)

        return super().apply_controlnet(positive, negative, control_net, image, strength, start_percent, end_percent)


NODE_CLASS_MAPPINGS = {
    "AV_ControlNetLoader": AVControlNetLoader,
    "AV_ControlNetEfficientLoader": AVControlNetEfficientLoader,
    "AV_ControlNetEfficientLoaderAdvanced": AVControlNetEfficientLoaderAdvanced,
    "AV_ControlNetEfficientStacker": AVControlNetEfficientStacker,
    "AV_ControlNetEfficientStackerSimple": AVControlNetEfficientStackerSimple,
    "AV_ControlNetPreprocessor": AV_ControlNetPreprocessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AV_ControlNetLoader": "ControlNet Loader",
    "AV_ControlNetEfficientLoader": "ControlNet Loader",
    "AV_ControlNetEfficientLoaderAdvanced": "ControlNet Loader Adv.",
    "AV_ControlNetEfficientStacker": "ControlNet Stacker Adv.",
    "AV_ControlNetEfficientStackerSimple": "ControlNet Stacker",
    "AV_ControlNetPreprocessor": "ControlNet Preprocessor",
}
