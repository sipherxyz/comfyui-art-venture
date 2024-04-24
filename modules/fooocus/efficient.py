import os
from typing import Dict

import folder_paths

from . import patch
from ..utils import load_module

custom_nodes = folder_paths.get_folder_paths("custom_nodes")
efficieny_dir_names = ["Efficiency", "efficiency-nodes-comfyui"]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    module_path = None

    for custom_node in custom_nodes:
        custom_node = (
            custom_node if not os.path.islink(custom_node) else os.readlink(custom_node)
        )
        for module_dir in efficieny_dir_names:
            if module_dir in os.listdir(custom_node):
                module_path = os.path.abspath(
                    os.path.join(custom_node, module_dir)
                )
                break

    if module_path is None:
        raise Exception("Could not find efficiency nodes")

    module = load_module(module_path)
    print("Loaded Efficiency nodes from", module_path)

    nodes: Dict = getattr(module, "NODE_CLASS_MAPPINGS")

    TSC_KSampler = nodes["KSampler (Efficient)"]
    TSC_KSamplerAdvanced = nodes["KSampler Adv. (Efficient)"]
    TSC_EfficientLoader = nodes["Efficient Loader"]

    class KSamplerEfficientWithSharpness(TSC_KSampler):
        @classmethod
        def INPUT_TYPES(cls):
            inputs = TSC_KSampler.INPUT_TYPES()
            inputs["optional"]["sharpness"] = (
                "FLOAT",
                {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.01},
            )

            return inputs

        CATEGORY = "Art Venture/Sampling"

        def sample(self, *args, sharpness=2.0, **kwargs):
            patch.sharpness = sharpness
            patch.patch_all()
            results = super().sample(*args, **kwargs)
            patch.unpatch_all()
            return results

    class KSamplerEfficientAdvancedWithSharpness(TSC_KSamplerAdvanced):
        @classmethod
        def INPUT_TYPES(cls):
            inputs = TSC_KSampler.INPUT_TYPES()
            inputs["optional"]["sharpness"] = (
                "FLOAT",
                {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.01},
            )

            return inputs

        CATEGORY = "Art Venture/Sampling"

        def sampleadv(self, *args, sharpness=2.0, **kwargs):
            patch.sharpness = sharpness
            patch.patch_all()
            results = super().sampleadv(*args, **kwargs)
            patch.unpatch_all()
            return results

    class AVCheckpointLoader(TSC_EfficientLoader):
        @classmethod
        def INPUT_TYPES(cls):
            inputs = TSC_EfficientLoader.INPUT_TYPES()
            inputs["optional"]["ckpt_override"] = ("STRING", {"default": "None"})
            inputs["optional"]["vae_override"] = ("STRING", {"default": "None"})
            inputs["optional"]["lora_override"] = ("STRING", {"default": "None"})
            return inputs

        CATEGORY = "Art Venture/Loaders"

        def efficientloader(
            self,
            ckpt_name,
            vae_name,
            clip_skip,
            lora_name,
            *args,
            ckpt_override="None",
            vae_override="None",
            lora_override="None",
            **kwargs
        ):
            if ckpt_override != "None":
                ckpt_name = ckpt_override
            if vae_override != "None":
                vae_name = vae_override
            if lora_override != "None":
                lora_name = lora_override

            return super().efficientloader(
                ckpt_name, vae_name, clip_skip, lora_name, *args, **kwargs
            )

    NODE_CLASS_MAPPINGS.update(
        {
            "Fooocus_KSamplerEfficient": KSamplerEfficientWithSharpness,
            "Fooocus_KSamplerEfficientAdvanced": KSamplerEfficientAdvancedWithSharpness,
            "AV_CheckpointLoader": AVCheckpointLoader,
        }
    )
    NODE_DISPLAY_NAME_MAPPINGS.update(
        {
            "Fooocus_KSamplerEfficient": "KSampler Efficient Fooocus",
            "Fooocus_KSamplerEfficientAdvanced": "KSampler Adv. Efficient Fooocus",
            "AV_CheckpointLoader": "Checkpoint Loader",
        }
    )


except Exception as e:
    print(e)
