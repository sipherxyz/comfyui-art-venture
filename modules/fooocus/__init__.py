import os
from typing import Dict

import folder_paths
from nodes import KSampler, KSamplerAdvanced

from ..utils import load_module
from .patch import patch_all, unpatch_all

custom_nodes = folder_paths.get_folder_paths("custom_nodes")
efficieny_dir_names = ["Efficiency", "efficiency-nodes-comfyui"]

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = {}, {}

try:
    efficieny_path = None

    for custom_node in custom_nodes:
        for efficiency_dir in efficieny_dir_names:
            if efficiency_dir in os.listdir(custom_node):
                efficieny_path = os.path.join(custom_node, efficiency_dir)
                break

    if efficieny_path is None:
        raise Exception("Could not find efficiency nodes")

    module = load_module(efficieny_path)
    print("Loaded efficiency nodes from", efficieny_path)

    efficieny_nodes: Dict = getattr(module, "NODE_CLASS_MAPPINGS")

    TSC_KSampler = efficieny_nodes["KSampler (Efficient)"]
    TSC_KSamplerAdvanced = efficieny_nodes["KSampler Adv. (Efficient)"]

    class KSamplerWithSharpness(KSampler):
        @classmethod
        def INPUT_TYPES(cls):
            inputs = KSampler.INPUT_TYPES()
            inputs["optional"] = {
                "sharpness": (
                    "FLOAT",
                    {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.01},
                )
            }

            return inputs

        CATEGORY = "Art Venture/Sampling"

        def sample(self, *args, sharpness=2.0, **kwargs):
            patch.sharpness = sharpness
            patch_all()
            results = super().sample(*args, **kwargs)
            unpatch_all()
            return results

    class KSamplerAdvancedWithSharpness(KSamplerAdvanced):
        @classmethod
        def INPUT_TYPES(cls):
            inputs = KSamplerAdvanced.INPUT_TYPES()
            inputs["optional"] = {
                "sharpness": (
                    "FLOAT",
                    {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.01},
                )
            }

            return inputs

        CATEGORY = "Art Venture/Sampling"

        def sample(self, *args, sharpness=2.0, **kwargs):
            patch.sharpness = sharpness
            patch_all()
            results = super().sample(*args, **kwargs)
            unpatch_all()
            return results

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
            patch_all()
            results = super().sample(*args, **kwargs)
            unpatch_all()
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
            patch_all()
            results = super().sampleadv(*args, **kwargs)
            unpatch_all()
            return results

    NODE_CLASS_MAPPINGS.update(
        {
            "Fooocus_KSampler": KSamplerWithSharpness,
            "Fooocus_KSamplerAdvanced": KSamplerAdvancedWithSharpness,
            "Fooocus_KSamplerEfficient": KSamplerEfficientWithSharpness,
            "Fooocus_KSamplerEfficientAdvanced": KSamplerEfficientAdvancedWithSharpness,
        }
    )
    NODE_DISPLAY_NAME_MAPPINGS.update(
        {
            "Fooocus_KSampler": "KSampler Fooocus",
            "Fooocus_KSamplerAdvanced": "KSampler Adv. Fooocus",
            "Fooocus_KSamplerEfficient": "KSampler Efficient Fooocus",
            "Fooocus_KSamplerEfficientAdvanced": "KSampler Adv. Efficient Fooocus",
        }
    )


except Exception as e:
    print(e)
