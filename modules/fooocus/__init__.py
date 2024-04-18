from nodes import KSampler, KSamplerAdvanced

from .patch import patch_all, unpatch_all
from .efficient import (
    NODE_CLASS_MAPPINGS as EFFICIENCY_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as EFFICIENCY_NODE_DISPLAY_NAME_MAPPINGS,
)


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


NODE_CLASS_MAPPINGS = {
    "Fooocus_KSampler": KSamplerWithSharpness,
    "Fooocus_KSamplerAdvanced": KSamplerAdvancedWithSharpness,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Fooocus_KSampler": "KSampler Fooocus",
    "Fooocus_KSamplerAdvanced": "KSampler Adv. Fooocus",
}

NODE_CLASS_MAPPINGS.update(EFFICIENCY_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(EFFICIENCY_NODE_DISPLAY_NAME_MAPPINGS)
