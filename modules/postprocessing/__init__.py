from .color_blend import ColorBlend
from .color_correct import ColorCorrect

NODE_CLASS_MAPPINGS = {
    "ColorBlend": ColorBlend,
    "ColorCorrect": ColorCorrect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorBlend": "Color Blend",
    "ColorCorrect": "Color Correct",
}
