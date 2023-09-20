from .blip_node import BlipCaption
from .danbooru import DeepDanbooruCaption

NODE_CLASS_MAPPINGS = {
    "BLIPCaption": BlipCaption,
    "DeepDanbooruCaption": DeepDanbooruCaption
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BLIPCaption": "BLIP Caption",
    "DeepDanbooruCaption": "Deep Danbooru Caption",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
