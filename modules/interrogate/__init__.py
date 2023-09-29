from .blip_node import BlipLoader, BlipCaption
from .danbooru import DeepDanbooruCaption

NODE_CLASS_MAPPINGS = {
    "BLIPLoader": BlipLoader,
    "BLIPCaption": BlipCaption,
    "DeepDanbooruCaption": DeepDanbooruCaption,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BLIPLoader": "BLIP Loader",
    "BLIPCaption": "BLIP Caption",
    "DeepDanbooruCaption": "Deep Danbooru Caption",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
