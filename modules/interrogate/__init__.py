from .blip_node import BlipLoader, BlipCaption, DownloadAndLoadBlip
from .danbooru import DeepDanbooruCaption

NODE_CLASS_MAPPINGS = {
    "BLIPLoader": BlipLoader,
    "BLIPCaption": BlipCaption,
    "DownloadAndLoadBlip": DownloadAndLoadBlip,
    "DeepDanbooruCaption": DeepDanbooruCaption,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BLIPLoader": "BLIP Loader",
    "BLIPCaption": "BLIP Caption",
    "DownloadAndLoadBlip": "Download and Load BLIP Model",
    "DeepDanbooruCaption": "Deep Danbooru Caption",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
