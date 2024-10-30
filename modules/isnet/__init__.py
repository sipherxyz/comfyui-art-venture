from .segmenter import ISNetLoader, ISNetSegment, DownloadISNetModel

NODE_CLASS_MAPPINGS = {
    "ISNetLoader": ISNetLoader,
    "ISNetSegment": ISNetSegment,
    "DownloadISNetModel": DownloadISNetModel,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ISNetLoader": "ISNet Loader",
    "ISNetSegment": "ISNet Segment",
    "DownloadISNetModel": "Download and Load ISNet Model",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
