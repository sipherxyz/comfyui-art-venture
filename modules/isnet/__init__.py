from .segmenter import ISNetLoader, ISNetSegment

NODE_CLASS_MAPPINGS = {
    "ISNetLoader": ISNetLoader,
    "ISNetSegment": ISNetSegment,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ISNetLoader": "ISNet Loader",
    "ISNetSegment": "ISNet Segment",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
