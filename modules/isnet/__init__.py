from .segmenter import ISNetSegment

NODE_CLASS_MAPPINGS = {
    "ISNetSegment": ISNetSegment,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ISNetSegment": "ISNet Segment",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
