from .nodes import GetSAMEmbedding, SAMEmbeddingToImage
from .lama import LaMaLoader, LaMaInpaint

NODE_CLASS_MAPPINGS = {
    "GetSAMEmbedding": GetSAMEmbedding,
    "SAMEmbeddingToImage": SAMEmbeddingToImage,
    "LaMaLoader": LaMaLoader,
    "LaMaInpaint": LaMaInpaint,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GetSAMEmbedding": "Get SAM Embedding",
    "SAMEmbeddingToImage": "SAM Embedding to Image",
    "LaMaLoader": "LaMa Loader",
    "LaMaInpaint": "LaMa Remove Object",
}
