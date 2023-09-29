from .nodes import GetSAMEmbedding, SAMEmbeddingToImage
from .lama.nodes import LaMaInpaint

NODE_CLASS_MAPPINGS = {
    "GetSAMEmbedding": GetSAMEmbedding,
    "SAMEmbeddingToImage": SAMEmbeddingToImage,
    "LaMaInpaint": LaMaInpaint,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GetSAMEmbedding": "Get SAM Embedding",
    "SAMEmbeddingToImage": "SAM Embedding to Image",
    "LaMaInpaint": "LaMa Remove Object",
}
