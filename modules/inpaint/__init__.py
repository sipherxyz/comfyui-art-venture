from .sam.nodes import SAMLoader, GetSAMEmbedding, SAMEmbeddingToImage
from .lama import LaMaInpaint

NODE_CLASS_MAPPINGS = {
    "AV_SAMLoader": SAMLoader,
    "GetSAMEmbedding": GetSAMEmbedding,
    "SAMEmbeddingToImage": SAMEmbeddingToImage,
    "LaMaInpaint": LaMaInpaint,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AV_SAMLoader": "SAM Loader",
    "GetSAMEmbedding": "Get SAM Embedding",
    "SAMEmbeddingToImage": "SAM Embedding to Image",
    "LaMaInpaint": "LaMa Remove Object",
}
