import numpy as np
from PIL import Image
from segment_anything import SamPredictor


import comfy.model_management

from ..utils import tensor2pil, pil2tensor


class GetSAMEmbedding:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sam_model": ("SAM_MODEL",),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("SAM_EMBEDDING",)
    CATEGORY = "Art Venture/Segmentation"
    FUNCTION = "get_sam_embedding"

    def get_sam_embedding(self, image, sam_model):
        if sam_model.is_auto_mode:
            device = comfy.model_management.get_torch_device()
            sam_model.to(device=device)

        try:
            predictor = SamPredictor(sam_model)
            image = tensor2pil(image)
            image = image.convert("RGB")
            image = np.array(image)
            predictor.set_image(image, "RGB")
            embedding = predictor.get_image_embedding().cpu().numpy()

            return (embedding,)
        finally:
            if sam_model.is_auto_mode:
                sam_model.to(device="cpu")


class SAMEmbeddingToImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "embedding": ("SAM_EMBEDDING",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Art Venture/Segmentation"
    FUNCTION = "sam_embedding_to_noise_image"

    def sam_embedding_to_noise_image(self, embedding: np.ndarray):
        # Flatten the array to a 1D array
        flat_arr = embedding.flatten()
        # Convert the 1D array to bytes
        bytes_arr = flat_arr.astype(np.float32).tobytes()
        # Convert bytes to RGBA PIL Image
        size = (embedding.shape[1], embedding.shape[2] * embedding.shape[3])

        img = Image.frombytes("RGBA", size, bytes_arr)

        return (pil2tensor(img),)
