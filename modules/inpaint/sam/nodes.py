import os
import torch
import numpy as np
from PIL import Image

import folder_paths
import comfy.model_management as model_management
import comfy.utils

from ...utils import ensure_package, tensor2pil, pil2tensor

if "sams" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["sams"] = (
        [
            os.path.join(folder_paths.models_dir, "sams"),
        ],
        folder_paths.supported_pt_extensions,
    )

gpu = model_management.get_torch_device()
cpu = torch.device("cpu")


class SAMLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("sams"),),
            }
        }

    RETURN_TYPES = ("AV_SAM_MODEL",)
    RETURN_NAMES = ("sam_model",)
    FUNCTION = "load_model"
    CATEGORY = "Art Venture/Segmentation"

    def load_model(self, model_name):
        modelname = folder_paths.get_full_path("sams", model_name)

        state_dict = comfy.utils.load_torch_file(modelname)
        encoder_size = state_dict["image_encoder.patch_embed.proj.bias"].shape[0]

        if encoder_size == 1280:
            model_kind = "vit_h"
        elif encoder_size == 1024:
            model_kind = "vit_l"
        else:
            model_kind = "vit_b"

        ensure_package("segment_anything")
        from segment_anything import sam_model_registry

        sam = sam_model_registry[model_kind]()
        sam.load_state_dict(state_dict)

        return (sam,)


class GetSAMEmbedding:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sam_model": ("AV_SAM_MODEL",),
                "image": ("IMAGE",),
            },
            "optional": {"device_mode": (["AUTO", "Prefer GPU", "CPU"],)},
        }

    RETURN_TYPES = ("SAM_EMBEDDING",)
    CATEGORY = "Art Venture/Segmentation"
    FUNCTION = "get_sam_embedding"

    def get_sam_embedding(self, image, sam_model, device_mode="AUTO"):
        device = gpu if device_mode != "CPU" else cpu
        sam_model.to(device)

        ensure_package("segment_anything")
        from segment_anything import SamPredictor

        try:
            predictor = SamPredictor(sam_model)
            image = tensor2pil(image)
            image = image.convert("RGB")
            image = np.array(image)
            predictor.set_image(image, "RGB")
            embedding = predictor.get_image_embedding().cpu().numpy()

            return (embedding,)
        finally:
            if device_mode == "AUTO":
                sam_model.to(cpu)


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
        size = (embedding.shape[1] * 4, int(embedding.shape[2] * embedding.shape[3] / 4))

        img = Image.frombytes("RGBA", size, bytes_arr)

        return (pil2tensor(img),)
