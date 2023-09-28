import os
import cv2
import torch
import numpy as np

import folder_paths
import comfy.model_management as model_management

from ...image_utils import extract_img, dilate_mask
from ...utils import tensor2pil
from .models.lama import LaMa
from .schema import Config


models_path = folder_paths.models_dir
folder_paths.folder_names_and_paths["lama"] = (
    [os.path.join(models_path, "lama")],
    folder_paths.supported_pt_extensions,
)
LAMA_MODEL_URL = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
LAMA_MODEL_MD5 = "e3aa4aaa15225a33ec84f9f4bc47e500"


class LaMaLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("lama"),),
                "device_mode": (["AUTO", "Prefer GPU", "CPU"],),
            },
        }

    RETURN_TYPES = ("LAMA_MODEL",)
    CATEGORY = "Art Venture/Loaders"
    FUNCTION = "load_lama"

    def load_lama(self, model_name, device_mode):
        from .models.lama import LaMa

        model_path = folder_paths.get_full_path("lama", model_name)
        device = model_management.get_torch_device() if device_mode == "Prefer GPU" else "cpu"
        is_auto_mode = device_mode == "AUTO"
        model = LaMa(device, model_path=model_path, is_auto_mode=is_auto_mode)

        return (model,)


class LaMaInpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lama_model": ("LAMA_MODEL",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "mask_dialation": (
                    "INT",
                    {"min": 0, "max": 1000, "step": 1, "default": 0},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Art Venture/Inpainting"
    FUNCTION = "lama_inpaint"

    def lama_inpaint(
        self,
        lama_model: LaMa,
        image: torch.Tensor,
        mask: torch.Tensor,
        mask_dialation=0,
    ):
        image = tensor2pil(image)
        image, alpha_channel, exif_infos = extract_img(image, return_exif=True)

        mask = mask.cpu().numpy()
        if mask_dialation > 0:
            mask = dilate_mask(mask, mask_dialation)
        mask = np.clip(mask * 255, 0, 255).astype(np.uint8)

        config = Config()
        res_np_img = lama_model(image, mask, config)
        res_np_img = cv2.cvtColor(res_np_img.astype(np.uint8), cv2.COLOR_BGR2RGB)

        if alpha_channel is not None:
            if alpha_channel.shape[:2] != res_np_img.shape[:2]:
                alpha_channel = cv2.resize(alpha_channel, dsize=(res_np_img.shape[1], res_np_img.shape[0]))
            res_np_img = np.concatenate((res_np_img, alpha_channel[:, :, np.newaxis]), axis=-1)

        res_np_img = res_np_img.astype("float32") / 255
        res = torch.from_numpy(res_np_img).unsqueeze(0)

        return (res,)
