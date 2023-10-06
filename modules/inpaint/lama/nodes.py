import os
import cv2
import torch
import numpy as np

import folder_paths
import comfy.model_management as model_management

from ...utils import tensor2pil
from ...image_utils import extract_img, dilate_mask
from ...model_utils import download_model
from .models.lama import LaMa
from .schema import Config


lama = None
gpu = model_management.get_torch_device()
cpu = torch.device("cpu")
model_dir = os.path.join(folder_paths.models_dir, "lama")
model_url = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"


def load_lama_model(device_mode):
    device = gpu if device_mode != "CPU" else cpu

    global lama
    if lama is None:
        files = download_model(
            model_path=model_dir,
            model_url=model_url,
            ext_filter=[".pt"],
            download_name="big-lama.pt",
        )

        from .models.lama import LaMa

        lama = LaMa(device, model_path=files[0])

    lama = lama.to(device)
    return lama


def unload_lama(device_mode):
    global lama
    if lama is not None and device_mode == "AUTO":
        lama = lama.to(cpu)

    model_management.soft_empty_cache()


class LaMaInpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "mask_dialation": (
                    "INT",
                    {"min": 0, "max": 1000, "step": 1, "default": 0},
                ),
            },
            "optional": {"device_mode": (["AUTO", "Prefer GPU", "CPU"],)},
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Art Venture/Inpainting"
    FUNCTION = "lama_inpaint"

    def lama_inpaint(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        mask_dialation=0,
        device_mode="AUTO",
    ):
        if image.shape[0] != mask.shape[0]:
            raise Exception("Image and mask must have the same batch size")

        model = load_lama_model(device_mode)
        config = Config()

        try:
            inpainted = []

            for i, img in enumerate(image):
                img = tensor2pil(img)
                img, alpha_channel, _ = extract_img(img, return_exif=True)

                msk = mask[i].cpu().numpy()
                if mask_dialation > 0:
                    msk = dilate_mask(msk, mask_dialation)
                msk = np.clip(msk * 255, 0, 255).astype(np.uint8)

                res_np_img = model(img, msk, config)
                res_np_img = cv2.cvtColor(res_np_img.astype(np.uint8), cv2.COLOR_BGR2RGB)

                if alpha_channel is not None:
                    if alpha_channel.shape[:2] != res_np_img.shape[:2]:
                        alpha_channel = cv2.resize(alpha_channel, dsize=(res_np_img.shape[1], res_np_img.shape[0]))
                    res_np_img = np.concatenate((res_np_img, alpha_channel[:, :, np.newaxis]), axis=-1)

                res_np_img = res_np_img.astype("float32") / 255
                res = torch.from_numpy(res_np_img).unsqueeze(0)
                inpainted.append(res)

            return (torch.cat(inpainted, dim=0),)
        finally:
            unload_lama(device_mode)
