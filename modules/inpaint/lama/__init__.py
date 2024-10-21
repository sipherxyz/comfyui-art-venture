# https://github.com/advimman/lama
import os
import yaml
import torch
import torch.nn.functional as F

import folder_paths
import comfy.model_management as model_management

from ...model_utils import download_model
from ...utils import ensure_package


lama = None
gpu = model_management.get_torch_device()
cpu = torch.device("cpu")
model_dir = os.path.join(folder_paths.models_dir, "lama")
model_url = "https://d111kwgh87c0gj.cloudfront.net/stable-diffusion/lama/big-lama.pt"
config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.yaml")


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def pad_tensor_to_modulo(img, mod):
    height, width = img.shape[-2:]
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return F.pad(img, pad=(0, out_width - width, 0, out_height - height), mode="reflect")


def load_model():
    global lama
    if lama is None:
        ensure_package("omegaconf")

        from omegaconf import OmegaConf
        from .saicinpainting.training.trainers import load_checkpoint

        files = download_model(
            model_path=model_dir,
            model_url=model_url,
            ext_filter=[".pt"],
            download_name="big-lama.pt",
        )

        cfg = yaml.safe_load(open(config_path, "rt"))
        cfg = OmegaConf.create(cfg)
        cfg.training_model.predict_only = True
        cfg.visualizer.kind = "noop"

        lama = load_checkpoint(cfg, files[0], strict=False, map_location="cpu")
        lama.freeze()

    return lama


class LaMaInpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
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
        device_mode="AUTO",
    ):
        if image.shape[0] != mask.shape[0]:
            raise Exception("Image and mask must have the same batch size")

        device = gpu if device_mode != "CPU" else cpu

        model = load_model()
        model.to(device)

        try:
            inpainted = []
            orig_w = image.shape[2]
            orig_h = image.shape[1]

            for i, img in enumerate(image):
                img = img.permute(2, 0, 1).unsqueeze(0)
                msk = mask[i].detach().cpu()
                msk = (msk > 0) * 1.0
                msk = msk.unsqueeze(0).unsqueeze(0)

                batch = {}
                batch["image"] = pad_tensor_to_modulo(img, 8).to(device)
                batch["mask"] = pad_tensor_to_modulo(msk, 8).to(device)

                res = model(batch)
                res = batch["inpainted"][0].permute(1, 2, 0)
                res = res.detach().cpu()
                res = res[:orig_h, :orig_w]

                inpainted.append(res)

            return (torch.stack(inpainted),)
        finally:
            if device_mode == "AUTO":
                model.to(cpu)
