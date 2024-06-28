import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision.transforms.functional import normalize

import folder_paths
import comfy.model_management as model_management
import comfy.utils

from ..model_utils import download_model
from ..utils import pil2tensor, tensor2pil, numpy2pil
from ..logger import logger


isnets = {}
gpu = model_management.get_torch_device()
cpu = torch.device("cpu")
model_dir = os.path.join(folder_paths.models_dir, "isnet")
model_url = "https://huggingface.co/NimaBoscarino/IS-Net_DIS-general-use/resolve/main/isnet-general-use.pth"
cache_size = [1024, 1024]

folder_paths.folder_names_and_paths["isnet"] = (
    [model_dir],
    folder_paths.supported_pt_extensions,
)


class GOSNormalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = normalize(image, self.mean, self.std)
        return image


transform = transforms.Compose([GOSNormalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])])


def load_isnet_model(model_name):
    if model_name not in isnets:
        isnet_path = folder_paths.get_full_path("isnet", model_name)
        state_dict = comfy.utils.load_torch_file(isnet_path)

        from .models import ISNetBase, ISNetDIS

        if "side2.weight" in state_dict:
            isnet = ISNetDIS()
        else:
            isnet = ISNetBase()

        # convert to half precision
        isnet.is_fp16 = model_management.should_use_fp16()
        if isnet.is_fp16:
            isnet.half()
            for layer in isnet.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.float()

        isnet.load_state_dict(state_dict)
        isnet.eval()
        isnets[model_name] = isnet

    return isnets[model_name]


def im_preprocess(im: torch.Tensor, size):
    im = im.clone()

    # Ensure the image has three channels
    if len(im.shape) < 3:
        im = im.unsqueeze(2)
    if im.shape[2] == 1:
        im = im.repeat(1, 1, 3)

    # Permute dimensions to match the model input format (C, H, W)
    im_tensor = im.permute(2, 0, 1)

    # Resize the image
    im_tensor = F.interpolate(im_tensor.unsqueeze(0), size=size, mode="bilinear").squeeze(0)

    # Normalize the image
    im_tensor = normalize(im_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

    # Return the processed image tensor with a batch dimension and original size
    return im_tensor.unsqueeze(0), im.shape[0:2]


def predict(model, im: torch.Tensor, device):
    image, orig_size = im_preprocess(im, cache_size)

    if model.is_fp16:
        image = image.type(torch.HalfTensor)
    else:
        image = image.type(torch.FloatTensor)

    image_v = Variable(image, requires_grad=False).to(device)
    ds_val = model(image_v)  # list of 6 results

    if isinstance(ds_val, tuple):
        ds_val = ds_val[0]

    if isinstance(ds_val, list):
        ds_val = ds_val[0]

    if len(ds_val.shape) < 4:
        ds_val = torch.unsqueeze(ds_val, 0)

    # B x 1 x H x W    # we want the first one which is the most accurate prediction
    pred_val = ds_val[0, :, :, :]

    # recover the prediction spatial size to the orignal image size
    # pred_val = torch.squeeze(F.interpolate(pred_val, size=orig_size, mode='bilinear'), 0)
    # pred_val = torch.squeeze(F.upsample(torch.unsqueeze(pred_val, 0), (orig_size[0], orig_size[1]), mode="bilinear"))
    pred_val = F.interpolate(pred_val.unsqueeze(0), size=orig_size, mode="bilinear").squeeze(0)

    ma = torch.max(pred_val)
    mi = torch.min(pred_val)
    pred_val = (pred_val - mi) / (ma - mi)  # max = 1

    # it is the mask we need
    return pred_val.detach().cpu()


class ISNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("isnet"),),
                "model_override": ("STRING", {"default": "None"}),
            },
        }

    RETURN_TYPES = ("ISNET_MODEL",)
    FUNCTION = "load_isnet"
    CATEGORY = "Art Venture/Segmentation"

    def load_isnet(self, model_name, model_override="None"):
        if model_override != "None":
            if model_override not in folder_paths.get_filename_list("isnet"):
                logger.warning(f"Model override {model_override} not found. Use {model_name} instead.")
            else:
                model_name = model_override

        model = load_isnet_model(model_name)
        return (model,)


class ISNetSegment:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.001}),
            },
            "optional": {
                "device_mode": (["AUTO", "Prefer GPU", "CPU"],),
                "enabled": ("BOOLEAN", {"default": True}),
                "isnet_model": ("ISNET_MODEL",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("segmented", "mask")
    CATEGORY = "Art Venture/Segmentation"
    FUNCTION = "segment_isnet"

    def segment_isnet(self, images: torch.Tensor, threshold, device_mode="AUTO", enabled=True, isnet_model=None):
        if not enabled:
            masks = torch.zeros((len(images), 64, 64), dtype=torch.float32)
            return (images, masks)

        if isnet_model is None:
            ckpts = folder_paths.get_filename_list("isnet")
            if len(ckpts) == 0:
                ckpts = download_model(
                    model_path=model_dir,
                    model_url=model_url,
                    ext_filter=[".pth"],
                    download_name="isnet-general-use.pth",
                )
            isnet_model = load_isnet_model(ckpts[0])

        device = gpu if device_mode != "CPU" else cpu
        isnet_model = isnet_model.to(device)

        try:
            segments = []
            masks = []
            for image in images:
                mask = predict(isnet_model, image, device)
                mask_im = tensor2pil(mask.permute(1, 2, 0))
                cropped = Image.new("RGBA", mask_im.size, (0,0,0,0))
                cropped.paste(tensor2pil(image), mask=mask_im)

                masks.append(mask)
                segments.append(pil2tensor(cropped))

            return (torch.cat(segments, dim=0), torch.stack(masks))
        finally:
            if device_mode == "AUTO":
                isnet_model = isnet_model.to(cpu)
