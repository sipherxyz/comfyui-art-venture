import cv2
import torch
import numpy as np
from PIL import Image, ImageOps
from typing import Dict

from .sam.nodes import SAMLoader, GetSAMEmbedding, SAMEmbeddingToImage
from .lama import LaMaInpaint

from ..masking import get_crop_region, expand_crop_region
from ..image_utils import ResizeMode, resize_image, flatten_image
from ..utils import numpy2pil, tensor2pil, pil2tensor


class PrepareImageAndMaskForInpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "mask_blur": ("INT", {"default": 4, "min": 0, "max": 64}),
                "inpaint_masked": ("BOOLEAN", {"default": False}),
                "mask_padding": ("INT", {"default": 32, "min": 0, "max": 256}),
                "width": ("INT", {"default": 0, "min": 0, "max": 2048}),
                "height": ("INT", {"default": 0, "min": 0, "max": 2048}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "CROP_REGION")
    RETURN_NAMES = ("inpaint_image", "inpaint_mask", "overlay_image", "crop_region")
    CATEGORY = "Art Venture/Inpainting"
    FUNCTION = "prepare"

    def prepare(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        # resize_mode: str,
        mask_blur: int,
        inpaint_masked: bool,
        mask_padding: int,
        width: int,
        height: int,
    ):
        if image.shape[0] != mask.shape[0]:
            raise ValueError("image and mask must have same batch size")

        if image.shape[1] != mask.shape[1] or image.shape[2] != mask.shape[2]:
            raise ValueError("image and mask must have same dimensions")

        if width == 0 and height == 0:
            height, width = image.shape[1:3]
            
        sourceheight, sourcewidth = image.shape[1:3]

        masks = []
        images = []
        overlay_masks = []
        overlay_images = []
        crop_regions = []

        for img, msk in zip(image, mask):
            np_mask: np.ndarray = msk.cpu().numpy()

            if mask_blur > 0:
                kernel_size = 2 * int(2.5 * mask_blur + 0.5) + 1
                np_mask = cv2.GaussianBlur(np_mask, (kernel_size, kernel_size), mask_blur)

            pil_mask = numpy2pil(np_mask, "L")
            crop_region = None

            if inpaint_masked:
                crop_region = get_crop_region(np_mask, mask_padding)
                crop_region = expand_crop_region(crop_region, width, height, sourcewidth, sourceheight)
                # crop mask
                overlay_mask = pil_mask
                pil_mask = resize_image(pil_mask.crop(crop_region), width, height, ResizeMode.RESIZE_TO_FIT)
                pil_mask = pil_mask.convert("L")
            else:
                np_mask = np.clip((np_mask.astype(np.float32)) * 2, 0, 255).astype(np.uint8)
                overlay_mask = numpy2pil(np_mask, "L")

            pil_img = tensor2pil(img)
            pil_img = flatten_image(pil_img)

            image_masked = Image.new("RGBa", (pil_img.width, pil_img.height))
            image_masked.paste(pil_img.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(overlay_mask))
            overlay_images.append(pil2tensor(image_masked.convert("RGBA")))
            overlay_masks.append(pil2tensor(overlay_mask))

            if crop_region is not None:
                pil_img = resize_image(pil_img.crop(crop_region), width, height, ResizeMode.RESIZE_TO_FIT)
            else:
                crop_region = (0, 0, 0, 0)

            images.append(pil2tensor(pil_img))
            masks.append(pil2tensor(pil_mask))
            crop_regions.append(torch.tensor(crop_region, dtype=torch.int64))

        return (
            torch.cat(images, dim=0),
            torch.cat(masks, dim=0),
            torch.cat(overlay_images, dim=0),
            torch.stack(crop_regions),
        )


class OverlayInpaintedLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original": ("LATENT",),
                "inpainted": ("LATENT",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    CATEGORY = "Art Venture/Inpainting"
    FUNCTION = "overlay"

    def overlay(self, original: Dict, inpainted: Dict, mask: torch.Tensor):
        s_original: torch.Tensor = original["samples"]
        s_inpainted: torch.Tensor = inpainted["samples"]

        if s_original.shape[0] != s_inpainted.shape[0]:
            raise ValueError("original and inpainted must have same batch size")

        if s_original.shape[0] != mask.shape[0]:
            raise ValueError("original and mask must have same batch size")

        overlays = []

        for org, inp, msk in zip(s_original, s_inpainted, mask):
            latmask = tensor2pil(msk.unsqueeze(0), "L").convert("RGB").resize((org.shape[2], org.shape[1]))
            latmask = np.moveaxis(np.array(latmask, dtype=np.float32), 2, 0) / 255
            latmask = latmask[0]
            latmask = np.around(latmask)
            latmask = np.tile(latmask[None], (4, 1, 1))

            msk = torch.asarray(1.0 - latmask)
            nmask = torch.asarray(latmask)

            overlayed = inp * nmask + org * msk
            overlays.append(overlayed)

        samples = torch.stack(overlays)
        return ({"samples": samples},)


class OverlayInpaintedImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "inpainted": ("IMAGE",),
                "overlay_image": ("IMAGE",),
                "crop_region": ("CROP_REGION",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Art Venture/Inpainting"
    FUNCTION = "overlay"

    def overlay(self, inpainted: torch.Tensor, overlay_image: torch.Tensor, crop_region: torch.Tensor):
        if inpainted.shape[0] != overlay_image.shape[0]:
            raise ValueError("inpainted and overlay_image must have same batch size")
        if inpainted.shape[0] != crop_region.shape[0]:
            raise ValueError("inpainted and crop_region must have same batch size")

        images = []
        for image, overlay, region in zip(inpainted, overlay_image, crop_region):
            image = tensor2pil(image.unsqueeze(0))
            overlay = tensor2pil(overlay.unsqueeze(0), mode="RGBA")

            x1, y1, x2, y2 = region.tolist()
            if (x1, y1, x2, y2) == (0, 0, 0, 0):
                pass
            else:
                base_image = Image.new("RGBA", (overlay.width, overlay.height))
                image = resize_image(image, x2 - x1, y2 - y1, ResizeMode.RESIZE_TO_FILL)
                base_image.paste(image, (x1, y1))
                image = base_image

            image = image.convert("RGBA")
            image.alpha_composite(overlay)
            image = image.convert("RGB")

            images.append(pil2tensor(image))

        return (torch.cat(images, dim=0),)


NODE_CLASS_MAPPINGS = {
    "AV_SAMLoader": SAMLoader,
    "GetSAMEmbedding": GetSAMEmbedding,
    "SAMEmbeddingToImage": SAMEmbeddingToImage,
    "LaMaInpaint": LaMaInpaint,
    "PrepareImageAndMaskForInpaint": PrepareImageAndMaskForInpaint,
    "OverlayInpaintedLatent": OverlayInpaintedLatent,
    "OverlayInpaintedImage": OverlayInpaintedImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AV_SAMLoader": "SAM Loader",
    "GetSAMEmbedding": "Get SAM Embedding",
    "SAMEmbeddingToImage": "SAM Embedding to Image",
    "LaMaInpaint": "LaMa Remove Object",
    "PrepareImageAndMaskForInpaint": "Prepare Image & Mask for Inpaint",
    "OverlayInpaintedLatent": "Overlay Inpainted Latent",
    "OverlayInpaintedImage": "Overlay Inpainted Image",
}
