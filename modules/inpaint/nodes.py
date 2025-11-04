import cv2
import torch
import numpy as np
from PIL import Image, ImageOps
from typing import Dict

from .sam.nodes import SAMLoader, GetSAMEmbedding, SAMEmbeddingToImage
from .lama import LoadLaMaModel, LaMaInpaint

from ..masking import get_crop_region, expand_crop_region
from ..image_utils import ResizeMode, resize_image
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
                "mask_padding": ("INT", {"default": 32, "min": 0, "max": 1024}),
                "width": ("INT", {"default": 0, "min": 0, "max": 2048}),
                "height": ("INT", {"default": 0, "min": 0, "max": 2048}),
            },
            "optional": {
                "controlnet_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "CROP_REGION", "IMAGE")
    RETURN_NAMES = ("inpaint_image", "inpaint_mask", "overlay_image", "crop_region", "controlnet_image")
    CATEGORY = "ArtVenture/Inpainting"
    FUNCTION = "prepare"

    def prepare(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        mask_blur: int,
        inpaint_masked: bool,
        mask_padding: int,
        width: int,
        height: int,
        controlnet_image: torch.Tensor = None,
    ):
        if image.shape[0] != mask.shape[0]:
            raise ValueError("image and mask must have same batch size")

        if controlnet_image is not None and image.shape[0] != controlnet_image.shape[0]:
            raise ValueError("image and controlnet_image must have same batch size")

        if image.shape[1] != mask.shape[1] or image.shape[2] != mask.shape[2]:
            raise ValueError("image and mask must have same dimensions")

        # These are only used if inpaint_masked is True
        out_width, out_height = width, height
        if inpaint_masked and out_width == 0 and out_height == 0:
            out_height, out_width = image.shape[1:3]

        source_height, source_width = image.shape[1:3]

        images = []
        masks = []
        overlay_images = []
        crop_regions = []
        processed_controlnet_images = []

        for idx, (img, msk) in enumerate(zip(image, mask)):
            np_mask: np.ndarray = msk.cpu().numpy()

            if mask_blur > 0:
                kernel_size = 2 * int(2.5 * mask_blur + 0.5) + 1
                np_mask = cv2.GaussianBlur(np_mask, (kernel_size, kernel_size), mask_blur)

            pil_mask = numpy2pil(np_mask, "L")
            pil_img = tensor2pil(img)

            # --- LOGIC SEPARATION ---

            if inpaint_masked:
                # --- MODE 1: CROP AND RESIZE ---
                crop_region = get_crop_region(np_mask, mask_padding)
                crop_region = expand_crop_region(crop_region, out_width, out_height, source_width, source_height)

                cropped_img = pil_img.crop(crop_region)
                cropped_mask = pil_mask.crop(crop_region)

                final_pil_img = resize_image(cropped_img, out_width, out_height, ResizeMode.RESIZE_TO_FIT)
                final_pil_mask = resize_image(cropped_mask, out_width, out_height, ResizeMode.RESIZE_TO_FIT).convert(
                    "L"
                )

                if controlnet_image is not None:
                    pil_cimg = tensor2pil(controlnet_image[idx])
                    cn_source_width, cn_source_height = pil_cimg.size
                    scale_x = cn_source_width / source_width
                    scale_y = cn_source_height / source_height

                    cn_target_width = int(out_width * scale_x)
                    cn_target_height = int(out_height * scale_y)

                    x1, y1, x2, y2 = crop_region
                    cn_crop_region = (int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y))
                    cropped_cn_img = pil_cimg.crop(cn_crop_region)
                    final_cn_img = resize_image(
                        cropped_cn_img, cn_target_width, cn_target_height, ResizeMode.RESIZE_TO_FIT
                    )
                    processed_controlnet_images.append(pil2tensor(final_cn_img))

            else:
                # --- MODE 2: PASS-THROUGH (NO RESIZING) ---
                final_pil_img = pil_img
                final_pil_mask = pil_mask  # Already blurred if requested
                crop_region = (0, 0, source_width, source_height)

                if controlnet_image is not None:
                    # Simply pass the original controlnet image through
                    final_cn_img = tensor2pil(controlnet_image[idx])
                    processed_controlnet_images.append(pil2tensor(final_cn_img))

            # --- COMMON LOGIC FOR BOTH MODES ---

            # The overlay/preview should always be based on the original full-size image
            image_masked = Image.new("RGBa", (pil_img.width, pil_img.height))
            # The mask used here is the potentially blurred one, but before any cropping/resizing
            image_masked.paste(pil_img.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(pil_mask))
            overlay_images.append(pil2tensor(image_masked.convert("RGBA")))

            images.append(pil2tensor(final_pil_img))
            masks.append(pil2tensor(final_pil_mask))
            crop_regions.append(torch.tensor(crop_region, dtype=torch.int64))

        if processed_controlnet_images:
            final_controlnet_tensor = torch.cat(processed_controlnet_images, dim=0)
        else:
            # If no controlnet image is provided, create a black 64x64 placeholder
            batch_size = image.shape[0]
            final_controlnet_tensor = torch.zeros((batch_size, 64, 64, 3), dtype=torch.float32, device=image.device)

        return (
            torch.cat(images, dim=0),
            torch.cat(masks, dim=0),
            torch.cat(overlay_images, dim=0),
            torch.stack(crop_regions, dim=0),
            final_controlnet_tensor,
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
    CATEGORY = "ArtVenture/Inpainting"
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
    CATEGORY = "ArtVenture/Inpainting"
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
    "LoadLaMaModel": LoadLaMaModel,
    "LaMaInpaint": LaMaInpaint,
    "PrepareImageAndMaskForInpaint": PrepareImageAndMaskForInpaint,
    "OverlayInpaintedLatent": OverlayInpaintedLatent,
    "OverlayInpaintedImage": OverlayInpaintedImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AV_SAMLoader": "SAM Loader",
    "GetSAMEmbedding": "Get SAM Embedding",
    "SAMEmbeddingToImage": "SAM Embedding to Image",
    "LoadLaMaModel": "LaMa Loader",
    "LaMaInpaint": "LaMa Remove Object",
    "PrepareImageAndMaskForInpaint": "Prepare Image & Mask for Inpaint",
    "OverlayInpaintedLatent": "Overlay Inpainted Latent",
    "OverlayInpaintedImage": "Overlay Inpainted Image",
}
