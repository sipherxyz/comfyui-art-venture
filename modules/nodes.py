import io
import os
import json
import torch
import requests

from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
import numpy as np

from .logger import logger
from .utils import upload_to_av

import folder_paths


class UtilImagesConcat:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"images": ("IMAGE",)},
            "optional": {
                "images_2": ("IMAGE",),
                "images_3": ("IMAGE",),
                "images_4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Art Venture"
    FUNCTION = "concat_images"

    def concat_images(self, images, images_2=None, images_3=None, images_4=None):
        print("images", type(images), images)
        print("images_2", type(images_2), images_2)
        print("images_3", type(images_3), images_3)
        print("images_4", type(images_4), images_4)

        all_images = []
        all_images.extend(images if isinstance(images, list) else [images])

        if images_2 is not None:
            all_images.extend(images_2 if isinstance(images_2, list) else [images_2])
        if images_3 is not None:
            all_images.extend(images_3 if isinstance(images_3, list) else [images_3])
        if images_4 is not None:
            all_images.extend(images_4 if isinstance(images_4, list) else [images_4])

        return all_images


class UtilLoadImageFromUrl:
    def __init__(self) -> None:
        self.output_dir = folder_paths.get_temp_directory()
        self.filename_prefix = "TempImageFromUrl"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"url": ("STRING", {"default": ""})},
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    CATEGORY = "image"
    FUNCTION = "load_image_from_url"

    def load_image_from_url(self, url: str):
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            raise Exception(response.text)

        i = Image.open(io.BytesIO(response.content))
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")

        # save image to temp folder
        (
            outdir,
            filename,
            counter,
            subfolder,
            _,
        ) = folder_paths.get_save_image_path(
            self.filename_prefix, self.output_dir, image.width, image.height
        )
        file = f"{filename}_{counter:05}.png"
        image.save(os.path.join(outdir, file), format="PNG", compress_level=4)
        preview = {
            "filename": file,
            "subfolder": subfolder,
            "type": "temp",
        }

        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if "A" in i.getbands():
            mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

        return {"ui": {"images": [preview]}, "result": (image, mask)}


class AVOutputUploadImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"images": ("IMAGE",)},
            "optional": {
                "folder_id": ("STRING", {"multiline": False}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    CATEGORY = "Art Venture"
    FUNCTION = "upload_images"

    def upload_images(
        self,
        images,
        folder_id: str = None,
        prompt=None,
        extra_pnginfo=None,
    ):
        files = list()
        for idx, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    logger.debug(f"Adding {x} to pnginfo: {extra_pnginfo[x]}")
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            buffer = io.BytesIO()
            img.save(buffer, format="PNG", pnginfo=metadata, compress_level=4)
            buffer.seek(0)
            files.append(
                (
                    "files",
                    (f"image-{idx}.png", buffer, "image/png"),
                )
            )

        additional_data = {"success": "true"}
        if folder_id is not None:
            additional_data["folderId"] = folder_id

        upload_to_av(files, additional_data=additional_data)
        return ("Uploaded to ArtVenture!",)


NODE_CLASS_MAPPINGS = {
    "ImagesConcat": UtilImagesConcat,
    "LoadImageFromUrl": UtilLoadImageFromUrl,
    "AV_UploadImage": AVOutputUploadImage,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImagesConcat": "Images Concat",
    "LoadImageFromUrl": "Load Image From URL",
    "AV_UploadImage": "Upload to Art Venture",
}
