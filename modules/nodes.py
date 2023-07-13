import io
import os
import json
import torch
import requests

from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
import numpy as np

from ..config import config
from .utils import request_with_retry
from .log import logger as log

import folder_paths


def upload_to_av(
    files: list,
    additional_data: dict = {},
    upload_url: str = None,
    task_id: str = None,
):
    if upload_url is None:
        upload_url = config.get("av_endpoint") + "/api/sd-tasks"
        if task_id is not None and task_id != "":
            upload_url += f"/complete/{task_id}"
        else:
            upload_url += "/upload"

    auth_token = config.get("av_token")
    headers = (
        {"Authorization": f"Bearer {auth_token}"}
        if auth_token and auth_token != ""
        else None
    )

    upload = lambda: requests.post(
        upload_url,
        timeout=5,
        headers=headers,
        files=files,
        data=additional_data,
    )

    return request_with_retry(upload)


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
        all_images = []
        all_images.extend(images)

        if images_2 is not None:
            all_images.extend(images_2)
        if images_3 is not None:
            all_images.extend(images_3)
        if images_4 is not None:
            all_images.extend(images_4)

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


class AVInputImageFromUrl(UtilLoadImageFromUrl):
    def __init__(self) -> None:
        super().__init__()
        self.filename_prefix = "AV_InputImage"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {"default": ""}),
                "name": ("STRING", {"default": "image"}),
            },
        }
    
    CATEGORY = "Art Venture"

    def load_image_from_url(self, url: str, name: str):
        UtilLoadImageFromUrl.load_image_from_url(self, url)


class AVOutputUploadImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"images": ("IMAGE",)},
            "optional": {
                "task_id": "STRING",
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
        self, images, task_id: str = None, prompt=None, extra_pnginfo=None
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
                    log.debug(f"Adding {x} to pnginfo: {extra_pnginfo[x]}")
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

        upload_to_av(files, additional_data={"success": True}, task_id=task_id)
        return ("Uploaded to ArtVenture!",)


NODE_CLASS_MAPPINGS = {
    "ImagesConcat": UtilImagesConcat,
    "LoadImageFromUrl": UtilLoadImageFromUrl,
    "AV_InputImage": AVInputImageFromUrl,
    "AV_UploadImage": AVOutputUploadImage,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImagesConcat": "Images Concat",
    "LoadImageFromUrl": "Load Image From URL",
    "AV_InputImage": "AV Receipt Image",
    "AV_UploadImage": "Upload to Art Venture",
}
