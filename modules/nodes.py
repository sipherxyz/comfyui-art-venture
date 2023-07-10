import io
import json
import requests

from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np

from ..config import config
from .utils import request_with_retry


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


class AVOutput_UploadImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "task_id": ("AV_TASK_ID",),
            },
            "required": {"images": ("IMAGE",)},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    CATEGORY = "ArtVenture"
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
    "AV_UploadImage": AVOutput_UploadImage,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AV_UploadImage": "AV Upload Image",
}
