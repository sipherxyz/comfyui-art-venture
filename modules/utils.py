import os
import sys
import time
import torch
import numpy as np
import requests
import traceback
import importlib
import subprocess
from typing import Callable, Dict
from PIL import Image

from ..config import config
from .logger import logger


def ensure_package(package, install_package_name=None):
    # Try to import the package
    try:
        importlib.import_module(package)
    except ImportError:
        logger.info(f"Package {package} is not installed. Installing now...")

        if "python_embeded" in sys.executable or "python_embedded" in sys.executable:
            pip_install = [sys.executable, "-s", "-m", "pip", "install"]
        else:
            pip_install = [sys.executable, "-m", "pip", "install"]

        subprocess.check_call(pip_install + [install_package_name or package])
    else:
        print(f"Package {package} is already installed.")


def request_with_retry(
    make_request: Callable[[], requests.Response],
    max_try: int = 3,
    retries: int = 0,
):
    try:
        res = make_request()
        if res.status_code > 400:
            raise Exception(res.text)

        return True
    except requests.exceptions.ConnectionError:
        logger.error("Connection error")
        if retries >= max_try - 1:
            return False

        time.sleep(2)
        logger.info(f"Retrying {retries + 1}...")
        return request_with_retry(
            make_request,
            max_try=max_try,
            retries=retries + 1,
        )
    except Exception as e:
        logger.error("Request error")
        logger.error(e)
        logger.debug(traceback.format_exc())
        return False


def upload_to_av(
    files: list,
    additional_data: dict = {},
    task_id: str = None,
    upload_url: str = None,
):
    if upload_url is None:
        upload_url = config.get("av_endpoint") + "/api/recipe/sd-tasks"
        if task_id is not None and task_id != "":
            upload_url += f"/complete/{task_id}"
        else:
            upload_url += "/upload"

    auth_token = config.get("av_token")
    headers = {"Authorization": f"Bearer {auth_token}"} if auth_token and auth_token != "" else None

    upload = lambda: requests.post(
        upload_url,
        timeout=30,
        headers=headers,
        files=files,
        data=additional_data,
    )

    return request_with_retry(upload)


def get_task_from_av():
    get_task_url = config.get("av_endpoint") + "/api/recipe/sd-tasks/one-in-queue"
    auth_token = config.get("av_token", None)
    headers = {"Authorization": f"Bearer {auth_token}"} if auth_token and auth_token != "" else None

    response = requests.get(get_task_url, timeout=10, headers=headers)
    if response.status_code >= 400:
        raise Exception(response.text)

    data: Dict = response.json()

    return data


def get_dict_attribute(dict_inst: dict, name_string: str, default=None):
    nested_keys = name_string.split(".")
    value = dict_inst

    for key in nested_keys:
        value = value.get(key, None)

        if value is None:
            return default

    return value


def set_dict_attribute(dict_inst: dict, name_string: str, value):
    """
    Set an attribute to a dictionary using dot notation.
    If the attribute does not already exist, it will create a nested dictionary.

    Parameters:
        - dict_inst: the dictionary instance to set the attribute
        - name_string: the attribute name in dot notation (ex: 'attributes[1].name')
        - value: the value to set for the attribute

    Returns:
        None
    """
    # Split the attribute names by dot
    name_list = name_string.split(".")

    # Traverse the dictionary and create a nested dictionary if necessary
    current_dict = dict_inst
    for name in name_list[:-1]:
        is_array = name.endswith("]")
        if is_array:
            open_bracket_index = name.index("[")
            idx = int(name[open_bracket_index + 1 : -1])
            name = name[:open_bracket_index]

        if name not in current_dict:
            current_dict[name] = [] if is_array else {}

        current_dict = current_dict[name]
        if is_array:
            while len(current_dict) <= idx:
                current_dict.append({})
            current_dict = current_dict[idx]

    # Set the final attribute to its value
    name = name_list[-1]
    if name.endswith("]"):
        open_bracket_index = name.index("[")
        idx = int(name[open_bracket_index + 1 : -1])
        name = name[:open_bracket_index]

        if name not in current_dict:
            current_dict[name] = []

        while len(current_dict[name]) <= idx:
            current_dict[name].append(None)

        current_dict[name][idx] = value
    else:
        current_dict[name] = value


def is_junction(src: str) -> bool:
    import subprocess

    child = subprocess.Popen('fsutil reparsepoint query "{}"'.format(src), stdout=subprocess.PIPE)
    streamdata = child.communicate()[0]
    rc = child.returncode
    return rc == 0


def load_module(module_path):
    import importlib.util

    module_name = os.path.basename(module_path)
    if os.path.isfile(module_path):
        module_spec = importlib.util.spec_from_file_location(module_name, module_path)
    else:
        module_spec = importlib.util.spec_from_file_location(module_name, os.path.join(module_path, "__init__.py"))

    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    return module


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def tensor2pil(image):
    return Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def resize_image(resize_mode, im: Image.Image, width, height):
    """
    Resizes an image with the specified resize_mode, width, and height.

    Args:
        resize_mode: The mode to use when resizing the image.
            0: Resize the image to the specified width and height.
            1: Resize the image to fill the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, cropping the excess.
            2: Resize the image to fit within the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, filling empty with data from image.
        im: The image to resize.
        width: The width to resize the image to.
        height: The height to resize the image to.
        upscaler_name: The name of the upscaler to use. If not provided, defaults to opts.upscaler_for_img2img.
    """

    def resize(im: Image.Image, w, h):
        return im.resize((w, h), resample=Image.LANCZOS)

    if resize_mode == 0:
        res = resize(im, width, height)

    elif resize_mode == 1:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

    else:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            if fill_height > 0:
                res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
                res.paste(
                    resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)),
                    box=(0, fill_height + src_h),
                )
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            if fill_width > 0:
                res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
                res.paste(
                    resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)),
                    box=(fill_width + src_w, 0),
                )

    return res


def tensor2bytes(image: torch.Tensor) -> bytes:
    return tensor2pil(image).tobytes()
