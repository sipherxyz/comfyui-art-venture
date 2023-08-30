import os
import time
import requests
import traceback
from typing import Callable, Dict

from ..config import config
from .logger import logger


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
    headers = (
        {"Authorization": f"Bearer {auth_token}"}
        if auth_token and auth_token != ""
        else None
    )

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
    headers = (
        {"Authorization": f"Bearer {auth_token}"}
        if auth_token and auth_token != ""
        else None
    )

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

    child = subprocess.Popen(
        'fsutil reparsepoint query "{}"'.format(src), stdout=subprocess.PIPE
    )
    streamdata = child.communicate()[0]
    rc = child.returncode
    return rc == 0


def load_module(module_path):
    import importlib.util

    module_name = os.path.basename(module_path)
    if os.path.isfile(module_path):
        module_spec = importlib.util.spec_from_file_location(module_name, module_path)
    else:
        module_spec = importlib.util.spec_from_file_location(
            module_name, os.path.join(module_path, "__init__.py")
        )

    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    return module
