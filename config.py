import os
import json
import inspect
from typing import Dict

from server import PromptServer

from .modules.logger import logger

comfy_dir = os.path.dirname(inspect.getfile(PromptServer))
ext_dir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(ext_dir, "config.json")


def __get_dir(root: str, subpath=None, mkdir=False):
    dir = root
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    dir = os.path.abspath(dir)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def get_ext_dir(subpath=None, mkdir=False):
    return __get_dir(ext_dir, subpath, mkdir)


def get_comfy_dir(subpath=None, mkdir=False):
    return __get_dir(comfy_dir, subpath, mkdir)


def write_config(config):
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


def load_config() -> Dict:
    default_config = {
        "av_endpoint": "https://api.artventure.ai",
        "av_token": "",
        "runner_enabled": False,
        "remove_runner_images_after_upload": False,
    }

    if not os.path.isfile(config_path):
        logger.info("Config file not found, creating...")
        write_config(default_config)

    with open(config_path, "r") as f:
        config = json.load(f)

        need_update = False
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
                need_update = True

        if need_update:
            write_config(config)

    logger.debug(f"Loaded config {config}")
    return config


config = load_config()
