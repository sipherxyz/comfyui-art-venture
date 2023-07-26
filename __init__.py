import os
import time
import shutil
from threading import Thread
from types import MethodType
from typing import Callable

from .config import get_ext_dir, get_comfy_dir
from .modules.logger import logger
from .modules.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .modules.workflow import update_checkpoints_hash
from .modules.art_venture import ArtVentureRunner

from server import PromptServer


def get_web_ext_dir():
    ext_dir = get_comfy_dir("web/extensions")
    return os.path.join(ext_dir, "art-venture")


def link_js(src, dst):
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    if os.name == "nt":
        try:
            import _winapi

            _winapi.CreateJunction(src, dst)
            return True
        except:
            pass
    try:
        os.symlink(src, dst)
        return True
    except:
        import logging

        logging.exception("")
        return False


def is_junction(path):
    if os.name != "nt":
        return False
    try:
        return bool(os.readlink(path))
    except OSError:
        return False


def install_js():
    src_dir = get_ext_dir("javascript")
    if not os.path.exists(src_dir):
        logger.error("No JS")
        return

    dst_dir = get_web_ext_dir()

    if os.path.exists(dst_dir):
        if os.path.islink(dst_dir) or is_junction(dst_dir):
            logger.info("JS already linked")
            return
    elif link_js(src_dir, dst_dir):
        logger.info("JS linked")
        return

    logger.info("Unable to make symlink, copying JS files")
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)


def init():
    install_js()

    # update checkpoint hash in background
    def _update_checkpoints_hash(cb: Callable):
        # hijack PromptServer
        orig_add_routes = PromptServer.instance.add_routes

        def add_routes(self):
            cb()
            orig_add_routes()

        PromptServer.instance.add_routes = MethodType(add_routes, PromptServer.instance)

        update_checkpoints_hash()

    def _cb():
        ArtVentureRunner().watching_for_new_task_threading()

    Thread(target=_update_checkpoints_hash, args=(_cb,), daemon=True).start()


init()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
