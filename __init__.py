import os
import time
import shutil
import threading
from typing import Callable

from .config import get_ext_dir, get_comfy_dir
from .modules.log import logger as log
from .modules.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .modules.workflow import update_checkpoints_hash
from .modules.art_venture import ArtVentureRunner


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
        log.error("No JS")
        return

    dst_dir = get_web_ext_dir()

    if os.path.exists(dst_dir):
        if os.path.islink(dst_dir) or is_junction(dst_dir):
            log.info("JS already linked")
            return
    elif link_js(src_dir, dst_dir):
        log.info("JS linked")
        return

    log.info("Unable to make symlink, copying JS files")
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)


def init():
    install_js()

    # update checkpoint hash in background
    def _update_checkpoints_hash(cb: Callable):
        update_checkpoints_hash()
        time.sleep(10) # wait for server to start
        cb()

    def _cb():
        ArtVentureRunner().watching_for_new_task_threading()

    background_thread = threading.Thread(target=_update_checkpoints_hash, args=(_cb,))
    background_thread.daemon = True
    background_thread.start()


init()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
