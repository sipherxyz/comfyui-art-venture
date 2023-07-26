import os
import time
import pathlib
import requests
import threading
import traceback
from uuid import uuid4
from typing import Callable, Dict, List
from types import MethodType

from ..config import config
from .logger import logger
from .utils import get_task_from_av, upload_to_av

import folder_paths
from server import PromptServer
from execution import validate_prompt, PromptExecutor


def update_task_result(callback_url: str, success: bool, images: List[str] = None):
    files = None
    if images is not None:
        files = []
        for img in images:
            img_path = pathlib.Path(img)
            ext = img_path.suffix.lower()
            content_type = f"image/{ext[1:]}"
            files.append(
                (
                    "files",
                    (img_path.name, open(os.path.abspath(img), "rb"), content_type),
                )
            )

    return upload_to_av(
        files,
        additional_data={"success": str(success).lower()},
        upload_url=callback_url,
    )


def patch_comfy():
    # monky patch PromptQueue
    orig_task_done = PromptServer.instance.prompt_queue.task_done

    def handle_task_done(queue, item_id, outputs):
        item = queue.currently_running.get(item_id, None)
        if item:
            task_id = item[1]
            ArtVentureRunner.instance.on_task_finished(task_id, outputs)

        orig_task_done(item_id, outputs)

    PromptServer.instance.prompt_queue.task_done = MethodType(
        handle_task_done, PromptServer.instance.prompt_queue
    )

    # monky patch PromptExecutor
    PromptExecutor.orig_handle_execution_error = PromptExecutor.handle_execution_error

    def handle_execution_error(
        self, prompt_id, prompt, current_outputs, executed, error, ex
    ):
        node_id = error["node_id"]
        class_type = prompt[node_id]["class_type"]
        mes = {
            "type": error.get("exception_type", ex.__class__.__name__),
            "message": error.get("exception_message", str(ex)),
        }
        details = {
            "node_id": node_id,
            "node_type": class_type,
            "traceback": error.get("traceback", None),
        }
        ArtVentureRunner.instance.current_task_exception = {
            "error": mes,
            "details": details,
        }
        self.orig_handle_execution_error(
            prompt_id, prompt, current_outputs, executed, error, ex
        )

    PromptExecutor.handle_execution_error = handle_execution_error


class ArtVentureRunner:
    instance: "ArtVentureRunner" = None

    def __init__(self) -> None:
        self.current_task_id: str = None
        self.callback_url: str = None
        self.current_task_exception = None
        self.current_thread: threading.Thread = None
        ArtVentureRunner.instance = self

        patch_comfy()

    def get_new_task(self):
        if config.get("runner_enabled", False) != True:
            return (None, None)

        if self.current_task_id is not None:
            return (None, None)

        try:
            data = get_task_from_av()
        except Exception as e:
            logger.error(f"Error while getting new task {e}")
            return (None, e)

        if data["has_task"] != True:
            return (None, None)

        prompt = data.get("prompt")
        callback_url: str = data.get("callback_url")
        logger.info(f"Got new task")
        logger.debug(prompt)

        valid = validate_prompt(prompt)
        if not valid[0]:
            logger.error(f"Invalid recipe: {valid[3]}")
            return (callback_url, Exception("Invalid recipe"))

        task_id = str(uuid4())
        outputs_to_execute = valid[2]
        PromptServer.instance.prompt_queue.put(
            (0, task_id, prompt, {}, outputs_to_execute)
        )

        logger.info(f"Task registered with id {task_id}")
        self.current_task_id = task_id
        self.callback_url = callback_url
        self.current_task_exception = None
        return (callback_url, None)

    def watching_for_new_task(self, get_task: Callable):
        logger.info("Watching for new task")

        failed_attempts = 0
        while True:
            if self.current_task_id is not None:
                time.sleep(2)
                continue

            try:
                callback_url, e = get_task()
                if callback_url and e is not None:
                    logger.error("Error while getting new task")
                    logger.error(e)
                    update_task_result(callback_url, False)
                    failed_attempts += 1
                else:
                    failed_attempts = 0
            except requests.exceptions.ConnectionError:
                logger.error("Connection error while getting new task")
                failed_attempts += 1
            except Exception as e:
                logger.error("Error while getting new task")
                logger.error(e)
                logger.debug(traceback.format_exc())
                failed_attempts += 1

            # increase sleep time based on failed attempts
            time.sleep(min(3 + 5 * failed_attempts, 60))

    def watching_for_new_task_threading(self):
        if config.get("runner_enabled", False) != True:
            logger.info("Runner is disabled")
            return

        if self.current_thread is not None and self.current_thread.is_alive():
            return

        self.current_thread = threading.Thread(
            target=self.watching_for_new_task, args=(self.get_new_task,)
        )
        self.current_thread.daemon = True
        self.current_thread.start()

    def on_task_finished(
        self,
        task_id: str,
        outputs: Dict,
    ):
        if task_id != self.current_task_id:
            return

        if self.current_task_exception is not None:
            logger.info(f"Task {task_id} failed: {self.current_task_exception}")
            update_task_result(callback_url=self.callback_url, success=False)
        else:
            images = []
            outdir = folder_paths.get_output_directory()
            for k, v in outputs.items():
                files = v.get("images", [])
                for image in files:
                    type = image.get("type", None)
                    if type == "output":
                        filename = image.get("filename")
                        subfolder = image.get("subfolder", "")
                        images.append(os.path.join(outdir, subfolder, filename))

            logger.info(f"Task {task_id} finished with {len(images)} image(s)")
            update_task_result(self.callback_url, True, images)
            if config.get("remove_runner_images_after_upload", False):
                for img in images:
                    if os.path.exists(img):
                        os.remove(img)

        self.current_task_id = None
        self.callback_url = None
        self.current_task_exception = None
