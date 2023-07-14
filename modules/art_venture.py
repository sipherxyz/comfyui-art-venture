import time
import requests
import threading
import traceback
from typing import Callable, Dict
from types import MethodType

from ..config import config
from .log import logger as log
from .workflow import load_workflow, workflow_to_prompt
from .utils import get_task_from_av, upload_to_av

from server import PromptServer
import execution


def upload_task_result(task_id: str, success: bool):
    return upload_to_av(None, additional_data={"success": success}, task_id=task_id)


class ArtVentureRunner:
    instance: "ArtVentureRunner" = None

    def __init__(self) -> None:
        self.current_task_id: str = None
        self.current_thread: threading.Thread = None
        ArtVentureRunner.instance = self

        # hijack PromptQueue
        orig_task_done = PromptServer.instance.prompt_queue.task_done

        def task_done(self, item_id, outputs):
            item = self.currently_running.get(item_id, None)
            if item:
                task_id = item[1]
                ArtVentureRunner.instance.on_task_finished(task_id)
            orig_task_done(item_id, outputs)

        PromptServer.instance.prompt_queue.task_done = MethodType(
            task_done, PromptServer.instance.prompt_queue
        )

    def get_new_task(self):
        if config.get("runner_enabled", False) != True:
            return (None, None)

        if self.current_task_id is not None:
            return (None, None)

        data = get_task_from_av()
        if data["has_task"] != True:
            return (None, None)

        workflow = data.get("receipt_id", None)
        args: dict = data.get("task", {})
        task_id = args.get("task_id")
        log.info(f"Got new task {task_id} with receipt {workflow}")

        # validate workflow
        if isinstance(workflow, str):
            workflow = load_workflow(workflow)

        if workflow is None:
            return (task_id, Exception("Missing receipt"))

        prompt = workflow_to_prompt(workflow, args)
        valid = execution.validate_prompt(prompt)
        if not valid[0]:
            log.error(f"Invalid receipt: {valid[3]}")
            return (task_id, Exception("Invalid receipt"))

        outputs_to_execute = valid[2]
        extra_data = {"extra_data": {"extra_pnginfo": {"workflow": workflow}}}
        PromptServer.instance.prompt_queue.put(
            (0, task_id, prompt, extra_data, outputs_to_execute)
        )

        self.current_task_id = task_id
        return (task_id, None)

    def watching_for_new_task(self, get_task: Callable):
        log.info("Watching for new task")

        failed_attempts = 0
        while True:
            if self.current_task_id is not None:
                time.sleep(2)
                continue

            try:
                api_task_id, e = get_task()
                if api_task_id and e is not None:
                    log.error("[ArtVenture] Error while getting new task")
                    log.error(e)
                    upload_task_result(api_task_id, False)
                    failed_attempts += 1
                else:
                    failed_attempts = 0
            except requests.exceptions.ConnectionError:
                log.error("Connection error while getting new task")
                failed_attempts += 1
            except Exception as e:
                log.error("Error while getting new task")
                log.error(e)
                log.debug(traceback.format_exc())
                failed_attempts += 1

            # increase sleep time based on failed attempts
            time.sleep(min(3 + 5 * failed_attempts, 60))

    def watching_for_new_task_threading(self):
        if config.get("runner_enabled", False) != True:
            log.info("[ArtVenture] Runner is disabled")
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
    ):
        if task_id != self.current_task_id:
            return

        log.info(f"Task {task_id} finished")
        self.current_task_id = None
