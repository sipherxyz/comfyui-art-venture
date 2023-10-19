import os
from typing import Callable

import folder_paths

from ..utils import load_module

custom_nodes = folder_paths.get_folder_paths("custom_nodes")
animatediff_dir_names = ["AnimateDiff", "comfyui-animatediff"]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    module_path = None

    for custom_node in custom_nodes:
        custom_node = custom_node if not os.path.islink(custom_node) else os.readlink(custom_node)
        for module_dir in animatediff_dir_names:
            if module_dir in os.listdir(custom_node):
                module_path = os.path.abspath(os.path.join(custom_node, module_dir))
                break

    if module_path is None:
        raise Exception("Could not find AnimateDiff nodes")

    module_path = os.path.join(module_path, "animatediff/sliding_schedule.py")
    module = load_module(module_path)
    print("Loaded AnimateDiff from", module_path)

    get_context_scheduler: Callable = module.get_context_scheduler

    class AnimateDiffFrameCalculator:
        @classmethod
        def INPUT_TYPES(s):
            return {
                "required": {
                    "frame_rate": ("INT", {"default": 8, "min": 1, "max": 24, "step": 1}),
                    "duration": ("INT", {"default": 2, "min": 1, "max": 10000, "step": 1}),
                    "sliding_window": ("SLIDING_WINDOW_OPTS",),
                }
            }

        RETURN_TYPES = ("INT", "INT", "INT", "INT")
        RETURN_NAMES = ("frame_number", "_1/2-1_index", "_1/2_index", "end_index")
        FUNCTION = "calculate"
        CATEGORY = "Animate Diff"

        def get_batch_count(self, frame_number, context_scheduler, ctx):
            batches = list(
                context_scheduler(
                    0,
                    0,
                    frame_number,
                    ctx.context_length,
                    ctx.context_stride,
                    ctx.context_overlap,
                    ctx.closed_loop,
                )
            )
            batch_count = len(batches)
            if len(batches[-1]) == 0:
                batch_count -= 1

            return batch_count

        def calculate(self, frame_rate: int, duration: int, sliding_window):
            frame_number = frame_rate * duration

            ctx = sliding_window
            context_scheduler = get_context_scheduler(ctx.context_schedule)
            batch_count = self.get_batch_count(frame_number, context_scheduler, ctx)

            while True:
                next_batch_count = self.get_batch_count(frame_number + 1, context_scheduler, ctx)
                if next_batch_count > batch_count:
                    break

                frame_number += 1

            snd_half_start = frame_number // 2 + frame_number % 2
            fst_half_end = snd_half_start - 1
            return (frame_number, fst_half_end, snd_half_start, frame_number - 1)

    NODE_CLASS_MAPPINGS["AnimateDiffFrameCalculator"] = AnimateDiffFrameCalculator
    NODE_DISPLAY_NAME_MAPPINGS["AnimateDiffFrameCalculator"] = "Animate Diff Frame Calculator"

except Exception as e:
    print(e)
