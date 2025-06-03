import os
import torch

import folder_paths
from comfy.utils import common_upscale

from ..utils import load_module, pil2tensor

custom_nodes = folder_paths.get_folder_paths("custom_nodes")
video_dir_names = ["VideoHelperSuite", "ComfyUI-VideoHelperSuite"]

output_dir = folder_paths.get_output_directory()
input_dir = folder_paths.get_input_directory()
temp_dir = folder_paths.get_temp_directory()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    module_path = None

    for custom_node in custom_nodes:
        custom_node = custom_node if not os.path.islink(custom_node) else os.readlink(custom_node)
        for module_dir in video_dir_names:
            if module_dir in os.listdir(custom_node):
                module_path = os.path.abspath(os.path.join(custom_node, module_dir))
                break

    if module_path is None:
        raise Exception("Could not find VideoHelperSuite nodes")

    module_path = os.path.join(module_path)
    module = load_module(module_path)
    print("Loaded VideoHelperSuite from", module_path)

    LoadVideoPath = module.NODE_CLASS_MAPPINGS["VHS_LoadVideoPath"]

    def target_size(width, height, force_size) -> tuple[int, int]:
        force_size = force_size.split("x")
        if force_size[0] == "?":
            width = (width * int(force_size[1])) // height
            width = int(width) + 4 & ~7
            height = int(force_size[1])
        elif force_size[1] == "?":
            height = (height * int(force_size[0])) // width
            height = int(height) + 4 & ~7
            width = int(force_size[0])

        return (width, height)

    class UtilLoadVideoFromUrl(LoadVideoPath):
        @classmethod
        def INPUT_TYPES(s):
            inputs = LoadVideoPath.INPUT_TYPES()
            inputs["required"]["video"] = ("STRING", {"default": ""})
            return inputs

        CATEGORY = "Art Venture/Loaders"
        FUNCTION = "load"
        RETURN_TYPES = ("IMAGE", "INT", "BOOLEAN")
        RETURN_NAMES = ("frames", "frame_count", "has_video")
        OUTPUT_IS_LIST = (True, True, False)

        def load_gif(
            self,
            gif_path: str,
            force_rate: int,
            force_size: str,
            skip_first_frames: int,
            frame_load_cap: int,
            select_every_nth: int,
        ):
            from PIL import Image, ImageSequence

            image = Image.open(gif_path)
            frames = []
            total_frames_evaluated = -1

            if force_rate != 0:
                print(f"Force rate is not supported for gifs/webps")
            if frame_load_cap == 0:
                frame_load_cap = 999999999

            for i, frame in enumerate(ImageSequence.Iterator(image)):
                if i < skip_first_frames:
                    continue
                elif i >= skip_first_frames + frame_load_cap:
                    break
                else:
                    total_frames_evaluated += 1
                    if total_frames_evaluated % select_every_nth == 0:
                        frames.append(pil2tensor(frame.copy().convert("RGB")))

            images = torch.cat(frames, dim=0)

            if force_size != "Disabled":
                height = images.shape[1]
                width = images.shape[2]
                new_size = target_size(width, height, force_size)
                if new_size[0] != width or new_size[1] != height:
                    s = images.movedim(-1, 1)
                    s = common_upscale(s, new_size[0], new_size[1], "lanczos", "disabled")
                    images = s.movedim(1, -1)

            return (images, len(frames))

        def load_url(self, video: str, **kwargs):
            url = video.strip('"')

            if url == "":
                return (None, 0)

            if os.path.isfile(url):
                pass
            elif url.startswith("file://"):
                url = url[7:]
                url = os.path.abspath(url)

                if not os.path.isfile(url):
                    raise Exception(f"File {url} does not exist")

                if url.startswith(input_dir):
                    video = url[len(input_dir) + 1 :] + " [input]"
                elif url.startswith(output_dir):
                    video = url[len(output_dir) + 1 :] + " [output]"
                elif url.startswith(temp_dir):
                    video = url[len(temp_dir) + 1 :] + " [temp]"
                else:
                    # move file to temp_dir
                    import shutil

                    tempdir = os.path.join(temp_dir, "video")
                    if not os.path.exists(tempdir):
                        os.makedirs(tempfile, exist_ok=True)

                    filename = os.path.basename(url)
                    filepath = os.path.join(tempdir, filename)

                    i = 1
                    split = os.path.splitext(filename)
                    while os.path.exists(filepath):
                        filename = f"{split[0]} ({i}){split[1]}"
                        filepath = os.path.join(tempdir, filename)
                        i += 1

                    shutil.copy(url, filepath)
                    video = "video/" + filename + " [temp]"
            elif url.startswith("http://") or url.startswith("https://"):
                from torch.hub import download_url_to_file
                from urllib.parse import urlparse

                parts = urlparse(url)
                filename = os.path.basename(parts.path)
                tempfile = os.path.join(temp_dir, "video")
                if not os.path.exists(tempfile):
                    os.makedirs(tempfile, exist_ok=True)
                tempfile = os.path.join(tempfile, filename)

                print(f'Downloading: "{url}" to {tempfile}\n')
                download_url_to_file(url, tempfile, progress=True)

                video = "video/" + filename + " [temp]"
            elif url.startswith(("/view?", "/api/view?")):
                from urllib.parse import parse_qs

                qs_idx = url.find("?")
                qs = parse_qs(url[qs_idx + 1:])
                filename = qs.get("name", qs.get("filename", None))
                if filename is None:
                    raise Exception(f"Invalid url: {url}")

                filename = filename[0]
                subfolder = qs.get("subfolder", None)
                if subfolder is not None:
                    filename = os.path.join(subfolder[0], filename)

                dirtype = qs.get("type", ["input"])
                video = f"{filename} [{dirtype[0]}]"
            else:
                raise Exception(f"Invalid url: {url}")

            if ".gif [" in video.lower() or ".webp [" in video.lower():
                gif_path = folder_paths.get_annotated_filepath(video.strip('"'))
                res = self.load_gif(gif_path, **kwargs)
            else:
                res = self.load_video(video=video, **kwargs)

            return res

        def load(self, video: str, **kwargs):
            urls = video.strip().split("\n")

            videos = []
            frame_counts = []

            for url in urls:
                images, frame_count = self.load_url(url, **kwargs)
                if images is not None and frame_count > 0:
                    videos.append(images)
                    frame_counts.append(frame_count)

            has_video = len(videos) > 0
            if not has_video:
                image = torch.zeros((1, 64, 64, 3), dtype=torch.float32, device="cpu")
                videos.append(image)
                frame_counts.append(1)

            return (videos, frame_counts, has_video)

        @classmethod
        def IS_CHANGED(s, video: str, **kwargs):
            return video

        @classmethod
        def VALIDATE_INPUTS(s, **kwargs):
            return True

    NODE_CLASS_MAPPINGS["LoadVideoFromUrl"] = UtilLoadVideoFromUrl
    NODE_DISPLAY_NAME_MAPPINGS["LoadVideoFromUrl"] = "Load Video From Url"


except Exception as e:
    print(e)
