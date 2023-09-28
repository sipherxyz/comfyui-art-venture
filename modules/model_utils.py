import os
import re
import torch
from urllib.parse import urlparse


def natural_sort_key(s, regex=re.compile("([0-9]+)")):
    return [int(text) if text.isdigit() else text.lower() for text in regex.split(s)]


def walk_files(path, allowed_extensions=None):
    if not os.path.exists(path):
        return

    if allowed_extensions is not None:
        allowed_extensions = set(allowed_extensions)

    items = list(os.walk(path, followlinks=True))
    items = sorted(items, key=lambda x: natural_sort_key(x[0]))

    for root, _, files in items:
        for filename in sorted(files, key=natural_sort_key):
            if allowed_extensions is not None:
                _, ext = os.path.splitext(filename)
                if ext not in allowed_extensions:
                    continue

            # Skip hidden files
            if "/." in root or "\\." in root:
                continue

            yield os.path.join(root, filename)


def load_file_from_url(
    url: str,
    *,
    model_dir: str,
    progress: bool = True,
    file_name: str | None = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        from torch.hub import download_url_to_file

        download_url_to_file(url, cached_file, progress=progress)
    return cached_file


def download_model(
    model_path: str,
    model_url: str = None,
    ext_filter=None,
    download_name=None,
    ext_blacklist=None,
) -> list:
    """
    A one-and done loader to try finding the desired models in specified directories.

    @param download_name: Specify to download from model_url immediately.
    @param model_url: If no other models are found, this will be downloaded on upscale.
    @param model_path: The location to store/find models in.
    @param ext_filter: An optional list of filename extensions to filter by
    @return: A list of paths containing the desired model(s)
    """
    output = []

    try:
        for full_path in walk_files(model_path, allowed_extensions=ext_filter):
            if os.path.islink(full_path) and not os.path.exists(full_path):
                print(f"Skipping broken symlink: {full_path}")
                continue
            if ext_blacklist is not None and any(full_path.endswith(x) for x in ext_blacklist):
                continue
            if full_path not in output:
                output.append(full_path)

        if model_url is not None and len(output) == 0:
            if download_name is not None:
                output.append(load_file_from_url(model_url, model_dir=model_path, file_name=download_name))
            else:
                output.append(model_url)

    except Exception:
        pass

    return output


def load_jit_torch_file(model_path: str):
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()
    return model
