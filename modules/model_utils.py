import os
import re
import torch
import hashlib
import urllib.request
import urllib.error
from tqdm import tqdm
from urllib.parse import urlparse
from typing import Dict, Optional


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


def calculate_sha(file: str, force=False) -> Optional[str]:
    sha_file = f"{file}.sha"

    # Check if the .sha file exists
    if not force and os.path.exists(sha_file):
        try:
            with open(sha_file, "r") as f:
                stored_hash = f.read().strip()
                if stored_hash:
                    return stored_hash
        except IOError as e:
            print(f"Failed to read hash: {e}")

    # Calculate the hash if the .sha file doesn't exist or is empty
    try:
        with open(file, "rb") as fp:
            file_hash = hashlib.sha256()
            while chunk := fp.read(8192):
                file_hash.update(chunk)
            calculated_hash = file_hash.hexdigest()

            # Write the calculated hash to the .sha file
            try:
                with open(sha_file, "w") as f:
                    f.write(calculated_hash)
            except IOError as e:
                print(f"Failed to write hash to {sha_file}: {e}")

            return calculated_hash
    except IOError as e:
        print(f"Failed to read file {file}: {e}")
        return None


def download_file(url: str, dst: str, sha256sum: Optional[str] = None) -> Dict[str, Optional[str]]:
    """
    Downloads a file from a URL to a destination path, optionally verifying its SHA-256 checksum.

    :param url: URL of the file to download
    :param dst: Destination path to save the downloaded file
    :param sha256sum: Optional SHA-256 checksum to verify the downloaded file
    :return: Dictionary with file path, download status, calculated checksum, and checksum match status
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    file_exists = os.path.isfile(dst)
    file_checksum = None
    checksum_match = None
    downloaded = False

    try:
        if file_exists:
            file_checksum = calculate_sha(dst)
            if sha256sum:
                checksum_match = file_checksum == sha256sum
                if not checksum_match:
                    os.remove(dst)

        if not file_exists or checksum_match == False:
            with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=dst.split("/")[-1]) as t:

                def reporthook(blocknum, blocksize, totalsize):
                    if t.total is None and totalsize > 0:
                        t.total = totalsize
                    read_so_far = blocknum * blocksize
                    t.update(max(0, read_so_far - t.n))

                urllib.request.urlretrieve(url, dst, reporthook=reporthook)
                downloaded = True

                file_checksum = calculate_sha(dst, force=True)
                if sha256sum:
                    checksum_match = file_checksum == sha256sum

    except urllib.error.URLError as ex:
        print("Download failed:", ex)
        if os.path.isfile(dst):
            os.remove(dst)
    except Exception as ex:
        print("An error occurred:", ex)
    finally:
        return {"file": dst, "downloaded": downloaded, "sha": file_checksum, "match": checksum_match}


def load_jit_torch_file(model_path: str):
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()
    return model
