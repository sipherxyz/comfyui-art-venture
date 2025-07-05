import os
import io
import sys
import torch
import base64
import numpy as np
import importlib
import subprocess
from importlib.metadata import version, PackageNotFoundError
from PIL import Image

from .logger import logger


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_type = AnyType("*")


def ensure_package(package, version_required=None, install_package_name=None):
    # Try to import the package
    try:
        module = importlib.import_module(package)
        if version_required:
            try:
                installed_version = version(package)
                # Compare versions
                if _compare_versions(installed_version, version_required) < 0:
                    logger.info(
                        f"Package {package} is outdated (installed: {installed_version}, required: {version_required}). Upgrading now..."
                    )
                    _install_package(install_package_name or package, version_required)
            except PackageNotFoundError:
                _install_package(install_package_name or package, version_required)
    except ImportError:
        logger.info(f"Package {package} is not installed. Installing now...")
        _install_package(install_package_name or package, version_required)


def _compare_versions(version1, version2):
    """Compare two version strings."""
    def normalize(v):
        return [int(x) for x in v.split(".")]
    
    v1 = normalize(version1)
    v2 = normalize(version2)
    
    for i in range(max(len(v1), len(v2))):
        n1 = v1[i] if i < len(v1) else 0
        n2 = v2[i] if i < len(v2) else 0
        if n1 < n2:
            return -1
        elif n1 > n2:
            return 1
    return 0


def _install_package(package_name, version=None):
    """Install a package using pip."""
    if "python_embeded" in sys.executable or "python_embedded" in sys.executable:
        pip_install = [sys.executable, "-s", "-m", "pip", "install"]
    else:
        pip_install = [sys.executable, "-m", "pip", "install"]

    if version:
        package_name = f"{package_name}=={version}"

    subprocess.check_call(pip_install + [package_name])


def get_dict_attribute(dict_inst: dict, name_string: str, default=None):
    nested_keys = name_string.split(".")
    value = dict_inst

    for key in nested_keys:
        # Handle array indexing
        if key.startswith("[") and key.endswith("]"):
            try:
                index = int(key[1:-1])
                if not isinstance(value, (list, tuple)) or index >= len(value):
                    return default
                value = value[index]
            except (ValueError, TypeError):
                return default
        else:
            if not isinstance(value, dict):
                return default
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

    child = subprocess.Popen('fsutil reparsepoint query "{}"'.format(src), stdout=subprocess.PIPE)
    streamdata = child.communicate()[0]
    rc = child.returncode
    return rc == 0


def load_module(module_path, module_name=None):
    import importlib.util

    if module_name is None:
        module_name = os.path.basename(module_path)
        if os.path.isdir(module_path):
            module_path = os.path.join(module_path, "__init__.py")

    module_spec = importlib.util.spec_from_file_location(module_name, module_path)

    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    return module


def pil2numpy(image: Image.Image):
    return np.array(image).astype(np.float32) / 255.0


def numpy2pil(image: np.ndarray, mode=None):
    return Image.fromarray(np.clip(255.0 * image, 0, 255).astype(np.uint8), mode)


def pil2tensor(image: Image.Image):
    return torch.from_numpy(pil2numpy(image)).unsqueeze(0)


def tensor2pil(image: torch.Tensor, mode=None):
    return numpy2pil(image.cpu().numpy().squeeze(), mode=mode)


def tensor2bytes(image: torch.Tensor) -> bytes:
    return tensor2pil(image).tobytes()


def pil2base64(image: Image.Image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str
