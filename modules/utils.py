import os
import io
import sys
import torch
import base64
import numpy as np
import importlib
import subprocess
import pkg_resources
from pkg_resources import parse_version
from PIL import Image

from .logger import logger


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_type = AnyType("*")


def is_uv_installed():
    """
    Check if UV (Astral) is installed in the current Python environment.

    This function attempts to import the UV module. If the import is successful, 
    it returns True, indicating that UV is installed. If the import fails, it 
    returns False, indicating that UV is not installed.

    Returns:
        bool: True if UV is installed, False otherwise.
    """
    try:
        import uv
        return True
    except ImportError:
        return False


def ensure_package(package, version=None, install_package_name=None):
    """
    Ensure that a specific package is installed in the environment.

    This function checks if the given package is already installed. If it is not, 
    it attempts to install it. If a specific version is provided, it ensures the 
    installed version is up to date. The installation can be done using either pip 
    or UV (Astral) depending on the user's preference or environment.

    Args:
        package (str): The name of the package to check or install.
        version (str, optional): The required version of the package. Default is None.
        install_package_name (str, optional): The package name to install if different 
                                               from `package`. Default is None.
        use_uv (bool, optional): Flag to force the use of UV (Astral) for installation. 
                                  If None, the function will check if UV is installed 
                                  automatically. Default is None.

    Raises:
        subprocess.CalledProcessError: If the installation process fails.
    """
    # Check if UV should be used for installation or fallback to pip
    use_uv = is_uv_installed()

    # Construct the installation command based on the selected method
    if use_uv:
        install_command = _construct_uv_command(install_package_name or package, version)
    else:
        install_command = _construct_pip_command(install_package_name or package, version)

    # Attempt to import the package and install it if necessary
    try:
        module = importlib.import_module(package)
    except ImportError:
        logger.info(f"Package {package} is not installed. Installing now...")
        subprocess.check_call(install_command)
    else:
        # Check if the installed version is up-to-date
        if version:
            installed_version = pkg_resources.get_distribution(package).version
            if parse_version(installed_version) < parse_version(version):
                logger.info(
                    f"Package {package} is outdated (installed: {installed_version}, required: {version}). Upgrading now..."
                )
                subprocess.check_call(install_command)


def _construct_uv_command(package_name, version=None):
    """
    Construct the installation command for UV (Astral).

    This function builds a command to install a package using UV (Astral) if it is available. 
    If a specific version is provided, it is included in the command.

    Args:
        package_name (str): The name of the package to install.
        version (str, optional): The version of the package to install. Default is None.

    Returns:
        list: The command to execute for UV installation.
    """
    if "python_embeded" in sys.executable or "python_embedded" in sys.executable:
        uv_install = [sys.executable, "-s", "-m", "uv", "add"]
    else:
        uv_install = [sys.executable, "-m", "uv", "add"]

    if version:
        package_name = f"{package_name}{version}"

    return uv_install + [package_name]


def _construct_pip_command(package_name, version=None):
    """
    Construct the installation command for pip.

    This function builds a command to install a package using pip. If a specific 
    version is provided, it is included in the command.

    Args:
        package_name (str): The name of the package to install.
        version (str, optional): The version of the package to install. Default is None.

    Returns:
        list: The command to execute for pip installation.
    """
    if "python_embeded" in sys.executable or "python_embedded" in sys.executable:
        pip_install = [sys.executable, "-s", "-m", "pip", "install"]
    else:
        pip_install = [sys.executable, "-m", "pip", "install"]

    if version:
        package_name = f"{package_name}=={version}"

    return pip_install + [package_name]


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
