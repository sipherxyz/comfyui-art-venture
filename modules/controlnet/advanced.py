import os

import folder_paths
import comfy.sd
import comfy.controlnet

from ..utils import load_module

custom_nodes = folder_paths.get_folder_paths("custom_nodes")
advanced_cnet_dir_names = ["AdvancedControlNet", "ComfyUI-Advanced-ControlNet"]

def comfy_load_controlnet(module_path: str, **_):
    return comfy.controlnet.load_controlnet(module_path)

try:
    module_path = None

    for custom_node in custom_nodes:
        custom_node = (
            custom_node if not os.path.islink(custom_node) else os.readlink(custom_node)
        )
        for module_dir in advanced_cnet_dir_names:
            if module_dir in os.listdir(custom_node):
                module_path = os.path.abspath(os.path.join(custom_node, module_dir))
                break

    if module_path is None:
        raise Exception("Could not find AdvancedControlNet nodes")

    module_path = os.path.join(module_path, "control.py")
    module = load_module(module_path)
    print("Loaded AdvancedControlNet nodes from", module_path)

    comfy_load_controlnet = getattr(module, "load_controlnet")

except Exception as e:
    print(e)

