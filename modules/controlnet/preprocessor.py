import os
from typing import Dict

import folder_paths

from ..utils import load_module

custom_nodes = folder_paths.get_folder_paths("custom_nodes")
preprocessors_dir_names = ["ControlNetPreprocessors", "comfyui_controlnet_aux"]

preprocessors: list[str] = []
_preprocessors_map = {
    "canny": "CannyEdgePreprocessor",
    "canny_pyra": "PyraCannyPreprocessor",
    "lineart": "LineArtPreprocessor",
    "lineart_anime": "AnimeLineArtPreprocessor",
    "lineart_manga": "Manga2Anime_LineArt_Preprocessor",
    "lineart_any": "AnyLineArtPreprocessor_aux",
    "scribble": "ScribblePreprocessor",
    "scribble_xdog": "Scribble_XDoG_Preprocessor",
    "scribble_pidi": "Scribble_PiDiNet_Preprocessor",
    "scribble_hed": "FakeScribblePreprocessor",
    "hed": "HEDPreprocessor",
    "pidi": "PiDiNetPreprocessor",
    "mlsd": "M-LSDPreprocessor",
    "pose": "DWPreprocessor",
    "openpose": "OpenposePreprocessor",
    "dwpose": "DWPreprocessor",
    "pose_dense": "DensePosePreprocessor",
    "pose_animal": "AnimalPosePreprocessor",
    "normalmap_bae": "BAE-NormalMapPreprocessor",
    "normalmap_dsine": "DSINE-NormalMapPreprocessor",
    "normalmap_midas": "MiDaS-NormalMapPreprocessor",
    "depth": "DepthAnythingV2Preprocessor",
    "depth_anything": "DepthAnythingPreprocessor",
    "depth_anything_v2": "DepthAnythingV2Preprocessor",
    "depth_anything_zoe": "Zoe_DepthAnythingPreprocessor",
    "depth_zoe": "Zoe-DepthMapPreprocessor",
    "depth_midas": "MiDaS-DepthMapPreprocessor",
    "depth_leres": "LeReS-DepthMapPreprocessor",
    "depth_metric3d": "Metric3D-DepthMapPreprocessor",
    "depth_meshgraphormer": "MeshGraphormer-DepthMapPreprocessor",
    "seg_ofcoco": "OneFormer-COCO-SemSegPreprocessor",
    "seg_ofade20k": "OneFormer-ADE20K-SemSegPreprocessor",
    "seg_ufade20k": "UniFormer-SemSegPreprocessor",
    "seg_animeface": "AnimeFace_SemSegPreprocessor",
    "shuffle": "ShufflePreprocessor",
    "teed": "TEEDPreprocessor",
    "color": "ColorPreprocessor",
    "sam": "SAMPreprocessor",
    "tile": "TilePreprocessor"
}


def apply_preprocessor(image, preprocessor, resolution=512):
    raise NotImplementedError("apply_preprocessor is not implemented")


try:
    module_path = None

    for custom_node in custom_nodes:
        custom_node = custom_node if not os.path.islink(custom_node) else os.readlink(custom_node)
        for module_dir in preprocessors_dir_names:
            if module_dir in os.listdir(custom_node):
                module_path = os.path.abspath(os.path.join(custom_node, module_dir))
                break

    if module_path is None:
        raise Exception("Could not find ControlNetPreprocessors nodes")

    module = load_module(module_path)
    print("Loaded ControlNetPreprocessors nodes from", module_path)

    nodes: Dict = getattr(module, "NODE_CLASS_MAPPINGS")
    available_preprocessors: list[str] = getattr(module, "PREPROCESSOR_OPTIONS")

    AIO_Preprocessor = nodes.get("AIO_Preprocessor", None)
    if AIO_Preprocessor is None:
        raise Exception("Could not find AIO_Preprocessor node")

    for name, preprocessor in _preprocessors_map.items():
        if preprocessor in available_preprocessors:
            preprocessors.append(name)

    aio_preprocessor = AIO_Preprocessor()

    def apply_preprocessor(image, preprocessor, resolution=512):
        if preprocessor == "None":
            return image

        if preprocessor not in preprocessors:
            raise Exception(f"Preprocessor {preprocessor} is not implemented")

        preprocessor_cls = _preprocessors_map[preprocessor]
        args = {"preprocessor": preprocessor_cls, "image": image, "resolution": resolution}

        function_name = AIO_Preprocessor.FUNCTION
        res = getattr(aio_preprocessor, function_name)(**args)
        if isinstance(res, dict):
            res = res["result"]

        return res[0]

except Exception as e:
    print(e)
