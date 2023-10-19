import os
import cv2
import torch
import numpy as np
import insightface

import folder_paths

from .utils import any_type


ANALYSIS_MODEL = None
models_path = folder_paths.models_dir
insightface_path = os.path.join(models_path, "insightface")


def getAnalysisModel():
    global ANALYSIS_MODEL
    if ANALYSIS_MODEL is None:
        ANALYSIS_MODEL = insightface.app.FaceAnalysis(
            root=insightface_path,
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "genderage"],
        )
    return ANALYSIS_MODEL


def analyze_faces(img_data: np.ndarray, det_size=(640, 640)):
    face_analyser = getAnalysisModel()
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    return face_analyser.get(img_data)


class FaceAnalyze:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "face_index": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1}),
            },
            "optional": {"male_input": (any_type,), "female_input": (any_type,)},
        }

    RETURN_TYPES = ("STRING", "INT", any_type)
    RETURN_NAMES = ("gender", "age", "gender_input")
    FUNCTION = "analyze"
    CATEGORY = "Art Venture/Post Processing"

    def analyze(self, image: torch.Tensor, face_index: int, male_input=None, female_input=None):
        source_img = image.cpu().numpy().squeeze()
        source_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        source_faces = analyze_faces(source_img)
        if face_index >= len(source_faces):
            raise ValueError(f"Face index {face_index} is out of range (max: {len(source_faces)-1})")

        # face = source_faces[face_index]
        for i, face in enumerate(source_faces):
            print(f"Face {i}", face.sex, face.age, face.gender)


NODE_CLASS_MAPPINGS = {
    "FaceAnalyze": FaceAnalyze,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceAnalyze": "Face Analyze"
}
