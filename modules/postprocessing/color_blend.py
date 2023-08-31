# Adapt from https://github.com/Stability-AI/stability-ComfyUI-nodes/blob/master/color_blend.py

# Color blend node by Yam Levi
# Property of Stability AI
import cv2
import torch
import numpy as np

import comfy.utils


def color_blend(bw_layer, color_layer):
    # Convert the color layer to LAB color space
    color_lab = cv2.cvtColor(color_layer, cv2.COLOR_BGR2Lab)
    # Convert the black and white layer to grayscale
    bw_layer_gray = cv2.cvtColor(bw_layer, cv2.COLOR_BGR2GRAY)
    # Replace the luminosity (L) channel in the color image with the black and white luminosity
    _, color_a, color_b = cv2.split(color_lab)
    blended_lab = cv2.merge((bw_layer_gray, color_a, color_b))
    # Convert the blended LAB image back to BGR color space
    blended_result = cv2.cvtColor(blended_lab, cv2.COLOR_Lab2BGR)
    return blended_result


class ColorBlend:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bw_layer": ("IMAGE",),
                "color_layer": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "color_blending_mode"
    CATEGORY = "Art Venture/Post Processing"

    def color_blending_mode(self, bw_layer, color_layer):
        if bw_layer.shape[0] < color_layer.shape[0]:
            bw_layer = bw_layer.repeat(color_layer.shape[0], 1, 1, 1)[
                : color_layer.shape[0]
            ]
        if bw_layer.shape[0] > color_layer.shape[0]:
            color_layer = color_layer.repeat(bw_layer.shape[0], 1, 1, 1)[
                : bw_layer.shape[0]
            ]

        batch_size, *_ = bw_layer.shape
        tensor_output = torch.empty_like(bw_layer)

        image1 = bw_layer.cpu()
        image2 = color_layer.cpu()
        if image1.shape != image2.shape:
            image2 = image2.permute(0, 3, 1, 2)
            image2 = comfy.utils.common_upscale(
                image2,
                image1.shape[2],
                image1.shape[1],
                upscale_method="bicubic",
                crop="center",
            )
            image2 = image2.permute(0, 2, 3, 1)
        image1 = (image1 * 255).to(torch.uint8).numpy()
        image2 = (image2 * 255).to(torch.uint8).numpy()

        for i in range(batch_size):
            blend = color_blend(image1[i], image2[i])
            blend = np.stack([blend])
            tensor_output[i : i + 1] = (
                torch.from_numpy(blend.transpose(0, 3, 1, 2)) / 255.0
            ).permute(0, 2, 3, 1)

        return (tensor_output,)
