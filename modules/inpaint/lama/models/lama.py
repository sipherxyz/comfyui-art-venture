import os

import cv2
import numpy as np
import torch

from ....image_utils import norm_img
from ....model_utils import load_jit_torch_file
from .base import InpaintModel


class LaMa(InpaintModel):
    name = "lama"
    pad_mod = 8

    def init_model(self, device, model_path: str, is_auto_mode=False):
        self.model = load_jit_torch_file(model_path)
        self.is_auto_mode = is_auto_mode

        if not is_auto_mode:
            self.model = self.model.to(device)

    def forward(self, image, mask, _):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W]
        return: BGR IMAGE
        """
        if self.is_auto_mode:
            self.model = self.model.to(self.device)

        try:
            image = norm_img(image)
            mask = norm_img(mask)

            mask = (mask > 0) * 1
            image = torch.from_numpy(image).unsqueeze(0).to(self.device)
            mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)

            inpainted_image = self.model(image, mask)

            cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
            cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
            cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
            return cur_res
        finally:
            if self.is_auto_mode:
                self.model = self.model.cpu()
