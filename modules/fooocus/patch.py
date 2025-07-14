import torch
import comfy.model_base
import comfy.ldm.modules.diffusionmodules.openaimodel
import comfy.samplers


from .anisotropic import bilateral_blur

sharpness = 2.0

original_unet_forward = comfy.ldm.modules.diffusionmodules.openaimodel.UNetModel.forward
original_sdxl_encode_adm = comfy.model_base.SDXL.encode_adm


def unet_forward_patched(
    self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs
):
    x0 = original_unet_forward(
        self,
        x,
        timesteps=timesteps,
        context=context,
        y=y,
        control=control,
        transformer_options=transformer_options,
        **kwargs
    )
    uc_mask = torch.Tensor(transformer_options["cond_or_uncond"]).to(x0).float()[:, None, None, None]

    alpha = 1.0 - (timesteps / 999.0)[:, None, None, None].clone()
    alpha *= 0.001 * sharpness
    degraded_x0 = bilateral_blur(x0) * alpha + x0 * (1.0 - alpha)

    # FIX: uc_mask is not always the same size as x0
    if uc_mask.shape[0] < x0.shape[0]:
        uc_mask = uc_mask.repeat(int(x0.shape[0] / uc_mask.shape[0]), 1, 1, 1)

    x0 = x0 * uc_mask + degraded_x0 * (1.0 - uc_mask)

    return x0


def sdxl_encode_adm_patched(self, **kwargs):
    clip_pooled = kwargs["pooled_output"]
    width = kwargs.get("width", 768)
    height = kwargs.get("height", 768)
    crop_w = kwargs.get("crop_w", 0)
    crop_h = kwargs.get("crop_h", 0)
    target_width = kwargs.get("target_width", width)
    target_height = kwargs.get("target_height", height)

    if kwargs.get("prompt_type", "") == "negative":
        width *= 0.8
        height *= 0.8
    elif kwargs.get("prompt_type", "") == "positive":
        width *= 1.5
        height *= 1.5

    out = []
    out.append(self.embedder(torch.Tensor([height])))
    out.append(self.embedder(torch.Tensor([width])))
    out.append(self.embedder(torch.Tensor([crop_h])))
    out.append(self.embedder(torch.Tensor([crop_w])))
    out.append(self.embedder(torch.Tensor([target_height])))
    out.append(self.embedder(torch.Tensor([target_width])))
    flat = torch.flatten(torch.cat(out))[None,]
    return torch.cat((clip_pooled.to(flat.device), flat), dim=1)


def patch_all():
    comfy.model_base.SDXL.encode_adm = sdxl_encode_adm_patched
    comfy.ldm.modules.diffusionmodules.openaimodel.UNetModel.forward = unet_forward_patched


def unpatch_all():
    comfy.model_base.SDXL.encode_adm = original_sdxl_encode_adm
    comfy.ldm.modules.diffusionmodules.openaimodel.UNetModel.forward = original_unet_forward
