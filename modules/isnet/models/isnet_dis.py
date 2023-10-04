import torch
import torch.nn as nn
import torch.nn.functional as F

from .isnet import RSU4, RSU4F, RSU5, RSU6, RSU7, _upsample_like, ISNetBase


bce_loss = nn.BCELoss(size_average=True)
fea_loss = nn.MSELoss(size_average=True)
kl_loss = nn.KLDivLoss(size_average=True)
l1_loss = nn.L1Loss(size_average=True)
smooth_l1_loss = nn.SmoothL1Loss(size_average=True)


def muti_loss_fusion(preds, target):
    loss0 = 0.0
    loss = 0.0

    for i in range(0, len(preds)):
        # print("i: ", i, preds[i].shape)
        if preds[i].shape[2] != target.shape[2] or preds[i].shape[3] != target.shape[3]:
            # tmp_target = _upsample_like(target,preds[i])
            tmp_target = F.interpolate(target, size=preds[i].size()[2:], mode="bilinear", align_corners=True)
            loss = loss + bce_loss(preds[i], tmp_target)
        else:
            loss = loss + bce_loss(preds[i], target)
        if i == 0:
            loss0 = loss
    return loss0, loss


def muti_loss_fusion_kl(preds, target, dfs, fs, mode="MSE"):
    loss0 = 0.0
    loss = 0.0

    for i in range(0, len(preds)):
        # print("i: ", i, preds[i].shape)
        if preds[i].shape[2] != target.shape[2] or preds[i].shape[3] != target.shape[3]:
            # tmp_target = _upsample_like(target,preds[i])
            tmp_target = F.interpolate(target, size=preds[i].size()[2:], mode="bilinear", align_corners=True)
            loss = loss + bce_loss(preds[i], tmp_target)
        else:
            loss = loss + bce_loss(preds[i], target)
        if i == 0:
            loss0 = loss

    for i in range(0, len(dfs)):
        if mode == "MSE":
            loss = loss + fea_loss(dfs[i], fs[i])  ### add the mse loss of features as additional constraints
            # print("fea_loss: ", fea_loss(dfs[i],fs[i]).item())
        elif mode == "KL":
            loss = loss + kl_loss(F.log_softmax(dfs[i], dim=1), F.softmax(fs[i], dim=1))
            # print("kl_loss: ", kl_loss(F.log_softmax(dfs[i],dim=1),F.softmax(fs[i],dim=1)).item())
        elif mode == "MAE":
            loss = loss + l1_loss(dfs[i], fs[i])
            # print("ls_loss: ", l1_loss(dfs[i],fs[i]))
        elif mode == "SmoothL1":
            loss = loss + smooth_l1_loss(dfs[i], fs[i])
            # print("SmoothL1: ", smooth_l1_loss(dfs[i],fs[i]).item())

    return loss0, loss


class ISNetDIS(ISNetBase):
    def __init__(self, in_ch=3, out_ch=1):
        super(ISNetDIS, self).__init__(in_ch=in_ch, out_ch=out_ch)

        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

    def compute_loss_kl(self, preds, targets, dfs, fs, mode="MSE"):
        # return muti_loss_fusion(preds,targets)
        return muti_loss_fusion_kl(preds, targets, dfs, fs, mode=mode)

    def compute_loss(self, preds, targets):
        # return muti_loss_fusion(preds,targets)
        return muti_loss_fusion(preds, targets)

    def forward(self, x):
        hx = x

        hxin = self.conv_in(hx)
        # hx = self.pool_in(hxin)

        # stage 1
        hx1 = self.stage1(hxin)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)
        d1 = _upsample_like(d1, x)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, x)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, x)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, x)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, x)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, x)

        return [F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)], [
            hx1d,
            hx2d,
            hx3d,
            hx4d,
            hx5d,
            hx6,
        ]
