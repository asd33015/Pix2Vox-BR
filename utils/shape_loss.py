import torch
import numpy as np


def proj_Front_mask(vol):
    # print(vol.shape)
    proj,_ = torch.max(vol, dim=4)
    # print(proj)
    # mask0 = cv2.resize(mask0, (mask0.shape[1] * 2, mask0.shape[0] * 2))
    # mask0 = cv2.resize(mask0, (mask0.shape[1] * 2, mask0.shape[0] * 2))
    true = torch.ones_like(proj, requires_grad=True).cuda()
    false = torch.zeros_like(proj, requires_grad=True).cuda()
    mask = torch.where(proj >0.4,true,false)
    # print(mask)
    # kernel = np.ones((3, 3), np.float32)
    # mask0 = cv2.erode(mask0, kernel, iterations=2)
    return mask

def proj_Side_mask(vol):
    # print(vol.shape)
    proj,_ = torch.max(vol, dim=2)
    # print(proj)
    # mask0 = cv2.resize(mask0, (mask0.shape[1] * 2, mask0.shape[0] * 2))
    # mask0 = cv2.resize(mask0, (mask0.shape[1] * 2, mask0.shape[0] * 2))
    true = torch.ones_like(proj, requires_grad=True).cuda()
    false = torch.zeros_like(proj, requires_grad=True).cuda()
    mask = torch.where(proj >0.4,true,false)
    # print(mask)
    # kernel = np.ones((3, 3), np.float32)
    # mask0 = cv2.erode(mask0, kernel, iterations=2)
    return mask

def proj_Front_mask_E(vol):
    # print(vol.shape)
    proj,_ = torch.max(vol, dim=3)
    # print(proj)
    # mask0 = cv2.resize(mask0, (mask0.shape[1] * 2, mask0.shape[0] * 2))
    # mask0 = cv2.resize(mask0, (mask0.shape[1] * 2, mask0.shape[0] * 2))
    true = torch.ones_like(proj, requires_grad=True).cuda()
    false = torch.zeros_like(proj, requires_grad=True).cuda()
    mask = torch.where(proj >0.4,true,false)
    # print(mask)
    # kernel = np.ones((3, 3), np.float32)
    # mask0 = cv2.erode(mask0, kernel, iterations=2)
    return mask

def proj_Side_mask_E(vol):
    # print(vol.shape)
    proj,_ = torch.max(vol, dim=1)
    # print(proj)
    # mask0 = cv2.resize(mask0, (mask0.shape[1] * 2, mask0.shape[0] * 2))
    # mask0 = cv2.resize(mask0, (mask0.shape[1] * 2, mask0.shape[0] * 2))
    true = torch.ones_like(proj, requires_grad=True).cuda()
    false = torch.zeros_like(proj, requires_grad=True).cuda()
    mask = torch.where(proj >0.4,true,false)
    # print(mask)
    # kernel = np.ones((3, 3), np.float32)
    # mask0 = cv2.erode(mask0, kernel, iterations=2)
    return mask

def shape_loss(pred,ground_truth):
    # print(pred)
    # print(ground_truth)
    # print(ground_truth.shape)
    sil_loss_sv = 100* (-0.7*(ground_truth*torch.log(pred+1e-8))-0.3*(1-ground_truth)*torch.log(1-pred+1e-8))

    return sil_loss_sv.mean()