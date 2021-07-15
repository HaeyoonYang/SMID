
import subprocess
import math
import logging
import numpy as np
import torch
import torch.nn as nn
from skimage.metrics.simple_metrics import peak_signal_noise_ratio

def weights_init_kaiming(lyr):
    r"""Initializes weights of the model according to the "He" initialization
    method described in "Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification" - He, K. et al. (2015), using a
    normal distribution.
    This function is to be called by the torch.nn.Module.apply() method,
    which applies weights_init_kaiming() to every layer of the model.
    """
    classname = lyr.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(lyr.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(lyr.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        lyr.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).\
            clamp_(-0.025, 0.025)
        nn.init.constant(lyr.bias.data, 0.0)

def batch_psnr(img, imclean, data_range):
    r"""
    Computes the PSNR along the batch dimension (not pixel-wise)

    Args:
        img: a `torch.Tensor` containing the restored image
        imclean: a `torch.Tensor` containing the reference image
        data_range: The data range of the input image (distance between
            minimum and maximum possible values). By default, this is estimated
            from the image data-type.
    """
    img_cpu = img.data.cpu().numpy().astype(np.float32)
    imgclean = imclean.data.cpu().numpy().astype(np.float32)
    psnr = 0
    for i in range(img_cpu.shape[0]):
        psnr += peak_signal_noise_ratio(imgclean[i, :, :, :], img_cpu[i, :, :, :], data_range=data_range)
    return psnr/img_cpu.shape[0]

def data_augmentation(image, mode):
    r"""Performs dat augmentation of the input image

    Args:
        image: a io image
        mode: int. Choice of transformation to apply to the image
            0 - no transformation
            1 - flip up and down
            2 - rotate counterwise 90 degree
            3 - rotate 90 degree and flip up and down
            4 - rotate 180 degree
            5 - rotate 180 degree and flip
            6 - rotate 270 degree
            7 - rotate 270 degree and flip
    """
    out = image.numpy()
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flip(out, 1)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out, axes=(1,2))
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out, axes=(1,2))
        out = np.flip(out, 1)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2, axes=(1,2))
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2, axes=(1,2))
        out = np.flip(out, 1)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3, axes=(1,2))
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3, axes=(1,2))
        out = np.flip(out, 1)
    else:
        raise Exception('Invalid choice of image transformation')
    return torch.from_numpy(out.copy())

def get_one_hot(targets, nb_classes):
    targets = np.clip(targets, 0, nb_classes-1)
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    res = res.reshape(list(targets.shape)+[nb_classes])
    res = res.squeeze()
    res = res.transpose(2,0,1)
    res = res[0:nb_classes-1][:][:]
    return torch.from_numpy(res).float()