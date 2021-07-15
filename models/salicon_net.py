import torch.nn as nn
from torchvision.models import vgg16

from .base_model import BaseModel
from .readout_nets import upsampling_modules

__all__ = ['salicon_net', 'salicon_net_single_fine_path', 'salicon_net_single_coarse_path', 'salicon_net_BIx1', 'salicon_net_BIx2', 'salicon_net_BIx3']


def _salicon_net_BI(pretrained, scale_factor):
    main_net = vgg16(pretrained).features
    readout_net = upsampling_modules.BilinearInterpolationModule(1024, scale_factor)
    model = BaseModel(main_net, readout_net)
    return model

def salicon_net(pretrained=True):
    main_net = vgg16(pretrained).features
    readout_net = nn.Conv2d(1024, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net)
    return model

def salicon_net_single_fine_path(pretrained=True):
    main_net = vgg16(pretrained).features
    readout_net = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net, single_fine_path=True)
    return model

def salicon_net_single_coarse_path(pretrained=True):
    main_net = vgg16(pretrained).features
    readout_net = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net, single_coarse_path=True)
    return model

def salicon_net_BIx1(pretrained=True):
    model = _salicon_net_BI(pretrained, scale_factor=2)
    return model

def salicon_net_BIx2(pretrained=True):
    model = _salicon_net_BI(pretrained, scale_factor=4)
    return model

def salicon_net_BIx3(pretrained=True):
    model = _salicon_net_BI(pretrained, scale_factor=8)
    return model
