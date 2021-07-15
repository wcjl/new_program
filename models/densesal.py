import torch.nn as nn
from torchvision.models import densenet121, densenet169, densenet201, densenet161

from .base_model import BaseModel
from .readout_nets import upsampling_modules
from .densenet_efficient import densenet_cosine_264_k48

__all__ = ['densesal_vanilla', 'densesal121', 'densesal169', 'densesal201', 'densesal161',
    'densesal264_k48', 'densesal_single_fine_path', 'densesal_single_coarse_path',
    'densesal_NIx1', 'densesal_NIx2', 'densesal_NIx3', 'densesal_BIx1', 'densesal_BIx2', 'densesal_BIx3',
    'densesal_DCx1', 'densesal_DCx2', 'densesal_DCx3', 'densesal_SPCx1', 'densesal_SPCx2', 'densesal_SPCx3']


def _change_stride(module, target_name, stride):
    for n, m in module.named_modules():
        if n == target_name:
            m.stride = stride

def _densesal_NI(pretrained, scale_factor):
    main_net = densenet161(pretrained).features
    _change_stride(main_net, 'transition3.pool', stride=1)
    readout_net = upsampling_modules.NearestInterpolationModule(4416, scale_factor)
    model = BaseModel(main_net, readout_net)
    return model

def _densesal_BI(pretrained, scale_factor):
    main_net = densenet161(pretrained).features
    _change_stride(main_net, 'transition3.pool', stride=1)
    readout_net = upsampling_modules.BilinearInterpolationModule(4416, scale_factor)
    model = BaseModel(main_net, readout_net)
    return model

def _densesal_DC(pretrained, num_upsampling_layers):
    main_net = densenet161(pretrained).features
    _change_stride(main_net, 'transition3.pool', stride=1)
    readout_net = upsampling_modules.DeconvolutionModule(4416, num_upsampling_layers)
    model = BaseModel(main_net, readout_net)
    return model

def _densesal_SPC(pretrained, num_upsampling_layers):
    main_net = densenet161(pretrained).features
    _change_stride(main_net, 'transition3.pool', stride=1)
    readout_net = upsampling_modules.SubPixelConvModule(4416, num_upsampling_layers)
    model = BaseModel(main_net, readout_net)
    return model

def densesal121(pretrained=True):
    main_net = densenet121(pretrained).features
    readout_net = nn.Conv2d(1024, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net, single_fine_path=True)
    return model

def densesal169(pretrained=True):
    main_net = densenet169(pretrained).features
    readout_net = nn.Conv2d(1664, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net, single_fine_path=True)
    return model

def densesal201(pretrained=True):
    main_net = densenet201(pretrained).features
    readout_net = nn.Conv2d(1920, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net, single_fine_path=True)
    return model

def densesal161(pretrained=True):
    main_net = densenet161(pretrained).features
    readout_net = nn.Conv2d(2208, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net, single_fine_path=True)
    return model

def densesal264_k48(pretrained=True):
    main_net = densenet_cosine_264_k48(pretrained).features
    readout_net = nn.Conv2d(4032, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net, single_fine_path=True)
    return model

def densesal_vanilla(pretrained=True):
    main_net = densenet161(pretrained).features
    _change_stride(main_net, 'transition3.pool', stride=1)
    readout_net = nn.Conv2d(4416, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net)
    return model

def densesal_single_fine_path(pretrained=True):
    main_net = densenet161(pretrained).features
    _change_stride(main_net, 'transition3.pool', stride=1)
    readout_net = nn.Conv2d(2208, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net, single_fine_path=True)
    return model

def densesal_single_coarse_path(pretrained=True):
    main_net = densenet161(pretrained).features
    _change_stride(main_net, 'transition3.pool', stride=1)
    readout_net = nn.Conv2d(2208, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net, single_coarse_path=True)
    return model

def densesal_NIx1(pretrained=True):
    model = _densesal_NI(pretrained, scale_factor=2)
    return model

def densesal_NIx2(pretrained=True):
    model = _densesal_NI(pretrained, scale_factor=4)
    return model

def densesal_NIx3(pretrained=True):
    model = _densesal_NI(pretrained, scale_factor=8)
    return model

def densesal_BIx1(pretrained=True):
    model = _densesal_BI(pretrained, scale_factor=2)
    return model

def densesal_BIx2(pretrained=True):
    model = _densesal_BI(pretrained, scale_factor=4)
    return model

def densesal_BIx3(pretrained=True):
    model = _densesal_BI(pretrained, scale_factor=8)
    return model

def densesal_DCx1(pretrained=True):
    model = _densesal_DC(pretrained, num_upsampling_layers=1)
    return model

def densesal_DCx2(pretrained=True):
    model = _densesal_DC(pretrained, num_upsampling_layers=2)
    return model

def densesal_DCx3(pretrained=True):
    model = _densesal_DC(pretrained, num_upsampling_layers=3)
    return model

def densesal_SPCx1(pretrained=True):
    model = _densesal_SPC(pretrained, num_upsampling_layers=1)
    return model

def densesal_SPCx2(pretrained=True):
    model = _densesal_SPC(pretrained, num_upsampling_layers=2)
    return model

def densesal_SPCx3(pretrained=True):
    model = _densesal_SPC(pretrained, num_upsampling_layers=3)
    return model
