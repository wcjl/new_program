from torchvision.models import densenet161

from .base_model import BaseModel
from .readout_nets.pyramid_pooling_module import PyramidPoolingModule

__all__ = ['densesal_v2', 'densesal_default_ppm']


def _change_stride(module, target_name, stride):
    for n, m in module.named_modules():
        if n == target_name:
            m.stride = stride

def densesal_v2(pretrained=True):
    main_net = densenet161(pretrained).features
    _change_stride(main_net, 'transition3.pool', stride=1)
    readout_net = PyramidPoolingModule(4416, reduction_rate=16, sizes=(1,2,3,6))
    model = BaseModel(main_net, readout_net)
    return model

def densesal_default_ppm(pretrained=True):
    main_net = densenet161(pretrained).features
    _change_stride(main_net, 'transition3.pool', stride=1)
    readout_net = PyramidPoolingModule(4416, reduction_rate='default', sizes=(1,2,3,6))
    model = BaseModel(main_net, readout_net)
    return model
