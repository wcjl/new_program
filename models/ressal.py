import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

from .base_model import BaseModel
from .readout_nets import upsampling_modules

__all__ = ['ressal18', 'ressal34', 'ressal50', 'ressal101', 'ressal152']


class FeatureExtractor(nn.Sequential):
    def __init__(self, resnet):
        super(FeatureExtractor, self).__init__()
        self.add_module('conv1', resnet.conv1)
        self.add_module('bn1', resnet.bn1)
        self.add_module('relu', resnet.relu)
        self.add_module('maxpool', resnet.maxpool)
        self.add_module('layer1', resnet.layer1)
        self.add_module('layer2', resnet.layer2)
        self.add_module('layer3', resnet.layer3)
        self.add_module('layer4', resnet.layer4)


def ressal18(pretrained=True):
    main_net = FeatureExtractor(resnet18(pretrained))
    readout_net = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net, single_fine_path=True)
    return model

def ressal34(pretrained=True):
    main_net = FeatureExtractor(resnet34(pretrained))
    readout_net = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net, single_fine_path=True)
    return model

def ressal50(pretrained=True):
    main_net = FeatureExtractor(resnet50(pretrained))
    readout_net = nn.Conv2d(2048, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net, single_fine_path=True)
    return model

def ressal101(pretrained=True):
    main_net = FeatureExtractor(resnet101(pretrained))
    readout_net = nn.Conv2d(2048, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net, single_fine_path=True)
    return model

def ressal152(pretrained=True):
    main_net = FeatureExtractor(resnet152(pretrained))
    readout_net = nn.Conv2d(2048, 1, kernel_size=1, stride=1, padding=0)
    model = BaseModel(main_net, readout_net, single_fine_path=True)
    return model

