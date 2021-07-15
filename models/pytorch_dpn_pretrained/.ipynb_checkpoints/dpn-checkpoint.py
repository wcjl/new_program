""" PyTorch implementation of DualPathNetworks
Based on original MXNet implementation https://github.com/cypw/DPNs with
many ideas from another PyTorch implementation https://github.com/oyam/pytorch-DPNs.

This implementation is compatible with the pretrained weights
from cypw's MXNet implementation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

from .adaptive_avgmax_pool import adaptive_avgmax_pool2d
from .convert_from_mxnet import convert_from_mxnet, has_mxnet


__all__ = ['DPN', 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn131', 'dpn107']


# If anyone able to provide direct link hosting, more than happy to fill these out.. -rwightman
model_urls = {
     'dpn68':
         'https://s3.amazonaws.com/dpn-pytorch-weights/dpn68-66bebafa7.pth',
     'dpn68b-extra':
         'https://s3.amazonaws.com/dpn-pytorch-weights/'
         'dpn68b_extra-84854c156.pth',
     'dpn92': '',
     'dpn92-extra':
         'https://s3.amazonaws.com/dpn-pytorch-weights/'
         'dpn92_extra-b040e4a9b.pth',
     'dpn98':
         'https://s3.amazonaws.com/dpn-pytorch-weights/dpn98-5b90dec4d.pth',
     'dpn131':
         'https://s3.amazonaws.com/dpn-pytorch-weights/dpn131-71dfe43e0.pth',
     'dpn107-extra':
         'https://s3.amazonaws.com/dpn-pytorch-weights/'
         'dpn107_extra-1ac7121e2.pth'
}

model_files = {
     'dpn68':
         '/home/wuchenjunlin/new_program/dpn_wight/dpn68-66bebafa7.pth',
     'dpn68b-extra':
         '/home/wuchenjunlin/new_program/dpn_wight/dpn68b_extra-84854c156.pth',
     'dpn92-extra':
         '/home/wuchenjunlin/new_program/dpn_wight/dpn92_extra-b040e4a9b.pth',
     'dpn98':
         '/home/wuchenjunlin/new_program/dpn_wight/dpn98-5b90dec4d.pth',
     'dpn131':
         '/home/wuchenjunlin/new_program/dpn_wight/dpn131-71dfe43e0.pth',
     'dpn107-extra':
         '/home/wuchenjunlin/new_program/dpn_wight/dpn107_extra-1ac7121e2.pth'
}


def dpn68(num_classes=1000, pretrained=False, test_time_pool=True):
    model = DPN(
        small=True, num_init_features=10, k_r=128, groups=32,
        k_sec=(3, 4, 12, 3), inc_sec=(16, 32, 32, 64),
        num_classes=num_classes, test_time_pool=test_time_pool)
    if pretrained:
        PATH = model_files['dpn68']
        if os.path.exists(PATH):
             model. load_state_dict(torch.load(PATH)) 
        elif has_mxnet and os.path.exists('./pretrained/'):
            convert_from_mxnet(model, checkpoint_prefix='./pretrained/dpn68')
        else:
            assert False, "Unable to load a pretrained model"
    return model


def dpn68b(num_classes=1000, pretrained=False, test_time_pool=True):
    model = DPN(
        small=True, num_init_features=10, k_r=128, groups=32,
        b=True, k_sec=(3, 4, 12, 3), inc_sec=(16, 32, 32, 64),
        num_classes=num_classes, test_time_pool=test_time_pool)
    if pretrained:
        PATH = model_files['dpn68b-extra']
        if os.path.exists(PATH):
             model. load_state_dict(torch.load(PATH)) 
        elif has_mxnet and os.path.exists('./pretrained/'):
            convert_from_mxnet(model, checkpoint_prefix='./pretrained/dpn68-extra')
        else:
            assert False, "Unable to load a pretrained model"
    return model


def dpn92(num_classes=1000, pretrained=False, test_time_pool=True, extra=True):
    model = DPN(
        num_init_features=64, k_r=96, groups=32,
        k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128),
        num_classes=num_classes, test_time_pool=test_time_pool)
    if pretrained:
        # there are both imagenet 5k trained, 1k finetuned 'extra' weights
        # and normal imagenet 1k trained weights for dpn92
        
        PATH = model_files['dpn92-extra']
        if os.path.exists(PATH):
             model. load_state_dict(torch.load(PATH)) 
        elif has_mxnet and os.path.exists('./pretrained/'):
            convert_from_mxnet(model, checkpoint_prefix='./pretrained/' + key)
        else:
            assert False, "Unable to load a pretrained model"
    return model


def dpn98(num_classes=1000, pretrained=False, test_time_pool=True):
    model = DPN(
        num_init_features=96, k_r=160, groups=40,
        k_sec=(3, 6, 20, 3), inc_sec=(16, 32, 32, 128),
        num_classes=num_classes, test_time_pool=test_time_pool)
    if pretrained:
        PATH = model_files['dpn98']
        if os.path.exists(PATH):
             model. load_state_dict(torch.load(PATH)) 
        elif has_mxnet and os.path.exists('./pretrained/'):
            convert_from_mxnet(model, checkpoint_prefix='./pretrained/dpn98')
        else:
            assert False, "Unable to load a pretrained model"
    return model


def dpn131(num_classes=1000, pretrained=False, test_time_pool=True):
    model = DPN(
        num_init_features=128, k_r=160, groups=40,
        k_sec=(4, 8, 28, 3), inc_sec=(16, 32, 32, 128),
        num_classes=num_classes, test_time_pool=test_time_pool)
    if pretrained:
        PATH = model_files['dpn131']
        print("PATH:",PATH)
        if os.path.exists(PATH):
             model. load_state_dict(torch.load(PATH)) 
                
        elif has_mxnet and os.path.exists('./pretrained/'):
            convert_from_mxnet(model, checkpoint_prefix='./pretrained/dpn131')
        else:
            assert False, "Unable to load a pretrained model"
    return model


def dpn107(num_classes=1000, pretrained=False, test_time_pool=True):
    model = DPN(
        num_init_features=128, k_r=200, groups=50,
        k_sec=(4, 8, 20, 3), inc_sec=(20, 64, 64, 128),
        num_classes=num_classes, test_time_pool=test_time_pool)
    if pretrained:
        PATH = model_files['dpn107-extra']
        if os.path.exists(PATH):
             model. load_state_dict(torch.load(PATH)) 
        elif has_mxnet and os.path.exists('./pretrained/'):
            convert_from_mxnet(model, checkpoint_prefix='./pretrained/dpn107-extra')
        else:
            assert False, "Unable to load a pretrained model"
    return model


class CatBnAct(nn.Module):
    def __init__(self, in_chs, activation_fn=nn.ReLU(inplace=True)):
        super(CatBnAct, self).__init__()
        self.bn = nn.BatchNorm2d(in_chs, eps=0.001)
        self.act = activation_fn

    def forward(self, x):
        x = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        return self.act(self.bn(x))


class BnActConv2d(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride,
                 padding=0, groups=1, activation_fn=nn.ReLU(inplace=True)):
        super(BnActConv2d, self).__init__()
        self.bn = nn.BatchNorm2d(in_chs, eps=0.001)
        self.act = activation_fn
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, groups=groups, bias=False)

    def forward(self, x):
        return self.conv(self.act(self.bn(x)))


class InputBlock(nn.Module):
    def __init__(self, num_init_features, kernel_size=7,
                 padding=3, activation_fn=nn.ReLU(inplace=True)):
        super(InputBlock, self).__init__()
        self.conv = nn.Conv2d(
            3, num_init_features, kernel_size=kernel_size, stride=2, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(num_init_features, eps=0.001)
        self.act = activation_fn
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class DualPathBlock(nn.Module):
    def __init__(
            self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, groups, block_type='normal', b=False):
        super(DualPathBlock, self).__init__()
        self.num_1x1_c = num_1x1_c
        self.inc = inc
        self.b = b
        if block_type is 'proj':
            self.key_stride = 1
            self.has_proj = True
        elif block_type is 'down':
            self.key_stride = 2
            self.has_proj = True
        else:
            assert block_type is 'normal'
            self.key_stride = 1
            self.has_proj = False

        if self.has_proj:
            # Using different member names here to allow easier parameter key matching for conversion
            if self.key_stride == 2:
                self.c1x1_w_s2 = BnActConv2d(
                    in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=2)
            else:
                self.c1x1_w_s1 = BnActConv2d(
                    in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=1)
        self.c1x1_a = BnActConv2d(in_chs=in_chs, out_chs=num_1x1_a, kernel_size=1, stride=1)
        self.c3x3_b = BnActConv2d(
            in_chs=num_1x1_a, out_chs=num_3x3_b, kernel_size=3,
            stride=self.key_stride, padding=1, groups=groups)
        if b:
            self.c1x1_c = CatBnAct(in_chs=num_3x3_b)
            self.c1x1_c1 = nn.Conv2d(num_3x3_b, num_1x1_c, kernel_size=1, bias=False)
            self.c1x1_c2 = nn.Conv2d(num_3x3_b, inc, kernel_size=1, bias=False)
        else:
            self.c1x1_c = BnActConv2d(in_chs=num_3x3_b, out_chs=num_1x1_c + inc, kernel_size=1, stride=1)

    def forward(self, x):
        x_in = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        if self.has_proj:
            if self.key_stride == 2:
                x_s = self.c1x1_w_s2(x_in)
            else:
                x_s = self.c1x1_w_s1(x_in)
            x_s1 = x_s[:, :self.num_1x1_c, :, :]
            x_s2 = x_s[:, self.num_1x1_c:, :, :]
        else:
            x_s1 = x[0]
            x_s2 = x[1]
        x_in = self.c1x1_a(x_in)
        x_in = self.c3x3_b(x_in)
        if self.b:
            x_in = self.c1x1_c(x_in)
            out1 = self.c1x1_c1(x_in)
            out2 = self.c1x1_c2(x_in)
        else:
            x_in = self.c1x1_c(x_in)
            out1 = x_in[:, :self.num_1x1_c, :, :]
            out2 = x_in[:, self.num_1x1_c:, :, :]
        resid = x_s1 + out1
        dense = torch.cat([x_s2, out2], dim=1)
        return resid, dense


class DPN(nn.Module):
    def __init__(self, small=False, num_init_features=64, k_r=96, groups=32,
                 b=False, k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128),
                 num_classes=1000, test_time_pool=False, dilation=True):
        super(DPN, self).__init__()
        self.test_time_pool = test_time_pool
        self.b = b
        bw_factor = 1 if small else 4
        print('bw_factor:', bw_factor)
        
        blocks = OrderedDict()

        # conv1
        if small:
            blocks['conv1_1'] = InputBlock(num_init_features, kernel_size=3, padding=1)
        else:
            blocks['conv1_1'] = InputBlock(num_init_features, kernel_size=7, padding=3)

        # conv2
        bw = 64 * bw_factor
        print('bw:', bw)
        
        inc = inc_sec[0]
        r = (k_r * bw) // (64 * bw_factor)
        print('r:', r)
        
        print('conv2 in_chs:', num_init_features)
        blocks['conv2_1'] = DualPathBlock(num_init_features, r, r, bw, inc, groups, 'proj', b)
        in_chs = bw + 3 * inc
        print('conv2_ in_chs:', in_chs)
        print('conv2_ r:', r)
        print('conv2_ bw:', bw)
        for i in range(2, k_sec[0] + 1):
            blocks['conv2_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc

        # conv3
        bw = 128 * bw_factor
        inc = inc_sec[1]
        r = (k_r * bw) // (64 * bw_factor)
        print('conv3 in_chs:', in_chs)
        blocks['conv3_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        print('conv3_ in_chs:', in_chs)
        print('conv3_ r:', r)
        print('conv3_ bw:', bw)
        for i in range(2, k_sec[1] + 1):
            blocks['conv3_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc

        # conv4
        bw = 256 * bw_factor
        inc = inc_sec[2]
        r = (k_r * bw) // (64 * bw_factor)
        print('conv4 in_chs:', in_chs)
        blocks['conv4_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        print('conv4_ in_chs:', in_chs)
        print('conv4_ r:', r)
        print('conv4_ bw:', bw)
        for i in range(2, k_sec[2] + 1):
            blocks['conv4_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc

        # conv5
        bw = 512 * bw_factor
        inc = inc_sec[3]
        r = (k_r * bw) // (64 * bw_factor)
        print('conv5 in_chs:', in_chs)
        blocks['conv5_1'] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        print('conv5_ in_chs:', in_chs)
        print('conv5_ r:', r)
        print('conv5_ bw:', bw)
        for i in range(2, k_sec[3] + 1):
            blocks['conv5_' + str(i)] = DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b)
            in_chs += inc
        blocks['conv5_bn_ac'] = CatBnAct(in_chs)
        print('conv last in_chs:', in_chs)

        self.features = nn.Sequential(blocks)
        
        
        # Using 1x1 conv for the FC layer to allow the extra pooling scheme
        self.classifier = nn.Conv2d(in_chs, num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        if not self.training and self.test_time_pool:
            x = F.avg_pool2d(x, kernel_size=7, stride=1)
            out = self.classifier(x)
            # The extra test time pool should be pooling an img_size//32 - 6 size patch
            out = adaptive_avgmax_pool2d(out, pool_type='avgmax')
        else:
            x = adaptive_avgmax_pool2d(x, pool_type='avg')
            out = self.classifier(x)
        return out.view(out.size(0), -1)

    
"""
 using pre-trained model 'dpnsal131_dilation_multipath'
bw_factor: 4
bw: 256
r: 160
conv2 in_chs: 128
conv2_ in_chs: 304
conv2_ r: 160
conv2_ bw: 256
conv3 in_chs: 352
conv3_ in_chs: 608
conv3_ r: 320
conv3_ bw: 512
conv4 in_chs: 832
conv4_ in_chs: 1120
conv4_ r: 640
conv4_ bw: 1024
conv5 in_chs: 1984
conv5_ in_chs: 2432
conv5_ r: 1280
conv5_ bw: 2048
conv last in_chs: 2688

OrderedDict([('conv1_1', InputBlock(
  (conv): Conv2d(3, 128, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
  (act): ReLU(inplace=True)
  (pool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
)), 




('conv2_1', DualPathBlock(
  (c1x1_w_s1): BnActConv2d(
    (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(128, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(128, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(160, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv2_2', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(304, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(304, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(160, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv2_3', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(320, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(160, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv2_4', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(336, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(336, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(160, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), 


('conv3_1', DualPathBlock(
  (c1x1_w_s2): BnActConv2d(
    (bn): BatchNorm2d(352, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(352, 576, kernel_size=(1, 1), stride=(2, 2), bias=False)
  )
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(352, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(352, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(320, 544, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)),('conv3_2', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(608, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(608, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(320, 544, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv3_3', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(320, 544, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv3_4', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(672, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(320, 544, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)),('conv3_5', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(704, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(704, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(320, 544, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv3_6', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(736, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(736, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(320, 544, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv3_7', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(768, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(768, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(320, 544, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv3_8', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(800, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(800, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(320, 544, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), 



('conv4_1', DualPathBlock(
  (c1x1_w_s2): BnActConv2d(
    (bn): BatchNorm2d(832, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(832, 1088, kernel_size=(1, 1), stride=(2, 2), bias=False)
  )
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(832, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(832, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv4_2', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(1120, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1120, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)),('conv4_3', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(1152, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1152, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv4_4', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(1184, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1184, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv4_5', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(1216, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1216, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv4_6', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(1248, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1248, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv4_7', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(1280, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv4_8', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(1312, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1312, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv4_9', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(1344, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1344, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv4_10', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(1376, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1376, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv4_11', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(1408, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1408, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv4_12', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(1440, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1440, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv4_13', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(1472, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1472, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv4_14', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(1504, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1504, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv4_15', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(1536, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1536, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv4_16', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(1568, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1568, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv4_17', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(1600, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1600, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv4_18', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(1632, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1632, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv4_19', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(1664, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1664, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv4_20', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(1696, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1696, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv4_21', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(1728, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1728, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv4_22', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(1760, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1760, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv4_23', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(1792, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1792, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv4_24', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(1824, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1824, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv4_25', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(1856, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1856, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv4_26', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(1888, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1888, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv4_27', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(1920, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1920, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv4_28', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(1952, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1952, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(640, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), 






('conv5_1', DualPathBlock(
  (c1x1_w_s2): BnActConv2d(
    (bn): BatchNorm2d(1984, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1984, 2304, kernel_size=(1, 1), stride=(2, 2), bias=False)
  )
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(1984, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1984, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(1280, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(1280, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1280, 2176, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv5_2', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(2432, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(2432, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(1280, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(1280, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1280, 2176, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv5_3', DualPathBlock(
  (c1x1_a): BnActConv2d(
    (bn): BatchNorm2d(2560, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (c3x3_b): BnActConv2d(
    (bn): BatchNorm2d(1280, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
  )
  (c1x1_c): BnActConv2d(
    (bn): BatchNorm2d(1280, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
    (act): ReLU(inplace=True)
    (conv): Conv2d(1280, 2176, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)), ('conv5_bn_ac', CatBnAct(
  (bn): BatchNorm2d(2688, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
  (act): ReLU(inplace=True)
))])
PATH: /home/wuchenjunlin/new_program/dpn_wight/dpn131-71dfe43e0.pth
model_lr 1e-05
"""