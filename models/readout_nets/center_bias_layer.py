import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterBiasLayer(nn.Module):
    def __init__(self, size=(20,20)):
        super(CenterBiasLayer, self).__init__()
        self.weight = nn.Parameter(torch.ones(1,1,size[0],size[1]).cuda())

    def forward(self, x):
        weight = nn.functional.interpolate(self.weight, size=(x.size(2), x.size(3)), mode='bilinear') # adjust height and width # adjust height and width
        weight = weight.expand_as(x) #adjust batch size
        out = x * ((weight / weight.data[0].sum()) * weight.size(2) * weight.size(3))
        #out = x * weight / weight.data[0].max()
        return F.relu(out, inplace=True)
        #out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        #if out.data[0].min() < 0:
        #    out = out - out.min()
        #return out
        #return F.relu(out, inplace=True)
