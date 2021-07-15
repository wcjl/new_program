import argparse
import shutil
import datetime
import time
import random
import os
import torch.nn.functional as F
import torchvision
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import math
import torchvision.transforms as transforms

import numpy as np

from .base_model import BaseModel
from .readout_nets import upsampling_modules
from .pytorch_dpn_pretrained.model_factory import create_model

best_prec1 = 0


__all__ = ['clique_dilation_multipath']



def clique_dilation_multipath(pretrained=True):
    main_net1 = clique1(pretrained=pretrained)
    main_net2 = clique2(pretrained=pretrained)
    readout_net = readout_net = nn.Conv2d(608, 1, kernel_size=1, stride=1, padding=0)  #5376->4384
    model = BaseModel(main_net1,main_net2, readout_net)
    return model


class build_cliquenet1(nn.Module):
    def __init__(self, input_channels, list_channels, list_layer_num, if_att):
        super(build_cliquenet1, self).__init__()
        self.fir_trans = nn.Conv2d(3, input_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.fir_bn = nn.BatchNorm2d(input_channels)
        self.fir_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block_num = len(list_channels)

        self.if_att = if_att
        self.list_block = nn.ModuleList()
        self.list_trans = nn.ModuleList()
        self.list_gb = nn.ModuleList()
        self.list_gb_channel = []
        self.list_compress = nn.ModuleList()
        input_size_init1_1 = 120  #この論文でimagenet　datasetサイズは224×224、rec_long_side=640, rec_short_side=480
        input_size_init1_2 = 160
        
        for i in range(self.block_num):

            if i == 0:
                self.list_block.append(clique_block(input_channels=input_channels, channels_per_layer=list_channels[0], layer_num=list_layer_num[0], loop_num=1, keep_prob=0.8))
                self.list_gb_channel.append(input_channels + list_channels[0] * list_layer_num[0])
            else :
                self.list_block.append(clique_block(input_channels=list_channels[i-1] * list_layer_num[i-1], channels_per_layer=list_channels[i], layer_num=list_layer_num[i], loop_num=1, keep_prob=0.8))
                self.list_gb_channel.append(list_channels[i-1] * list_layer_num[i-1] + list_channels[i] * list_layer_num[i])

            if i < self.block_num - 1:
                self.list_trans.append(transition1(self.if_att, current_size=(input_size_init1_1,input_size_init1_2), input_channels=list_channels[i] * list_layer_num[i], keep_prob=0.8))

            #self.list_gb.append(global_pool(input_size=(input_size_init1_1,input_size_init1_2), input_channels=self.list_gb_channel[i] // 2))
            
            self.list_compress.append(compress(input_channels=self.list_gb_channel[i], keep_prob=0.8))
            print("self.list_gb_channel:",self.list_gb_channel)
            input_size_init1_1 = input_size_init1_1 // 2
            input_size_init1_2 = input_size_init1_2 // 2

        self.fc = nn.Linear(in_features=sum(self.list_gb_channel) // 2, out_features=1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):

        output = self.fir_trans(x)
        output = self.fir_bn(output)
        output = F.relu(output)
        output = self.fir_pool(output)

        feature_I_list = []

        # use stage II + stage II mode
        for i in range(self.block_num):
            block_feature_I, block_feature_II = self.list_block[i](output)
            #print("\nblock_feature_I:",np.shape(block_feature_I))
            #print("block_feature_II:",np.shape(block_feature_II))
            #print(np.shape(block_feature_I))
            #print(np.shape(block_feature_II))
            #print(i,self.list_gb[i])
            #print(self.list_trans)
            block_feature_I = self.list_compress[i](block_feature_I)
            #feature_I_list.append(self.list_gb[i](block_feature_I))
            feature_I_list.append(block_feature_I)
            #print("\nfeature_I_list:",feature_I_list[i].size(),'\n')
            if i < self.block_num - 1:
                output = self.list_trans[i](block_feature_II)

        #print(self.list_trans)
        
        #final_feature = feature_I_list[0]
        #for block_id in range(1, len(feature_I_list)):
            #final_feature=torch.cat((final_feature, feature_I_list[block_id]), 1)
            
            #print("net1_final_feature:",final_feature.size())
        #final_feature = final_feature.view(final_feature.size()[0], final_feature.size()[1])
        #print("final_feature:",final_feature.size()) #final_feature: torch.Size([160, 2192])

        #output = self.fc(final_feature) #torch.Size([160, 1000])
        
        final_feature = feature_I_list[len(feature_I_list)-1]
        block_id = len(feature_I_list)-1
        while(block_id >= 0):
            _,C1,H1,W1 = final_feature.size()
            _,C2,H2,W2 =  feature_I_list[block_id].size()
            conv1 = nn.Conv2d(C1,C2,kernel_size=1)
            conv1 = conv1.cuda()
            final_feature = conv1(final_feature)
            #print(final_feature.size())
            final_feature = F.upsample(final_feature, size = (H2,W2), mode='bilinear') 
            #print(final_feature.size())
            final_feature =torch.cat((final_feature, feature_I_list[block_id]), 1)
            block_id -= 1
            
      
        output = final_feature
        #print("main_net1 output:",output.size())
        
        return output

def clique1(pretrained):
    model = build_cliquenet1(input_channels=64, list_channels=[40, 80, 160, 160], list_layer_num=[6, 6, 6, 6], if_att=True)
    model = torch.nn.DataParallel(model).cuda()
    if pretrained:
        PATH = "/home/wuchenjunlin/CliqueNet/S3_att_best.pth"
        pt = torch.load('/home/wuchenjunlin/CliqueNet/S3_att_best.pth', map_location=torch.device('cpu'))
        if os.path.exists(PATH):
             model.load_state_dict(pt['state_dict'],strict=False)
        else:
            assert False, "Unable to load a pretrained model"
    return model


class build_cliquenet2(nn.Module):
    def __init__(self, input_channels, list_channels, list_layer_num, if_att):
        super(build_cliquenet2, self).__init__()
        self.fir_trans = nn.Conv2d(3, input_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.fir_bn = nn.BatchNorm2d(input_channels)
        self.fir_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block_num = len(list_channels)

        self.if_att = if_att
        self.list_block = nn.ModuleList()
        self.list_trans = nn.ModuleList()
        self.list_gb = nn.ModuleList()
        self.list_gb_channel = []
        self.list_compress = nn.ModuleList()
        input_size_init2_1 = 60
        input_size_init2_2 = 80
        

        for i in range(self.block_num):

            if i == 0:
                self.list_block.append(clique_block(input_channels=input_channels, channels_per_layer=list_channels[0], layer_num=list_layer_num[0], loop_num=1, keep_prob=0.8))
                self.list_gb_channel.append(input_channels + list_channels[0] * list_layer_num[0])
            else :
                self.list_block.append(clique_block(input_channels=list_channels[i-1] * list_layer_num[i-1], channels_per_layer=list_channels[i], layer_num=list_layer_num[i], loop_num=1, keep_prob=0.8))
                self.list_gb_channel.append(list_channels[i-1] * list_layer_num[i-1] + list_channels[i] * list_layer_num[i])

            if i < self.block_num - 1:
                self.list_trans.append(transition2(self.if_att, current_size=(input_size_init2_1,input_size_init2_2), input_channels=list_channels[i] * list_layer_num[i], keep_prob=0.8))

            #self.list_gb.append(global_pool(input_size=(input_size_init2_1,input_size_init2_2), input_channels=self.list_gb_channel[i] // 2))
            self.list_compress.append(compress(input_channels=self.list_gb_channel[i], keep_prob=0.8))
            #print("self.list_gb_channel:",self.list_gb_channel)
            input_size_init2_1 = input_size_init2_1 // 2
            input_size_init2_2 = input_size_init2_2 // 2

        self.fc = nn.Linear(in_features=sum(self.list_gb_channel) // 2, out_features=1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):

        output = self.fir_trans(x)
        output = self.fir_bn(output)
        output = F.relu(output)
        output = self.fir_pool(output)

        feature_I_list = []

        # use stage II + stage II mode
        for i in range(self.block_num):
            block_feature_I, block_feature_II = self.list_block[i](output)
            #print("\nblock_feature_I:",np.shape(block_feature_I))
            #print("block_feature_II:",np.shape(block_feature_II))
            #print(np.shape(block_feature_I))
            #print(np.shape(block_feature_II))
            #print(i,self.list_gb[i])
            #print(i,self.list_trans[i])
            #print(self.list_trans)
            block_feature_I = self.list_compress[i](block_feature_I)
            #feature_I_list.append(self.list_gb[i](block_feature_I))
            feature_I_list.append(block_feature_I)
            if i < self.block_num - 1:
                output = self.list_trans[i](block_feature_II)
                

        #final_feature = feature_I_list[0]
        #for block_id in range(1, len(feature_I_list)):
            #final_feature=torch.cat((final_feature, feature_I_list[block_id]), 1)
            #print("net2_final_feature:",final_feature.size())
        #final_feature = final_feature.view(final_feature.size()[0], final_feature.size()[1])
        #output = self.fc(final_feature)
        
        
        final_feature = feature_I_list[len(feature_I_list)-1]
        block_id = len(feature_I_list)-2
        while(block_id >= 0):
            _,C1,H1,W1 = final_feature.size()
            _,C2,H2,W2 =  feature_I_list[block_id].size()
            print(final_feature.size())
            print(feature_I_list[block_id].size())
            
            conv1 = nn.Conv2d(C1,C2,kernel_size=1)
            conv1 = conv1.cuda()
            final_feature = conv1(final_feature)
            #print(final_feature.size())
            final_feature = F.upsample(final_feature, size = (H2,W2), mode='bilinear') 
            #print(final_feature.size())
            final_feature =torch.cat((final_feature, feature_I_list[block_id]), 1)
            print(final_feature.size())
            block_id -= 1
            
        output = final_feature
        #print("main_net2 output:",output.size())
        
        return output

def clique2(pretrained):
    model = build_cliquenet2(input_channels=64, list_channels=[40, 80, 160, 160], list_layer_num=[6, 6, 6, 6], if_att=True)
    model = torch.nn.DataParallel(model).cuda()
    if pretrained:
        PATH = "/home/wuchenjunlin/CliqueNet/S3_att_best.pth"
        pt = torch.load('/home/wuchenjunlin/CliqueNet/S3_att_best.pth', map_location=torch.device('cpu'))
        if os.path.exists(PATH):
             model.load_state_dict(pt['state_dict'],strict=False)
        else:
            assert False, "Unable to load a pretrained model"
    return model

#fine 图片torch.Size([1, 3, 480, 640])

class attention(nn.Module):
    def __init__(self, input_channels, map_size):
        super(attention, self).__init__()
        #print("attetion input_channels:",input_channels)
        
        #self.map_size = map_size
        self.pool = nn.AvgPool2d(kernel_size = map_size)
        #print("map_size:",map_size)
        #self.fc1 = nn.Linear(in_features = input_channels,out_features = input_channels // 2)   #240 ,120 
        self.fc1 = nn.Linear(in_features = input_channels,out_features = input_channels // 2)   #240 ,120 
        self.fc2 = nn.Linear(in_features = input_channels // 2, out_features = input_channels)  #120, 120
        

        #x: torch.Size([160, 240, 8, 8])
        #output0: torch.Size([160, 240, 1, 1])
        #output1: torch.Size([160, 120])
        #output2: torch.Size([160, 240])
        #output3: torch.Size([160, 240, 8, 8])
        
    def forward(self, x):
        
        
        #print("\nattention input:",x.size())
        #x: torch.Size([1, 240, 120, 160])  batchsize,channels,H,W

        output = self.pool(x)
        
        #print("attention pool:",output.size())
        #output: torch.Size([1, 240, 15, 20])
        
        output = output.view(output.size()[0], output.size()[1])       
        output = self.fc1(output)
        output = F.relu(output)
        #print("attention fc1:",output.size())
        
    
        output = self.fc2(output)
        output = torch.sigmoid(output)
        #print("attention fc2:",output.size())
        
        output = output.contiguous().view(output.size()[0],output.size()[1],1,1)
        output = torch.mul(x, output)
        #print("attention output3:",output.size())
        return output


class transition1(nn.Module):
    def __init__(self, if_att, current_size, input_channels, keep_prob):
        super(transition1, self).__init__()
        self.input_channels = input_channels
        self.keep_prob = keep_prob
        self.bn = nn.BatchNorm2d(self.input_channels)
        self.conv = nn.Conv2d(self.input_channels, self.input_channels, kernel_size = 1, bias = False)
        # self.dropout = nn.Dropout2d(1 - self.keep_prob)
        self.pool = nn.AvgPool2d(kernel_size = 2)
        self.if_att = if_att
        if self.if_att == True:
            self.attention = attention(input_channels = self.input_channels, map_size = current_size)

    def forward(self, x):
        output = self.bn(x)
        output = F.relu(output)
        output = self.conv(output)
        if self.if_att==True:
            output = self.attention(output)
        # output = self.dropout(output)
        output = self.pool(output)
        return output
    
    
class transition2(nn.Module):
    def __init__(self, if_att, current_size, input_channels, keep_prob):
        super(transition2, self).__init__()
        self.input_channels = input_channels
        self.keep_prob = keep_prob
        self.bn = nn.BatchNorm2d(self.input_channels)
        self.conv = nn.Conv2d(self.input_channels, self.input_channels, kernel_size = 1, bias = False)
        # self.dropout = nn.Dropout2d(1 - self.keep_prob)
        self.pool = nn.AvgPool2d(kernel_size = 2)
        self.if_att = if_att
        if self.if_att == True:
            self.attention = attention(input_channels = self.input_channels, map_size = current_size)

    def forward(self, x):
        output = self.bn(x)
        output = F.relu(output)
        output = self.conv(output)
        if self.if_att==True:
            output = self.attention(output)
        # output = self.dropout(output)
        output = self.pool(output)
        return output    

class global_pool(nn.Module):
    def __init__(self, input_size, input_channels):
        super(global_pool, self).__init__()
        self.input_size = input_size
        #print("global pool input_size:",input_size)
        self.input_channels = input_channels
        
        self.bn = nn.BatchNorm2d(self.input_channels)
        #a = int(self.input_size[0]/15),int(self.input_size[1]/20)
        self.pool = nn.AvgPool2d(kernel_size = self.input_size)
        
    def forward(self, x):
        #torch.Size([160, 304, 8, 8])
        #print("global input:",x.size())
        #output = self.bn(x)
        #output = F.relu(output)
        #output = self.pool(output)
        output = x
        #print("global pool output:",output.size())
        return output

class compress(nn.Module):
    def __init__(self, input_channels, keep_prob):
        super(compress, self).__init__()
        self.keep_prob = keep_prob
        self.bn = nn.BatchNorm2d(input_channels)
        self.conv = nn.Conv2d(input_channels, input_channels//2, kernel_size = 1, padding = 0, bias = False)


    def forward(self, x):
        output = self.bn(x)
        output = F.relu(output)
        output = self.conv(output)
        # output = F.dropout2d(output, 1 - self.keep_prob)
        return output

class clique_block(nn.Module):
    def __init__(self, input_channels, channels_per_layer, layer_num, loop_num, keep_prob):
        super(clique_block, self).__init__()
        self.input_channels = input_channels
        self.channels_per_layer = channels_per_layer
        self.layer_num = layer_num
        self.loop_num = loop_num
        self.keep_prob = keep_prob

        # conv 1 x 1
        self.conv_param = nn.ModuleList([nn.Conv2d(self.channels_per_layer, self.channels_per_layer, kernel_size = 1, padding = 0, bias = False)
                                   for i in range((self.layer_num + 1) ** 2)])

        for i in range(1, self.layer_num + 1):
            self.conv_param[i] = nn.Conv2d(self.input_channels, self.channels_per_layer, kernel_size = 1, padding = 0, bias = False)
        for i in range(1, self.layer_num + 1):
            self.conv_param[i * (self.layer_num + 2)] = None
        for i in range(0, self.layer_num + 1):
            self.conv_param[i * (self.layer_num + 1)] = None

        self.forward_bn = nn.ModuleList([nn.BatchNorm2d(self.input_channels + i * self.channels_per_layer) for i in range(self.layer_num)])
        self.forward_bn_b = nn.ModuleList([nn.BatchNorm2d(self.channels_per_layer) for i in range(self.layer_num)])
        self.loop_bn = nn.ModuleList([nn.BatchNorm2d(self.channels_per_layer * (self.layer_num - 1)) for i in range(self.layer_num)])
        self.loop_bn_b = nn.ModuleList([nn.BatchNorm2d(self.channels_per_layer) for i in range(self.layer_num)])

        # conv 3 x 3
        self.conv_param_bottle = nn.ModuleList([nn.Conv2d(self.channels_per_layer, self.channels_per_layer, kernel_size = 3, padding = 1, bias = False)
                                   for i in range(self.layer_num)])


    def forward(self, x):
        # key: 1, 2, 3, 4, 5, update every loop
        self.blob_dict={}
        # save every loops results
        self.blob_dict_list=[]

        # first forward
        for layer_id in range(1, self.layer_num + 1):
            bottom_blob = x
            # bottom_param = self.param_dict['0_' + str(layer_id)]

            bottom_param = self.conv_param[layer_id].weight
            for layer_id_id in range(1, layer_id):
                # pdb.set_trace()
                bottom_blob = torch.cat((bottom_blob, self.blob_dict[str(layer_id_id)]), 1)
                # bottom_param = torch.cat((bottom_param, self.param_dict[str(layer_id_id) + '_' + str(layer_id)]), 1)
                bottom_param = torch.cat((bottom_param, self.conv_param[layer_id_id * (self.layer_num + 1) + layer_id].weight), 1)
            next_layer = self.forward_bn[layer_id - 1](bottom_blob)
            next_layer = F.relu(next_layer)
            # conv 1 x 1
            next_layer = F.conv2d(next_layer, bottom_param, stride = 1, padding = 0)
            # conv 3 x 3
            next_layer = self.forward_bn_b[layer_id - 1](next_layer)
            next_layer = F.relu(next_layer)
            next_layer = F.conv2d(next_layer, self.conv_param_bottle[layer_id - 1].weight, stride = 1, padding = 1)
            # next_layer = F.dropout2d(next_layer, 1 - self.keep_prob)
            self.blob_dict[str(layer_id)] = next_layer
        self.blob_dict_list.append(self.blob_dict)

        # loop
        for loop_id in range(self.loop_num):
            for layer_id in range(1, self.layer_num + 1):

                layer_list = [l_id for l_id in range(1, self.layer_num + 1)]
                layer_list.remove(layer_id)

                bottom_blobs = self.blob_dict[str(layer_list[0])]
                # bottom_param = self.param_dict[layer_list[0] + '_' + str(layer_id)]
                bottom_param = self.conv_param[layer_list[0] * (self.layer_num + 1) + layer_id].weight
                for bottom_id in range(len(layer_list) - 1):
                    bottom_blobs = torch.cat((bottom_blobs, self.blob_dict[str(layer_list[bottom_id + 1])]), 1)
                    # bottom_param = torch.cat((bottom_param, self.param_dict[layer_list[bottom_id+1]+'_'+str(layer_id)]), 1)
                    bottom_param = torch.cat((bottom_param, self.conv_param[layer_list[bottom_id + 1] * (self.layer_num + 1) + layer_id].weight), 1)
                bottom_blobs = self.loop_bn[layer_id - 1](bottom_blobs)
                bottom_blobs = F.relu(bottom_blobs)
                # conv 1 x 1
                mid_blobs = F.conv2d(bottom_blobs, bottom_param, stride = 1, padding = 0)
                # conv 3 x 3
                top_blob = self.loop_bn_b[layer_id - 1](mid_blobs)
                top_blob = F.relu(top_blob)
                top_blob = F.conv2d(top_blob, self.conv_param_bottle[layer_id - 1].weight, stride = 1, padding = 1)
                self.blob_dict[str(layer_id)] = top_blob
            self.blob_dict_list.append(self.blob_dict)

        assert len(self.blob_dict_list) == 1 + self.loop_num

        # output
        block_feature_I = self.blob_dict_list[0]['1']
        for layer_id in range(2, self.layer_num + 1):
            block_feature_I = torch.cat((block_feature_I, self.blob_dict_list[0][str(layer_id)]), 1)
        block_feature_I = torch.cat((x, block_feature_I), 1)

        block_feature_II = self.blob_dict_list[self.loop_num]['1']
        for layer_id in range(2, self.layer_num + 1):
            block_feature_II = torch.cat((block_feature_II, self.blob_dict_list[self.loop_num][str(layer_id)]), 1)
        return block_feature_I, block_feature_II
