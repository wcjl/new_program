import torch
import torch.nn as nn
import torch.nn.functional as F


class ArgumentError(Exception):
    pass


class BaseModel(nn.Module):
    def __init__(self, main_net1,main_net2, readout_net, single_fine_path=False, single_coarse_path=False):
        super(BaseModel, self).__init__()
        if single_fine_path and single_coarse_path:
            raise ArgumentError("Don't set both single_fine_path and single_coarse_path to True.")
        self.main_net1 = main_net1
        self.main_net2 = main_net2
        self.readout_net = readout_net
        self.single_fine_path = single_fine_path
        self.single_coarse_path = single_coarse_path

    def forward(self, x):
        fine = x  
        #print("fine_size:",fine.size())
        h_fine = self.main_net1(fine)
        #print("h_fine:",h_fine.size())
        
        if not self.single_fine_path:
            coarse = nn.functional.interpolate(x, (x.size(2)//2, x.size(3)//2), mode='bilinear')
            #print("corarse.size:",coarse.size())
            
            h_coarse = nn.functional.interpolate(self.main_net2(coarse), (h_fine.size(2), h_fine.size(3)), mode='bilinear')
            #print("h_coarse:",h_coarse.size())

            if self.single_coarse_path:
                #out = F.relu(self.readout_net(h_coarse), inplace=True)
                out = F.leaky_relu(self.readout_net(h_coarse), negative_slope=0.01, inplace=True)
                if out.data[0].min() < 0:
                    out = out - out.min()
                return out
            
            h_fine = torch.cat([h_fine, h_coarse], dim=1)  #h_fine: torch.Size([1, 4384, 1, 1])

            #print("h_fine_all:",h_fine.size())

        #out = F.relu(self.readout_net(h_fine), inplace=True)
        out = F.leaky_relu(self.readout_net(h_fine), negative_slope=0.01, inplace=True)
        #print("out",out.size())
        if out.data[0].min() < 0:
            out = out - out.min()
        return out





class BaseModel2(nn.Module):
    def __init__(self, main_net, readout_net, single_fine_path=False, single_coarse_path=False):
        super(BaseModel2, self).__init__()
        if single_fine_path and single_coarse_path:
            raise ArgumentError("Don't set both single_fine_path and single_coarse_path to True.")
        self.main_net = main_net
        self.readout_net = readout_net
        self.single_fine_path = single_fine_path
        self.single_coarse_path = single_coarse_path

    def forward(self, x):
        fine = x  
        #print("fine_size:",fine.size())
        #fine 图片torch.Size([1, 3, 480, 640])
        
        h_fine = self.main_net(fine)
        #print("h_fine:",h_fine.size())
        
        #h_fine:torch.Size([1, 2208, 29, 39])

        if not self.single_fine_path:
            coarse = nn.functional.interpolate(x, (x.size(2)//2, x.size(3)//2), mode='bilinear')
            #print("corarse.size:",coarse.size())
            #corarse.size: torch.Size([1, 3, 240, 320])
            
            h_coarse = nn.functional.interpolate(self.main_net(coarse), (h_fine.size(2), h_fine.size(3)), mode='bilinear')
            #print("h_coarse:",h_coarse.size())
            if self.single_coarse_path:
                #out = F.relu(self.readout_net(h_coarse), inplace=True)
                out = F.leaky_relu(self.readout_net(h_coarse), negative_slope=0.01, inplace=True)
                if out.data[0].min() < 0:
                    out = out - out.min()
                return out
            h_fine = torch.cat([h_fine, h_coarse], dim=1)  

            #print("h_fine_all:",h_fine.size())

        #out = F.relu(self.readout_net(h_fine), inplace=True)
        
        out = F.leaky_relu(self.readout_net(h_fine), negative_slope=0.01, inplace=True)
        #print("out",out.size())
        if out.data[0].min() < 0:
            out = out - out.min()
        return out

        """ to measure time for coarse path
        fine = x
        if not self.single_coarse_path:
            h_fine = self.main_net(fine)

        if not self.single_fine_path:
            coarse = F.upsample(x, (x.size(2)//2, x.size(3)//2), mode='bilinear')
            #h_coarse = F.upsample(self.main_net(coarse), (15, 20), mode='bilinear') # saliconnet
            #h_coarse = F.upsample(self.main_net(coarse), (29, 39), mode='bilinear') # densesal
            h_coarse = F.upsample(self.main_net(coarse), (30, 40), mode='bilinear') # dpnsal
            if self.single_coarse_path:
                #out = F.relu(self.readout_net(h_coarse), inplace=True)
                out = F.leaky_relu(self.readout_net(h_coarse), negative_slope=0.01, inplace=True)
                if out.data[0].min() < 0:
                    out = out - out.min()
                return out
            h_fine = torch.cat([h_fine, h_coarse], dim=1)

        #out = F.relu(self.readout_net(h_fine), inplace=True)
        out = F.leaky_relu(self.readout_net(h_fine), negative_slope=0.01, inplace=True)
        if out.data[0].min() < 0:
            out = out - out.min()
        return out
        """

class BaseModel_tri(nn.Module):
    def __init__(self, main_net1,main_net2, main_net3,readout_net, single_fine_path=False, single_coarse_path=False):
        super(BaseModel_tri, self).__init__()
        if single_fine_path and single_coarse_path:
            raise ArgumentError("Don't set both single_fine_path and single_coarse_path to True.")
        self.main_net1 = main_net1
        self.main_net2 = main_net2
        self.main_net3 = main_net3
        self.readout_net = readout_net
        self.single_fine_path = single_fine_path
        self.single_coarse_path = single_coarse_path

    def forward(self, x):
        fine = x  
        #print("fine_size:",fine.size())
        #fine 图片torch.Size([1, 3, 480, 640])
        
        h_fine = self.main_net(fine)
        #print("h_fine:",h_fine.size())
        #h_fine:torch.Size([1, 2208, 29, 39])

        if not self.single_fine_path:
            coarse = nn.functional.interpolate(x, (x.size(2)//2, x.size(3)//2), mode='bilinear')
            #print("corarse.size:",coarse.size())
            #corarse.size: torch.Size([1, 3, 240, 320])
            
            h_coarse = nn.functional.interpolate(self.main_net2(coarse), (h_fine.size(2), h_fine.size(3)), mode='bilinear')
            #print("h_coarse:",h_coarse.size())
            if self.single_coarse_path:
                #out = F.relu(self.readout_net(h_coarse), inplace=True)
                out = F.leaky_relu(self.readout_net(h_coarse), negative_slope=0.01, inplace=True)
                if out.data[0].min() < 0:
                    out = out - out.min()
                return out
            h_fine = torch.cat([h_fine, h_coarse], dim=1)  
            m_coarse = nn.functional.interpolate(x, (x.size(2)//4, x.size(3)//4), mode='bilinear')
            #print("corarse.size:",coarse.size())
            #corarse.size: torch.Size([1, 3, 240, 320]) 
            h_m_coarse = nn.functional.interpolate(self.main_net3(m_coarse), (h_fine.size(2), h_fine.size(3)), mode='bilinear')

            #print("h_fine_all:",h_fine.size())
            h_fine = torch.cat([h_fine, h_m_coarse], dim=1) 

        #out = F.relu(self.readout_net(h_fine), inplace=True)
        
        out = F.leaky_relu(self.readout_net(h_fine), negative_slope=0.01, inplace=True)
        #print("out",out.size())
        if out.data[0].min() < 0:
            out = out - out.min()
        return out
