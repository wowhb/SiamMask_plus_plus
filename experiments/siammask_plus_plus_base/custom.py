# --------------------------------------------------------
# SiamMask++ 
# Modified from Hyunbin Choi for SiamMask++
# Written by Qiang Wang (Licensed under The MIT License)
# --------------------------------------------------------
from models.siammask_plus_plus_bi import SiamMask_pp
from models.features import MultiStageFeature
from models.rpn import RPN, DepthCorr, DepthwiseXCorr
from models.mask import Mask, DepthCorr, DepthwiseXCorr

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.load_helper import load_pretrain
from resnet import resnet50
#from neck import AdjustLayer, AdjustAllLayer

class ResDownS(nn.Module):
    def __init__(self, inplane, outplane):
        super(ResDownS, self).__init__()
        self.downsample = nn.Sequential(
                nn.Conv2d(inplane, outplane, kernel_size=1, bias=False),
                nn.BatchNorm2d(outplane))

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:
            l = 4
            r = -4
            x = x[:, :, l:r, l:r]
        return x

class ResDown(MultiStageFeature):
    def __init__(self, pretrain=False):
        super(ResDown, self).__init__()
        self.features = resnet50(layer3=True, layer4=True)
        if pretrain:
            load_pretrain(self.features, 'resnet.model')

        self.downsample1 = ResDownS(512, 256) 
        self.downsample2 = ResDownS(1024, 256) 
        self.downsample3 = ResDownS(2048, 256)         
#        self.neck = AdjustAllLayer(**{'in_channels': [512, 1024, 2048], 'out_channels': [256, 256, 256]})
        self.layers = [self.downsample1,self.downsample2,self.downsample3,self.features.layer2, self.features.layer3, self.features.layer4]#, self.neck]
        self.train_nums = [1, 6]
        self.change_point = [0, 0.5]
        self.unfix(0.0)

    def param_groups(self, start_lr, feature_mult=1):
        lr = start_lr * feature_mult

        def _params(module, mult=1):
            params = list(filter(lambda x:x.requires_grad, module.parameters()))
            if len(params):
                return [{'params': params, 'lr': lr * mult}]
            else:
                return []
        groups = []
        groups += _params(self.downsample1)
        groups += _params(self.downsample2)
        groups += _params(self.downsample3)
        groups += _params(self.features, 0.1)
#        groups += _params(self.neck)
        return groups

    def forward(self, x):
        output = self.features(x)
        p2 = self.downsample1(output[0])
        p3 = self.downsample2(output[1])
        p4 = self.downsample3(output[2])
        return p2, p3, p4

class DepthwiseRPN(RPN):
    def __init__(self, anchor_num=5, in_channels=256, out_channels=256):
        super(DepthwiseRPN, self).__init__()
        self.rpn_pred_cls = DepthwiseXCorr(in_channels, out_channels, 2 * anchor_num)
        self.rpn_pred_loc = DepthwiseXCorr(in_channels, out_channels, 4 * anchor_num)

    def forward(self, z_f, x_f):
        rpn_pred_cls = self.rpn_pred_cls(z_f, x_f)
        rpn_pred_loc = self.rpn_pred_loc(z_f, x_f)
        return rpn_pred_cls, rpn_pred_loc

class MultiRPN(RPN):
    def __init__(self, anchor_num, in_channels, weighted=False):
        super(MultiRPN, self).__init__()
        self.weighted = weighted      
        for i in range(len(in_channels)):
            self.add_module('rpn'+str(i+2),
                    DepthwiseRPN(anchor_num, in_channels[i], in_channels[i]))
        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.loc_weight = nn.Parameter(torch.ones(len(in_channels)))

    def forward(self, z_fs, x_fs):
        rpn_pred_cls = []
        rpn_pred_loc = []
        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
            rpn = getattr(self, 'rpn'+str(idx))
            c, l = rpn(z_f, x_f)
            rpn_pred_cls.append(c)
            rpn_pred_loc.append(l)

        if self.weighted:
            cls_weight = F.softmax(self.cls_weight, 0)
            loc_weight = F.softmax(self.loc_weight, 0)

        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        if self.weighted:
            return weighted_avg(rpn_pred_cls, cls_weight), weighted_avg(rpn_pred_loc, loc_weight)
        else:
            return avg(rpn_pred_cls), avg(loc)

    def param_groups(self, start_lr, feature_mult=1, key=None):
        if key is None:
            params = filter(lambda x:x.requires_grad, self.parameters())
        else:
            params = [v for k, v in self.named_parameters() if (key in k) and v.requires_grad]
        params = [{'params': params, 'lr': start_lr * feature_mult}]
        return params   

class DepthwiseMask(Mask):
    def __init__(self, in_channels=256, out_channels=256):
        super(DepthwiseMask, self).__init__()
        self.rpn_pred_mask = DepthwiseXCorr(in_channels, out_channels, 63*63)

    def forward(self, z_f, x_f):
        rpn_pred_mask = self.rpn_pred_mask(z_f, x_f)
        return rpn_pred_mask        
        
class MultiMask(Mask):        
    def __init__(self, in_channels, weighted=False):
        super(MultiMask, self).__init__()
        self.weighted = weighted
        for i in range(len(in_channels)):
            self.add_module('mask'+str(i+2),
                    DepthwiseMask(in_channels[i], in_channels[i]))
        if self.weighted:
            self.mask_weight = nn.Parameter(torch.ones(len(in_channels)))

    def forward(self, z_fs, x_fs):
        rpn_pred_mask = []

        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
            mask = getattr(self, 'mask'+str(idx))
            m = mask(z_f, x_f)
            rpn_pred_mask.append(m)
           

        if self.weighted:
            mask_weight = F.softmax(self.mask_weight, 0)


        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        if self.weighted:
            return weighted_avg(rpn_pred_mask, mask_weight)
        else:
            return avg(rpn_pred_mask)

    def param_groups(self, start_lr, feature_mult=1, key=None):
        if key is None:
            params = filter(lambda x:x.requires_grad, self.parameters())
        else:
            params = [v for k, v in self.named_parameters() if (key in k) and v.requires_grad]
        params = [{'params': params, 'lr': start_lr * feature_mult}]
        return params   


class Custom(SiamMask_pp):
    def __init__(self, pretrain=False, **kwargs):
        super(Custom, self).__init__(**kwargs)
        self.features = ResDown(pretrain=pretrain)        
#        self.neck = AdjustAllLayer(**{'in_channels': [512, 1024, 2048], 'out_channels': [256, 256, 256]})
        self.rpn_model = MultiRPN(**{'anchor_num': 5, 'in_channels': [256, 256, 256], 'weighted': True})
        self.mask_model = MultiMask(**{'in_channels': [256, 256, 256], 'weighted': True})

    def template(self, z):
        zf = self.features(z)
#        zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        x = self.features(x)
#        x = self.neck(x)
        rpn_pred_cls, rpn_pred_loc = self.rpn_model(self.zf, x)
        return {
                'cls': rpn_pred_cls,
                'loc': rpn_pred_loc,               
               }

    def track_mask(self, search):
        search = self.features(search)
#        search = self.neck(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn_model(self.zf, search)
        pred_mask = self.mask_model(self.zf, search)
        return rpn_pred_cls, rpn_pred_loc, pred_mask
        
     
        
