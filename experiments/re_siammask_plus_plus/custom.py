from models.siammask_sharp import SiamMask
from models.features import MultiStageFeature
from models.rpn import RPN, DepthCorr, DepthwiseXCorr
from models.mask import Mask, DepthCorr, DepthwiseXCorr
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.load_helper import load_pretrain
#from resnet_original import resnet50
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

        self.downsample1 = ResDownS(512, 256) #256
        self.downsample2 = ResDownS(1024, 256) #256
        self.downsample3 = ResDownS(2048, 256) #256
        
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

        return groups

    def forward(self, x):
        output = self.features(x)
        p2 = self.downsample1(output[2])
        p3 = self.downsample2(output[3])
        p4 = self.downsample3(output[4])
        return p2, p3, p4
        
    def forward_all(self, x):
        output = self.features(x)
        return output  


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
    def __init__(self, in_channels, weighted=False, oSz=63):
        super(MultiMask, self).__init__()
        self.weighted = weighted
        self.oSz = oSz
        
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


class Refine(nn.Module):
    def __init__(self):
        super(Refine, self).__init__()
        self.v0 = nn.Sequential(nn.Conv2d(64, 16, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(16, 4, 3, padding=1),nn.ReLU())

        self.v1 = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(64, 16, 3, padding=1), nn.ReLU())

        self.v2 = nn.Sequential(nn.Conv2d(512, 128, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(128, 32, 3, padding=1), nn.ReLU())

        self.h2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())

        self.h1 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())

        self.h0 = nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(4, 4, 3, padding=1), nn.ReLU())

        self.deconv = nn.ConvTranspose2d(63*63, 32, 15, 15)

        self.post0 = nn.Conv2d(32, 16, 3, padding=1)
        self.post1 = nn.Conv2d(16, 4, 3, padding=1)
        self.post2 = nn.Conv2d(4, 1, 3, padding=1)
        
        for modules in [self.v0, self.v1, self.v2, self.h2, self.h1, self.h0, self.deconv, self.post0, self.post1, self.post2,]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, f, corr_feature, pos=None, test=False):
        if test:
            p0 = torch.nn.functional.pad(f[0], [16, 16, 16, 16])[:, :, 4*pos[0]:4*pos[0]+61, 4*pos[1]:4*pos[1]+61]
            p1 = torch.nn.functional.pad(f[1], [8, 8, 8, 8])[:, :, 2 * pos[0]:2 * pos[0] + 31, 2 * pos[1]:2 * pos[1] + 31]
            p2 = torch.nn.functional.pad(f[2], [4, 4, 4, 4])[:, :, pos[0]:pos[0] + 15, pos[1]:pos[1] + 15]
        else:
            p0 = F.unfold(f[0], (61, 61), padding=0, stride=4).permute(0, 2, 1).contiguous().view(-1, 64, 61, 61)
            if not (pos is None): p0 = torch.index_select(p0, 0, pos)
            p1 = F.unfold(f[1], (31, 31), padding=0, stride=2).permute(0, 2, 1).contiguous().view(-1, 256, 31, 31)
            if not (pos is None): p1 = torch.index_select(p1, 0, pos)
            p2 = F.unfold(f[2], (15, 15), padding=0, stride=1).permute(0, 2, 1).contiguous().view(-1, 512, 15, 15)
            if not (pos is None): p2 = torch.index_select(p2, 0, pos)

        if not(pos is None):
            p3 = corr_feature[:, :, pos[0], pos[1]].view(-1, 63*63, 1, 1)
        else:
            p3 = corr_feature.permute(0, 2, 3, 1).contiguous().view(-1, 63*63, 1, 1)

        out = self.deconv(p3)
        out = self.post0(F.upsample(self.h2(out) + self.v2(p2), size=(31, 31)))
        out = self.post1(F.upsample(self.h1(out) + self.v1(p1), size=(61, 61)))
        out = self.post2(F.upsample(self.h0(out) + self.v0(p0), size=(127, 127)))
        out = out.view(-1, 127*127)
        return out

    def param_groups(self, start_lr, feature_mult=1):
        params = filter(lambda x:x.requires_grad, self.parameters())
        params = [{'params': params, 'lr': start_lr * feature_mult}]
        return params
        

class Custom(SiamMask):
    def __init__(self, pretrain=False, **kwargs):
        super(Custom, self).__init__(**kwargs)
        self.features = ResDown(pretrain=pretrain)        
#        self.neck = AdjustAllLayer(**{'in_channels': [512, 1024, 2048], 'out_channels': [256, 256, 256]})
        self.rpn_model = MultiRPN(**{'anchor_num': 5, 'in_channels': [256, 256, 256], 'weighted': True})
        self.mask_model = MultiMask(**{'in_channels': [256, 256, 256], 'weighted': True})
        self.refine_model = Refine()


    def refine(self, f, pos=None):
        return self.refine_model(f,pos)


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
        
        self.feature = self.features.forward_all(search)           
        search = self.features(search)
#        search = self.neck(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn_model(self.zf, search)        
        self.corr_feature = self.mask_model(self.zf, search)
        pred_mask = self.mask_model(self.zf, search)

        return rpn_pred_cls, rpn_pred_loc, pred_mask
        
    def track_refine(self, pos):
        pred_mask = self.refine_model(self.feature, self.corr_feature, pos=pos, test=True)
        return pred_mask
        
        
        
