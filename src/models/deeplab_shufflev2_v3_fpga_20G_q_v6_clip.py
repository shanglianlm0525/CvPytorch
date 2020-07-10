import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from torch.autograd import Variable
from functools import reduce
from src.integrated_models.base import IntegratedModel
from ..reusable_modules.base import BaseModule
from ..quantization.modules import (
    QConv2d,
    QConcat,
    QShuffleOp,
    QAvgPool2d,
    QPixelShuffle,
    QEltwiseAdd,
    QHalfSplit,
    QUpsample
)
# from lib.mvutils.nn.torch_extended import EltwiseMul,EltwiseAdd_root
from lib.mvutils.utils.config import CommonConfiguration
from src.quantization.clipped_layers import QClippedLayer, QClippedReLU, set_clip_layer_info
from ..off_network.loss import dice_loss
#(enforced_min=-8, enforced_max=7.9375)


class _ShuffleBottleneck(BaseModule):
    def __init__(self, in_channels, out_channels, mid_channels=None, mid_stride=1, mid_groups=6, out_groups=2):
        super(_ShuffleBottleneck, self).__init__()
        mid_channels = mid_channels or in_channels
        self.conv1 = QConv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=2)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.ReLU1 = QClippedReLU(clip_at=8)
        self.shfl_op = QShuffleOp(2)
        self.conv2 = QConv2d(mid_channels, mid_channels, kernel_size=3, stride=mid_stride, padding=1,
                             groups=mid_channels//4)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.clip2 = QClippedLayer(enforced_min=-8, enforced_max=7.9375)
        self.conv3 = QConv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=2)

        self.bn3 = nn.BatchNorm2d(out_channels)

        self.clip3 = QClippedLayer(enforced_min=-8, enforced_max=7.9375)


    def forward(self, x):
        x = self.ReLU1(self.bn1(self.conv1(x)))
        x = self.shfl_op(x)
        x = self.clip2(self.bn2(self.conv2(x)))
        # x = self.bn2(self.conv2(x))
        x = self.clip3(self.bn3(self.conv3(x)))
        return x

class _ShuffleResUnitC(BaseModule):
    """C is short for Concat"""
    def __init__(self, in_channels, out_channels, mid_channels=None, mid_groups=6):
        super(_ShuffleResUnitC, self).__init__()
        mid_channels = mid_channels or in_channels
        self.bottleneck = _ShuffleBottleneck(in_channels, out_channels - in_channels, mid_channels, mid_stride=2)
        self.pooling = QAvgPool2d(kernel_size=2, stride=2)
        self.concat = QConcat()
        self.out_ReLU = QClippedReLU(clip_at=8)

    def forward(self, x):
        return self.out_ReLU(self.concat(self.pooling(x), self.bottleneck(x)))        






class _ShuffleResUnitE(nn.Module):
    def __init__(self, inp, oup, stride):
        super(_ShuffleResUnitE, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        
        oup_inc = oup//2
        self.first_half = QHalfSplit(dim=1, first_half=True)
        self.second_split = QHalfSplit(dim=1, first_half=False)                        

        self.banch2 = nn.Sequential(
            # pw
            QConv2d(oup_inc, oup_inc, 1, 1, 0, bias=True),
            nn.BatchNorm2d(oup_inc),
            QClippedReLU(clip_at=8),
            # dw
            QConv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc//4, bias=True),
            nn.BatchNorm2d(oup_inc),
            QClippedLayer(enforced_min=-8, enforced_max=7.9375),
            # pw-linear
            QConv2d(oup_inc, oup_inc, 1, 1, 0, bias=True),
            nn.BatchNorm2d(oup_inc),
            QClippedReLU(clip_at=8),
        )                
        self.concat_res = QConcat()
        self.shfl_op = QShuffleOp(2)
    # @staticmethod
    def forward(self, x):
        x1 = self.first_half(x)
        x2 = self.second_split(x)
        out = self.concat_res(x1, self.banch2(x2))


        return self.shfl_op(out)


class ShuffleNet(BaseModule):
    # TODO: implement __new__ to generate either normal ShuffleNet or QuantizedShuffleNet in the future
    def __init__(self):
        super(ShuffleNet, self).__init__()
        # self.conv1 = QConv2d(3, 24, kernel_size=3, stride=2, padding=1)
        # self.conv1_bn = nn.BatchNorm2d(24)
        # self.conv1_ReLU = QClippedReLU(clip_at=8)

        self.conv1 = QConv2d(3, 24, kernel_size=3, stride=2, padding=1)
        self.conv1_bn = nn.BatchNorm2d(24)
        self.conv1_ReLU = QClippedReLU(clip_at=8)

        self.res1a = _ShuffleResUnitC(24, 72)
        self.res1b = _ShuffleResUnitE(72, 72, stride=1)
        self.res1c = _ShuffleResUnitE(72, 72, stride=1)
        self.res1d = _ShuffleResUnitE(72, 72, stride=1)
        self.res2a = _ShuffleResUnitC(72, 120)
        self.res2b = _ShuffleResUnitE(120, 120, stride=1)
        self.res2c = _ShuffleResUnitE(120, 120, stride=1)
        self.res2d = _ShuffleResUnitE(120, 120, stride=1)
        # self.res2e = InvertedResidual(128, 128, stride=1)
        self.res3a = _ShuffleResUnitC(120, 240)
        self.res3b = _ShuffleResUnitE(240, 240, stride=1)
        self.res3c = _ShuffleResUnitE(240, 240, stride=1)
        self.res3d = _ShuffleResUnitE(240, 240, stride=1)
        self.res3e = _ShuffleResUnitE(240, 240, stride=1)
        self.res3f = _ShuffleResUnitE(240, 240, stride=1)

        self.res4a = _ShuffleResUnitC(240, 480)
        self.res4b = _ShuffleResUnitE(480, 480, stride=1)
        self.res4c = _ShuffleResUnitE(480, 480, stride=1)
        self.res4d = _ShuffleResUnitE(480, 480, stride=1)
        self.res4e = _ShuffleResUnitE(480, 480, stride=1)
        # self.res4f = InvertedResidual(480, 480, stride=1)
        # self.res3g = InvertedResidual(256, 256, stride=1)
        # self.res3h = InvertedResidual(256, 256, stride=1)
        # self.res2_pooling = QAvgPool2d(kernel_size=2, stride=2)
        # self.res4_up = nn.Upsample(scale_factor=2, mode='nearest')
        # self.concat = Concat(576)
        self.initialize_params()

    def initialize_params(self):
        for m in self.modules():
            if isinstance(m, QConv2d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0., 0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.running_var, 1)
                nn.init.constant_(m.running_mean, 0)

    def forward(self, x):
        # print(x.type())
        x = self.conv1_ReLU(self.conv1_bn(self.conv1(x)))
        x = self.res1a(x)
        x = self.res1b(x)
        x = self.res1c(x)
        res1 = x = self.res1d(x)
        x = self.res2a(x)
        x = self.res2b(x)
        x = self.res2c(x)
        res2 = x = self.res2d(x)
        x = self.res3a(x)
        x = self.res3b(x)
        x = self.res3c(x)
        x = self.res3d(x)
        x = self.res3e(x)
        res3 = x = self.res3f(x)
        x = self.res4a(x)
        x = self.res4b(x)
        x = self.res4c(x)
        x = self.res4d(x)
        # x = self.res4e(x)
        res4 = x = self.res4e(x)
        return  res4, res3, res2


                

class ASPP_module_3(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module_3, self).__init__()
        kernel_size = 3
        padding = rate
        self.atrous_convolution_DW = QConv2d(inplanes, inplanes, groups=inplanes//4,kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=True)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.clip1 = QClippedLayer(enforced_min=-8, enforced_max=7.9375)       
        self.atrous_convolution_PW = QConv2d(inplanes, planes, kernel_size=1,
                                            stride=1, padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = QClippedReLU(clip_at=8)


        self.__init_weight()

    def forward(self, x):
        x = self.atrous_convolution_DW(x)
        x = self.clip1(self.bn1(x))
        x = self.atrous_convolution_PW(x)
        x = self.bn2(x)
        return self.relu(x)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, QConv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP_module_1(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module_1, self).__init__()

        kernel_size = 1
        padding = 0

        self.atrous_convolution = QConv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=True)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = QClippedReLU(clip_at=8)

        self.__init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, QConv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DeepLabv3_plus(IntegratedModel):
    def __init__(self, cfg,dataset_cfg, _print=True):
        nInputChannels=3
        n_classes=8
        os=16
        pretrained=False
        _print=True
        if _print:
            print("Constructing DeepLabv3+ model...")
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(os))
            print("Number of Input Channels: {}".format(nInputChannels))
        super(DeepLabv3_plus, self).__init__(cfg,dataset_cfg)
        
        # Atrous Conv
        self.shuffle_features = ShuffleNet()

        # ASPP
        rates_res4 = [1, 2, 4, 4]

        rates_res3 = [1, 4, 8, 8]

        rates_res2 = [1, 4, 8, 8]


        # self.fea_num= 1024

        self.fea_num_res4= 480
        self.aspp1_res4 = ASPP_module_1(self.fea_num_res4, 96, rate=rates_res4[0])
        self.aspp2_res4 = ASPP_module_3(self.fea_num_res4, 96, rate=rates_res4[1])
        self.aspp3_res4 = ASPP_module_3(self.fea_num_res4, 96, rate=rates_res4[2])
        self.aspp4_res4 = ASPP_module_3(self.fea_num_res4, 96, rate=rates_res4[3])
        self.concat_r4 = QConcat()
        self.r4_conv  = QConv2d(384, 384, 1, bias=True)
        self.r4_bn    = nn.BatchNorm2d(384)
        self.r4_relu  = QClippedReLU(clip_at=8)
        self.upsample_r4=QPixelShuffle(upscale_factor=4)

        self.fea_num_res3= 240
        self.aspp1_res3 = ASPP_module_1(self.fea_num_res3, 36, rate=rates_res3[0])
        self.aspp2_res3 = ASPP_module_3(self.fea_num_res3, 36, rate=rates_res3[1])
        self.aspp3_res3 = ASPP_module_3(self.fea_num_res3, 36, rate=rates_res3[2])
        self.aspp4_res3 = ASPP_module_3(self.fea_num_res3, 36, rate=rates_res3[3])
        # self.aspp5_res3 = ASPP_module_3(self.fea_num_res3, 48, rate=rates_res3[4])
        self.concat_r3 = QConcat()
        self.r3_conv  = QConv2d(144, 144, 1, bias=True)
        self.r3_bn    = nn.BatchNorm2d(144)
        self.r3_relu  = QClippedReLU(clip_at=8)
        self.upsample_r3=QPixelShuffle(upscale_factor=2)

        self.fea_num_res2= 120
        self.aspp1_res2 = ASPP_module_1(self.fea_num_res2, 24, rate=rates_res2[0])
        self.aspp2_res2 = ASPP_module_3(self.fea_num_res2, 24, rate=rates_res2[1])
        self.aspp3_res2 = ASPP_module_3(self.fea_num_res2, 24, rate=rates_res2[2])
        self.aspp4_res2 = ASPP_module_3(self.fea_num_res2, 24, rate=rates_res2[3])
        # self.aspp5_res2 = ASPP_module_3(self.fea_num_res2, 12, rate=rates_res2[4])
        self.concat_r2 = QConcat()
        self.r2_conv  = QConv2d(96, 96, 1, bias=True)
        self.r2_bn    = nn.BatchNorm2d(96)
        self.r2_relu  = QClippedReLU(clip_at=8)

        self.concat_x = QConcat()
        self.x_conv  = QConv2d(156, 144, 1, bias=True)
        self.x_bn    = nn.BatchNorm2d(144)
        self.x_relu  = QClippedReLU(clip_at=8)
        self.upsample_x=QPixelShuffle(upscale_factor=2)
        # self.upsample_x=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 

        # self.concat_fea = Concat()

        self.last_conv_DW= QConv2d(36, 36, kernel_size=3, groups=9, stride=1, padding=1, bias=True)
        self.last_conv_DW_bn=nn.BatchNorm2d(36)
        self.clip1 = QClippedLayer(enforced_min=-8, enforced_max=7.9375)        

        self.last_conv_PW= QConv2d(36, 36, kernel_size=1, stride=1, bias=True)
        self.last_conv_PW_bn=nn.BatchNorm2d(36)
        self.last_conv_PW_relu=QClippedReLU(clip_at=8)

        self.last_conv_linear= QConv2d(36, 36, kernel_size=1, stride=1)
        self.clip2 = QClippedLayer(enforced_min=-8, enforced_max=7.9375)
        self.last_conv_reg = QConv2d(36, 24, kernel_size=1, stride=1)
        self.clip3 = QClippedLayer(enforced_min=-8, enforced_max=7.9375)        
        self.last_conv_seg = QConv2d(36, 4, kernel_size=1, stride=1)
        self.last_conv_pt = QConv2d(36, 8, kernel_size=1, stride=1)


        # self.sig1=F.sigmoid()
        # self.sig2=F.sigmoid()
        self.pt_mul_x = QEltwiseAdd()
        self.pt_mul_y = QEltwiseAdd()
        self.pt_add = QEltwiseAdd()

        self.criterion = torch.nn.MSELoss(size_average=True).cuda()
        self.seg_bce_loss = torch.nn.BCEWithLogitsLoss()
        self.seg_bce_loss1 = torch.nn.BCELoss()
        # self.seg_bce_loss1 = torch.nn.BCEWithLogitsLoss()
        self.seg_dice_loss = dice_loss
        self.initialize_params()

    # def _set_bound_estimate_func(self, n_sigma):
    #     for m in self.modules():
    #         if isinstance(m, QClippedLayer) or isinstance(m, QClippedReLUWithInputStats):
    #             m.bound_estimate_func = lambda mu, std: (-(abs(mu) + n_sigma * abs(std)), abs(mu) + n_sigma * abs(std))

    # def set_clip_layer_info(self):
    #     print(self.cfg)
    #     self._set_bound_estimate_func(self.cfg.N_SIGMA_TO_BOUND)
    #     if self.cfg.ACTIVATE_BOUND is not None and bool(self.cfg.ACTIVATE_BOUND):
    #         self._activate_bound()
    #     if self.cfg.ENFORCED_MIN is not None and self.cfg.ENFORCED_MAX is not None:
    #         self._set_forced_min_max(self.cfg.ENFORCED_MIN, self.cfg.ENFORCED_MAX)
            
    # def _set_forced_min_max(self, enforced_min, enforced_max):
    #     for m in self.modules():
    #         if isinstance(m, QClippedReLU):
    #             m.enforced_max = enforced_max
    #         elif isinstance(m, QClippedLayer):
    #             m.enforced_min = enforced_min
    #             m.enforced_max = enforced_max


    # def _activate_bound(self):
    #     for m in self.modules():
    #         if isinstance(m, QClippedLayer) or isinstance(m, QClippedReLUWithInputStats) or isinstance(m, QClippedReLU):
    #             m.track_running_stats = False
    #             m.activate_bound()
   

    def _forward(self, dataset_names, im_paths, iter_n, epoch_n, im, labels=None, mode='eval', **kwargs):
        
        # set_clip_layer_info(self, self.cfg)                                         
        fea_4,fea_3,fea_2= self.shuffle_features(im)    #res4,res2,res1

        x1_f4 = self.aspp1_res4(fea_4)
        x2_f4 = self.aspp2_res4(fea_4)
        x3_f4 = self.aspp3_res4(fea_4)
        x4_f4 = self.aspp4_res4(fea_4)
        x_f4  = self.concat_r4(x1_f4, x2_f4, x3_f4,x4_f4, dim=1)
        x_f4  = self.r4_relu(self.r4_bn(self.r4_conv(x_f4)))
        print(x_f4.max())
        print(x_f4.min())
        x_f4  = self.upsample_r4(x_f4) #16x80x80
        #F.interpolate(x_f4, scale_factor=4, mode='bilinear', align_corners=True)


        x1_f3 = self.aspp1_res3(fea_3)
        x2_f3 = self.aspp2_res3(fea_3)
        x3_f3 = self.aspp3_res3(fea_3)
        x4_f3 = self.aspp4_res3(fea_3)
        # x5_f3 = self.aspp5_res3(fea_3)
        x_f3  = self.concat_r3(x1_f3, x2_f3, x3_f3,x4_f3, dim=1)
        x_f3  = self.r3_relu(self.r3_bn(self.r3_conv(x_f3)))
        # print(x_f3.max())
        # print(x_f3.min())        
        x_f3  = self.upsample_r3(x_f3)  #32x80x80
        #F.interpolate(x_f3, scale_factor=2, mode='bilinear', align_corners=True)        

        

        x1_f2 = self.aspp1_res2(fea_2)
        x2_f2 = self.aspp2_res2(fea_2)
        x3_f2 = self.aspp3_res2(fea_2)
        x4_f2 = self.aspp4_res2(fea_2)
        # x5_f2 = self.aspp5_res2(fea_2)
        x_f2 = self.concat_r2(x1_f2,x2_f2,x3_f2,x4_f2, dim=1)
        x_f2  = self.r2_relu(self.r2_bn(self.r2_conv(x_f2)))


        x = self.concat_x(x_f2, x_f3, x_f4, dim=1)
        x = self.x_relu(self.x_bn(self.x_conv(x)))
        x = self.upsample_x(x) 
        # x = self.conv_pixshuffle(x)        
        # x = self.bn_pix(x)
        # x = self.ReLU4(x)
        # fea= self.concat_fea(x, fea_1, dim=1)
        fea=self.clip1(self.last_conv_DW_bn(self.last_conv_DW(x)))
        print(fea.max())
        print(fea.min())        
        fea=self.last_conv_PW_relu(self.last_conv_PW_bn(self.last_conv_PW(fea)))
        fea = self.clip2(self.last_conv_linear(fea))
        # print(fea.max())
        # print(fea.min())        
        out_reg = self.clip3(self.last_conv_reg(fea))
        x_seg = self.last_conv_seg(fea)
        out_seg = F.sigmoid(x_seg)        
        x_point = self.last_conv_pt(fea)
        out_point = F.sigmoid(x_point)        
        # out_reg=self.reg_out(out_reg)

        out_reg_x=out_reg[:,[0,2,4,6],:]
        out_reg_y=out_reg[:,[1,3,5,7],:]

        # out_pt_x=self.pt_mul_x(out_reg_x,out_reg_x)
        # out_pt_y=self.pt_mul_y(out_reg_y,out_reg_y)
        # out_pt=self.pt_add(out_pt_x,out_pt_y)
        # out_pt_sigmoid = F.sigmoid(out_pt)        
        
        # x = F.upsample(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        if mode == 'eval':
            return torch.cat((out_reg,x_seg,x_point),1)
        else:

            labels_seg=labels[:,8:12,:,:]
            labels_pts=labels[:,12:16,:,:]
            labels_points=labels[:,12:16,:,:]
            labels_reg=labels[:,0:8,:,:]
            loss_seg=self.seg_bce_loss(x_seg, labels_seg).unsqueeze(0)
            # print(out_pt_sigmoid[:,0,:,:].max())
            # print(out_pt_sigmoid[:,0,:,:].min())
            # print(labels_pts[:,0,:,:].max())
            # print(labels_pts[:,0,:,:].min())
            loss_pts=self.criterion(out_pt, labels_pts).unsqueeze(0)
            loss_reg=self.criterion(out_reg, labels_reg).unsqueeze(0)
            loss_points=self.seg_bce_loss(x_point, labels_points).unsqueeze(0)
            # loss_pts_1 = self.seg_bce_loss(out_pt[:,0:1,:,:], labels_pts[:,0:1,:,:]).unsqueeze(0)
            # loss_pts_2 = self.seg_bce_loss(out_pt[:,1:2,:,:], labels_pts[:,1:2,:,:]).unsqueeze(0)
            # loss_pts_3 = self.seg_bce_loss(out_pt[:,2:3,:,:], labels_pts[:,2:3,:,:]).unsqueeze(0)
            # loss_pts_4 = self.seg_bce_loss(out_pt[:,3:4,:,:], labels_pts[:,3:4,:,:]).unsqueeze(0)
            # loss_pts   = loss_pts_1+loss_pts_2+loss_pts_3+loss_pts_4

            loss_dice_1 = self.seg_dice_loss(x_seg[:,0:1,:,:], out_seg[:,0:1,:,:], labels_seg[:,0:1,:,:]).unsqueeze(0)
            loss_dice_2 = self.seg_dice_loss(x_seg[:,1:2,:,:], out_seg[:,1:2,:,:], labels_seg[:,1:2,:,:]).unsqueeze(0)
            loss_dice_3 = self.seg_dice_loss(x_seg[:,2:3,:,:], out_seg[:,2:3,:,:], labels_seg[:,2:3,:,:]).unsqueeze(0)
            loss_dice_4 = self.seg_dice_loss(x_seg[:,3:4,:,:], out_seg[:,3:4,:,:], labels_seg[:,3:4,:,:]).unsqueeze(0)

            pts_dice_1 = self.seg_dice_loss(x_point[:,0:1,:,:], out_point[:,0:1,:,:], labels_points[:,0:1,:,:]).unsqueeze(0)
            pts_dice_2 = self.seg_dice_loss(x_point[:,1:2,:,:], out_point[:,1:2,:,:], labels_points[:,1:2,:,:]).unsqueeze(0)
            pts_dice_3 = self.seg_dice_loss(x_point[:,2:3,:,:], out_point[:,2:3,:,:], labels_points[:,2:3,:,:]).unsqueeze(0)
            pts_dice_4 = self.seg_dice_loss(x_point[:,3:4,:,:], out_point[:,3:4,:,:], labels_points[:,3:4,:,:]).unsqueeze(0)

            # loss=self.criterion(x, labels).unsqueeze(0)
            #50*loss_reg+0.5/20*loss_seg+0.5/800*(loss_dice_1+loss_dice_2+loss_dice_3+loss_dice_4) \
                                    
            losses = {'total_loss': 50*loss_reg+0.5/20*loss_seg+0.5/800*(loss_dice_1+loss_dice_2+loss_dice_3+loss_dice_4)\
                         +0.5/20*loss_points+0.5/800*(pts_dice_1+pts_dice_2+pts_dice_3+pts_dice_4)}
            losses_reg = {'total_loss': 10*loss_reg} 
            losses_seg = {'total_loss': 0.5/20*loss_seg} 
            losses_dice = {'total_loss': 0.5/800*(loss_dice_1+loss_dice_2+loss_dice_3+loss_dice_4+pts_dice_1+pts_dice_2+pts_dice_3+pts_dice_4) } 
            losses_pts_seg = {'total_loss': 10*loss_pts} 
            losses_points = {'total_loss': 0.5/2*loss_points} 
            
            # print('loss_reg=%f'%loss_reg)
            # print('loss_seg=%f'%loss_seg)
            # print('loss_dice=%f'%loss_dice)
            # print(loss_seg)
            # print('loss_reg')
            # print(loss_reg)            
            performances = [{'mse_loss': 50*loss_reg+0.5/20*loss_seg}]  #+0.05/200*loss_dice
            
            # losses_pts_dice=0
            if mode == 'train':
                return losses,losses_reg,losses_seg,losses_dice,losses_pts_seg,losses_points#,0.05/200*loss_dice

            if mode == 'val':
                return losses, performances,torch.cat((out_reg,out_seg,out_point),1)

        return torch.cat((out_reg,out_seg),1)

    def dummy_forward(self, im):
        return self._forward('_', '_', 0, 0, im)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, QConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = [model.shuffle_features]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = [model.aspp1, model.aspp2, model.aspp3, model.aspp4, model.conv1, model.conv2, model.last_conv]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k

def deeplabv3_shufflenetv1b_simple(**kwargs):
    model = DeepLabv3_plus( nInputChannels=3, n_classes=8, os=16, pretrained=False, _print=True)
    # model =Network(1000, 1)
    return model

if __name__ == "__main__":
    model = DeepLabv3_plus(nInputChannels=3, n_classes=21, os=16, pretrained=True, _print=True)
    model.eval()
    image = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        output = model.forward(image)
    print(output.size())



