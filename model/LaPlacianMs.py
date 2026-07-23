import torch
import torch.nn as nn
from torch.nn import functional as F
# 相对导入是以包的形式进行导入的，所以不能直接运行。
# --------------解决方案----------------------
# 第一步：cd ~/HiFi_IFDL-main
# 第二步：python3 -m models.LaPlacianMs (没有.py的后缀，用模块方式运行)

from .GaussianSmoothing import GaussianSmoothing

# junfu：多尺度拉普拉斯算子
class LaPlacianMs(nn.Module):
    def __init__(self,in_c,gauss_ker_size=3,scale=[2],drop_rate=0.2): # in_c输入通道数,gauss_ker_size高斯卷积核大小,scale多尺度列表(例如 [2, 4]),drop_rate dropout 比例
        super(LaPlacianMs, self).__init__()
        self.scale = scale
        self.gauss_ker_size = gauss_ker_size
        ## apply gaussian smoothing to input feature maps with 3 planes
        ## with kernel size K and sigma s
        ## 为每个尺度生成一个 Gaussian blur 模块
        ## s 是 sigma（或者核大小变化），表示模糊程度不同
        self.smoothing = nn.ModuleDict()
        for s in self.scale:
            self.smoothing['scale-'+str(s)] = GaussianSmoothing(in_c, self.gauss_ker_size, s)
        
        # 传统神经元：1×1卷积负责通道融合、特征压缩、非线性增强
        self.conv_1x1 = nn.Sequential(nn.Conv2d(in_c*len(scale), in_c,
                                                kernel_size=1, stride=1,
                                                bias=False,groups=1),
                                                nn.BatchNorm2d(in_c),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(p=drop_rate)
        )
        
        # junfu：SNP型神经元
        
        
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    # junfu：下采样
    def down(self,x,s):
        return F.interpolate(x,scale_factor=s,
                             mode='bilinear',
                             align_corners=False)
    # junfu：上采样
    def up (self,x, size):
        return F.interpolate(x,size=size,mode='bilinear',align_corners=False)

    # junfu：前向传播
    def forward(self, x):
        # 核心步骤
        for i, s in enumerate(self.scale):
            sm = self.smoothing['scale-'+str(s)](x) # 1.Gaussian blur
            sm = self.down(sm,1/s)                  # 2.下采样
            sm = self.up(sm,(x.shape[2],x.shape[3]))# 3.插值恢复原大小
            # 然后计算插值(类似Laplacian Pyramid)
            if i == 0:
                diff = x - sm
            else:
                diff = torch.cat((diff, x - sm), dim=1)
        # 4.得到所有尺度的高频响应后通过 1×1 卷积融合：
        return self.conv_1x1(diff)