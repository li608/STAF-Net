import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch
from torch.nn import init
from .resnest import ResNeSt, Bottleneck
from  .cafe import CBAM,SEAttention



def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x



class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1)) #在内部压缩了通道数
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())
        A = A + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


class STFEM(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1,kt=9, residual=True):
        super(STFEM, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels,kernel_size=kt, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class AFEM(nn.Module):
    def __init__(self,in_channels,num_point):
        super(AFEM,self).__init__()
        self.conv = nn.Conv2d(num_point,num_point,1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.cbam = CBAM(in_planes=in_channels, ratio=16, kernel_size=7)
        self.se = SEAttention(channel=in_channels)
        conv_init(self.conv)
        bn_init(self.bn,1)
    def forward(self, x):
        res = x #N,C,T,V
        x = x.permute(0,3,2,1).contiguous() #N,V,T,C
        x = self.conv(x)
        x = x.permute(0,3,2,1).contiguous()
        x = self.bn(x)
        x = self.cbam(x)
        x = self.se(x)
        x = [res,x]
        x = torch.cat(x,1)
        return  x #N,2C,T,V


def DSEM(**kwargs):
    model = ResNeSt(Bottleneck, [2, 2, 2, 2],
                   radix=2, groups=2, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True,dilation=2, avd_first=False, **kwargs)

    return model


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,drop_out=0.0):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.l1 = STFEM(in_channels, 64, A,kt=1, residual=False)
        self.l2 = STFEM(64, 64, A,kt=3)
        self.l3 = STFEM(64, 64, A)
        self.l4 = STFEM(64, 64, A)
        self.afem = AFEM(64, num_point)
        self.backbone = DSEM()
        self.drop_out = nn.Dropout(drop_out)
        self.fc = nn.Linear(1088, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / 1088))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size() # N: batch_size, C: channels, T: time_steps, V: joints, M: people

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V) # N*2,3,300,25

        x = self.l1(x) # N*2,64,300,25
        x = self.l2(x) # N*2,64,300,25
        c_new = x.size(1)
        y = x.view(N, M, c_new, -1) # N,2,64,300*25
        y = y.mean(3).mean(1) # N,64
        x = self.l3(x) # N*2,64,300,25
        x = self.l4(x) # N*2,64,300,25
        x = self.afem(x) # N*2,128,300,25
        x = self.backbone(x) # N*2,1024,1,1
        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x= torch.cat((y,x),1)
        x = self.drop_out(x)
        return self.fc(x)

