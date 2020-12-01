import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.GroupNorm(32,in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.GroupNorm(32,int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.GroupNorm(32,int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.GroupNorm(32,in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(32,planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.GroupNorm(32,planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(32,planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(256, 256))

        self.add_module('b2_' + str(level), ConvBlock(256, 256))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(256, 256))

        self.add_module('b3_' + str(level), ConvBlock(256, 256))
        self.add_module('b4_' + str(level), ConvBlock(256, 256))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        # Lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2

        low3 = self._modules['b3_' + str(level)](low3)

        up3 = F.upsample(low3, scale_factor=2, mode='nearest')

        return up1 + up3

    def forward(self, x):
        return self._forward(self.depth, x)







class FAN(nn.Module):

    def __init__(self,numberofclusters,step=1):
        super(FAN, self).__init__()
        self.step=step
        self.useShapeModel=False
        self.num_modules = 1
        # Base part
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.GroupNorm(32,64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)
        # Stacking part


        self.add_module('m0', HourGlass(1, 4, 256))


        #detector
        self.add_module('top_m_0' , ConvBlock(256, 256))
        self.add_module('conv_last0' ,
                        nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
        self.add_module('bn_end0' , nn.GroupNorm(32,256))
        self.add_module('l0' , nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0))

        #descriptor
        self.add_module('top_m_1' , ConvBlock(256, 256))
        self.add_module('conv_last1' ,
                        nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
        self.add_module('bn_end1', nn.GroupNorm(32, 256))
        self.add_module('l1', nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))



    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), True)
        x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)

        previous = x
        hg= self._modules['m0'](previous)

        ll = hg
        ll = self._modules['top_m_0' ](ll)
        ll = F.relu(self._modules['bn_end0' ] (self._modules['conv_last0'](ll)), True)


        tmp_out = self._modules['l0'](ll)
        detectorOutput=torch.nn.functional.sigmoid(tmp_out)

        ll1 = hg
        ll1 = self._modules['top_m_1'](ll1)
        ll1 = F.relu(self._modules['bn_end1'](self._modules['conv_last1'](ll1)), True)
        decriptorOutput=self._modules['l1'](ll1)

        if(self.step==1):
            return detectorOutput,decriptorOutput
        else:
            return decriptorOutput




def weight_init(m):

    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight)
        if m.bias is not None:
            torch.nn.init.constant(m.bias, 0)
    # elif isinstance(m, torch.nn.BatchNorm2d):
    #     torch.nn.init.constant(m.weight, 1)
    #     torch.nn.init.constant(m.bias, 0)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.normal(m.weight, std=1e-3)
        if m.bias:
            torch.nn.init.constant(m.bias, 0)





