import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class InceptionBlock(nn.Module):
    def __init__(self, inplanes, planes_1x1, planes_3x3_mid, planes_3x3, planes_5x5_mid, planes_5x5, planes_maxpool):
        super(InceptionBlock, self).__init__()

        # 1x1
        self.conv1x1 = BasicConv2d(inplanes, planes_1x1, kernel_size = 1, stride = 1, padding = 0)

        # 3x3
        self.conv3x3 = nn.Sequential(
            BasicConv2d(inplanes, planes_3x3, kernel_size = 3, stride = 1, padding = 1)
        )

        # 5x5
        self.conv5x5 = nn.Sequential(
            BasicConv2d(inplanes, planes_5x5, kernel_size = 5, stride = 1, padding = 2)
        )

        # maxpool
        self.maxpool1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(inplanes, planes_maxpool, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        conv1x1 = self.conv1x1(x)
        conv3x3 = self.conv3x3(x)
        conv5x5 = self.conv5x5(x)
        maxpool1 = self.maxpool1(x)
        outputs = [conv1x1, conv3x3, conv5x5, maxpool1]
        return torch.cat(outputs, 1)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, need_pool = True):
        super(BasicBlock, self).__init__()
        self.need_pool = need_pool
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size, 1, 1)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size, 1, 1)
        self.bn2 = nn.BatchNorm2d(outplanes)
        #self.relu2 = nn.ReLU()

        if self.need_pool:
            self.pool1 = nn.MaxPool2d(2)

        self.downsample = None
        if need_pool:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, 1),
                nn.BatchNorm2d(outplanes),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample != None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu1(out)

        if self.need_pool:
            out = self.pool1(out)
        return out

class CNN(nn.Module):
    def __init__(self, n_classes, pretrained = False):
        super(CNN, self).__init__()
        #self.conv1 = BasicBlock(3, 32, 3, False)
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 7, stride = 2, padding = 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.inception1 = InceptionBlock(inplanes = 32, planes_1x1 = 8, planes_3x3_mid = 8,
                                         planes_3x3 = 8, planes_5x5_mid = 8, planes_5x5 = 8, planes_maxpool = 8)
        self.inception11 = InceptionBlock(inplanes = 32, planes_1x1 = 8, planes_3x3_mid = 8,
                                         planes_3x3 = 8, planes_5x5_mid = 8, planes_5x5 = 8, planes_maxpool = 8)

        #self.conv11 = BasicBlock(32, 32, 3, need_pool=False)
        #self.conv111 = BasicBlock(32, 32, 3, need_pool=False)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        #self.conv2 = BasicBlock(32, 64, 3)
        #self.conv22 = BasicBlock(64, 64, 3, need_pool=False)
        self.inception2 = InceptionBlock(inplanes = 32, planes_1x1 = 16, planes_3x3_mid = 16,
                                         planes_3x3 = 16, planes_5x5_mid = 16, planes_5x5 = 16, planes_maxpool = 16)
        self.inception22 = InceptionBlock(inplanes = 64, planes_1x1 = 16, planes_3x3_mid = 16,
                                         planes_3x3 = 16, planes_5x5_mid = 16, planes_5x5 = 16, planes_maxpool = 16)

        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        #self.conv3 = BasicBlock(64, 128, 3)
        #self.conv33 = BasicBlock(128, 128, 3, need_pool=False)
        self.inception3 = InceptionBlock(inplanes = 64, planes_1x1 = 32, planes_3x3_mid = 32,
                                         planes_3x3 = 32, planes_5x5_mid = 32, planes_5x5 = 32, planes_maxpool = 32)
        self.inception33 = InceptionBlock(inplanes = 128, planes_1x1 = 32, planes_3x3_mid = 32,
                                         planes_3x3 = 32, planes_5x5_mid = 32, planes_5x5 = 32, planes_maxpool = 32)

        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        #self.conv4 = BasicBlock(128, 256, 3)
        #self.conv44 = BasicBlock(256, 256, 3, need_pool=False)
        self.inception4 = InceptionBlock(inplanes = 128, planes_1x1 = 64, planes_3x3_mid = 64,
                                         planes_3x3 = 64, planes_5x5_mid = 64, planes_5x5 = 64, planes_maxpool = 64)
        self.inception44 = InceptionBlock(inplanes = 256, planes_1x1 = 64, planes_3x3_mid = 64,
                                         planes_3x3 = 64, planes_5x5_mid = 64, planes_5x5 = 64, planes_maxpool = 64)


        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(256, 512, 3, 1, 1),  # 输出 (512, 14, 14)
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),  # relu层
        #     nn.MaxPool2d(2),  # 输出 (512, 7, 7)
        # )
        self.maxpool5 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.inception5 = InceptionBlock(inplanes = 256, planes_1x1 = 128, planes_3x3_mid = 128,
                                         planes_3x3 = 128, planes_5x5_mid = 128, planes_5x5 = 128, planes_maxpool = 128)
        self.inception55 = InceptionBlock(inplanes = 512, planes_1x1 = 128, planes_3x3_mid = 128,
                                         planes_3x3 = 128, planes_5x5_mid = 128, planes_5x5 = 128, planes_maxpool = 128)

        self.out = nn.Linear(512 * 7 * 7, n_classes)  # 全连接层得到的结果

    def forward(self, x):
        #x = self.conv1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        #x = self.conv11(x)
        #x = self.conv111(x)
        x = self.inception1(x)
        x = self.inception11(x)
        x = self.maxpool2(x)
        #x = self.conv2(x)
        #x = self.conv22(x)
        x = self.inception2(x)
        x = self.inception22(x)
        x = self.maxpool3(x)
        #x = self.conv3(x)
        #x = self.conv33(x)
        x = self.inception3(x)
        x = self.inception33(x)
        x = self.maxpool4(x)
        #x = self.conv4(x)
        #x = self.conv44(x)
        x = self.inception4(x)
        x = self.inception44(x)
        #x = self.conv5(x)
        x = self.maxpool5(x)
        x = self.inception5(x)
        x = self.inception55(x)
        x = x.view(x.size(0), -1)  # flatten操作，结果为：(batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
