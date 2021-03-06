import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size , need_pool = True ):
        super(BasicBlock, self).__init__()

        self.need_pool = need_pool
        midplanes = int(outplanes / 4)
        self.conv01 = nn.Conv2d(inplanes, midplanes, 1)
        self.bn01 = nn.BatchNorm2d(midplanes)
        self.relu = nn.ReLU(True)

        self.conv1 = nn.Conv2d(midplanes, midplanes, kernel_size, 1, 1)
        self.bn1 = nn.BatchNorm2d(midplanes)
        #self.relu = nn.ReLU(True)


        self.conv02 = nn.Conv2d(midplanes, midplanes, 1)
        self.bn02 = nn.BatchNorm2d(midplanes)

        self.conv2 = nn.Conv2d(midplanes, midplanes, kernel_size, 1, 1)
        self.bn2 = nn.BatchNorm2d(midplanes)

        self.conv03 = nn.Conv2d(midplanes, midplanes, 1)
        self.bn03 = nn.BatchNorm2d(midplanes)

        self.conv3 = nn.Conv2d(midplanes, outplanes, kernel_size, 1, 1)
        self.bn3 = nn.BatchNorm2d(outplanes)

        if self.need_pool:
            self.pool1 = nn.MaxPool2d(2)

        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1),
            nn.BatchNorm2d(outplanes),
        )


    def forward(self, x):
        identity = x
        out = self.conv01(x)
        out = self.bn01(out)
        out = self.relu(out)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv02(out)
        out = self.bn02(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv03(out)
        out = self.bn03(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample != None:
            identity = self.downsample(identity)
            out += identity
        out = self.relu(out)

        if self.need_pool:
            out = self.pool1(out)
        return out


class CNN(nn.Module):
    def __init__(self, n_classes, init_weights=True):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # 输入大小(3,224,224)
            nn.Conv2d(
                in_channels=3,              #
                out_channels=64,            # 要得到几多少个特征图
                kernel_size=3,              # 卷积核大小
                stride=1,                   # 步长
                padding=1,                  # 如果希望卷积后大小跟原来一样，需要设置padding=(kernel_size-1)/2 if stride=1
            ),                              # 输出的特征图为 (64, 224, 224)
            nn.BatchNorm2d(64),
            nn.ReLU(True),                      # relu层
            nn.Conv2d(64, 64, 3, 1, 1),     # 输出    (64, 224, 224)
            nn.BatchNorm2d(64),
            nn.ReLU(True),                      # relu层
            nn.MaxPool2d(kernel_size=2),    # 进行池化操作（2x2 区域）, 输出结果为： (64, 112, 112)
        )

        self.conv11 = BasicBlock(64, 64, 3, False)
        self.conv2 = BasicBlock(64, 128, 3)
        self.conv22 = BasicBlock(128, 128, 3, False)
        self.conv222 = BasicBlock(128, 128, 3, False)
        self.conv2222 = BasicBlock(128, 128, 3, False)

        self.conv3 = BasicBlock(128, 256, 3)
        self.conv33 = BasicBlock(256, 256, 3, False)
        self.conv333 = BasicBlock(256, 256, 3, False)
        self.conv3333 = BasicBlock(256, 256, 3, False)

        self.conv4 = BasicBlock(256, 512, 3)
        self.conv44 = BasicBlock(512, 512, 3, False)
        self.conv444 = BasicBlock(512, 512, 3, False)
        self.conv4444 = BasicBlock(512, 512, 3, False)

        # self.conv5 = BasicBlock(512, 1024, 3)
        # self.conv55 = BasicBlock(1024, 1024, 3, False)
        #
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        #
        # self.fc6 = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.BatchNorm1d(4096),
        #     nn.ReLU(True),  # relu层
        #     nn.Dropout(0.5),
        # )
        #
        # self.fc7 = nn.Sequential(
        #     nn.Linear(4096, 4096),
        #     nn.BatchNorm1d(4096),
        #     nn.ReLU(True),  # relu层
        #     nn.Dropout(0.5),
        # )
        #
        # self.fc8 = nn.Linear(4096, 1000)   # 全连接层得到的结果

        self.out = nn.Linear(512 * 7 * 7, n_classes)   # 全连接层得到的结果

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv11(x)
        x = self.conv2(x)
        x = self.conv22(x)
        x = self.conv222(x)
        x = self.conv2222(x)
        x = self.conv3(x)
        x = self.conv33(x)
        x = self.conv333(x)
        x = self.conv3333(x)
        x = self.conv4(x)
        x = self.conv44(x)
        x = self.conv444(x)
        x = self.conv4444(x)
        # x = self.conv5(x)
        # x = self.conv55(x)
        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)           # flatten操作
        #fc = self.fc6(x)
        #fc = self.fc7(fc)
        #fc = self.fc8(fc)
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

