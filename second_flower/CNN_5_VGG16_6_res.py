import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size, 1, 1)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(True)

        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size, 1, 1)
        self.bn2 = nn.BatchNorm2d(outplanes)

        self.conv3 = nn.Conv2d(outplanes, outplanes, kernel_size, 1, 1)
        self.bn3 = nn.BatchNorm2d(outplanes)

        self.pool1 = nn.MaxPool2d(2)

        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1),
            nn.BatchNorm2d(outplanes),
        )


    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample != None:
            identity = self.downsample(identity)
            out += identity
        out = self.relu(out)

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

        self.conv2 = BasicBlock(64, 128, 3)

        self.conv3 = BasicBlock(128, 256, 3)

        self.conv4 = BasicBlock(256, 512, 3)

        self.conv5 = BasicBlock(512, 512, 3)

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
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
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

