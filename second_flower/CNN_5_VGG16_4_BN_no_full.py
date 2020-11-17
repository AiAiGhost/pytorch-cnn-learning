import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, n_classes):
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

        self.conv2 = nn.Sequential(         #
            nn.Conv2d(64, 128, 3, 1, 1),     # 输出 (128, 112, 112)
            nn.BatchNorm2d(128),
            nn.ReLU(True),                      # relu层
            nn.Conv2d(128, 128, 3, 1, 1),     # 输出 (128, 112, 112)
            nn.BatchNorm2d(128),
            nn.ReLU(True),                      # relu层
            nn.MaxPool2d(2),                # 输出 (128, 56, 56)
        )

        self.conv3 = nn.Sequential(         #
            nn.Conv2d(128, 256, 3, 1, 1),     # 输出 (256, 56, 56)
            nn.BatchNorm2d(256),
            nn.ReLU(True),                      # relu层
            nn.Conv2d(256, 256, 3, 1, 1),     # 输出 (256, 56, 56)
            nn.BatchNorm2d(256),
            nn.ReLU(True),                      # relu层
            nn.Conv2d(256, 256, 3, 1, 1),     # 输出 (256, 56, 56)
            nn.BatchNorm2d(256),
            nn.ReLU(True),                      # relu层
            nn.MaxPool2d(2),                # 输出 (256, 28, 28)
        )

        self.conv4 = nn.Sequential(         #
            nn.Conv2d(256, 512, 3, 1, 1),     # 输出 (512, 28, 28)
            nn.BatchNorm2d(512),
            nn.ReLU(True),                      # relu层
            nn.Conv2d(512, 512, 3, 1, 1),     # 输出 (512, 28, 28)
            nn.BatchNorm2d(512),
            nn.ReLU(True),                      # relu层
            nn.Conv2d(512, 512, 3, 1, 1),     # 输出 (512, 28, 28)
            nn.BatchNorm2d(512),
            nn.ReLU(True),                      # relu层
            nn.MaxPool2d(2),                # 输出 (512, 14, 14)
        )

        self.conv5 = nn.Sequential(         #
            nn.Conv2d(512, 512, 3, 1, 1),     # 输出 (512, 14, 14)
            nn.BatchNorm2d(512),
            nn.ReLU(True),                      # relu层
            nn.Conv2d(512, 512, 3, 1, 1),     # 输出 (512, 14, 14)
            nn.BatchNorm2d(512),
            nn.ReLU(True),                      # relu层
            nn.Conv2d(512, 512, 3, 1, 1),     # 输出 (512, 14, 14)
            nn.BatchNorm2d(512),
            nn.ReLU(True),                      # relu层
            nn.MaxPool2d(2),                # 输出 (512, 7, 7)
        )

        #self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

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

