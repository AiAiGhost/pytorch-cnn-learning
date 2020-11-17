import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, n_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # 输入大小(3,224,224)
            nn.Conv2d(
                in_channels=3,  #
                out_channels=32,  # 要得到几多少个特征图
                kernel_size=5,  # 卷积核大小
                stride=1,  # 步长
                padding=2,  # 如果希望卷积后大小跟原来一样，需要设置padding=(kernel_size-1)/2 if stride=1
            ),  # 输出的特征图为
            nn.BatchNorm2d(32),
            nn.ReLU(),  # relu层
            nn.MaxPool2d(kernel_size=2),  # 进行池化操作（2x2 区域）, 输出结果为： (32, 112, 112)
        )
        self.conv2 = nn.Sequential(  # 下一个套餐的输入 (32, 112, 112)
            nn.Conv2d(32, 64, 5, 1, 2),  # 输出 (64, 112, 112)
            nn.BatchNorm2d(64),
            nn.ReLU(),  # relu层
            nn.MaxPool2d(2),  # 输出 (64, 56, 56)
        )

        self.conv3 = nn.Sequential(  # 下一个套餐的输入 (64, 56, 56)
            nn.Conv2d(64, 128, 5, 1, 2),  # 输出 (128, 56, 56)
            nn.BatchNorm2d(128),
            nn.ReLU(),  # relu层
            nn.MaxPool2d(2),  # 输出 (128, 28, 28)
        )

        self.out = nn.Linear(128 * 28 * 28, n_classes)  # 全连接层得到的结果

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
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
