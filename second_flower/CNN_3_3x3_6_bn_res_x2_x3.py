import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, isResAdd = True, need_pool = True):
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
        if isResAdd:
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
    def __init__(self, n_classes):
        super(CNN, self).__init__()
        self.conv1 = BasicBlock(3, 32, 3, False)
        self.conv11 = BasicBlock(32, 32, 3, need_pool=False)
        self.conv111 = BasicBlock(32, 32, 3, need_pool=False)
        self.conv2 = BasicBlock(32, 64, 3)
        self.conv22 = BasicBlock(64, 64, 3, need_pool=False)
        self.conv222 = BasicBlock(64, 64, 3, need_pool=False)
        self.conv3 = BasicBlock(64, 128, 3)
        self.conv33 = BasicBlock(128, 128, 3, need_pool=False)
        self.conv333 = BasicBlock(128, 128, 3, need_pool=False)
        self.conv4 = BasicBlock(128, 256, 3)
        self.conv44 = BasicBlock(256, 256, 3, need_pool=False)
        self.conv444 = BasicBlock(256, 256, 3, need_pool=False)


        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),  # 输出 (512, 14, 14)
            nn.BatchNorm2d(512),
            nn.ReLU(),  # relu层
            nn.MaxPool2d(2),  # 输出 (512, 7, 7)
        )

        self.out = nn.Linear(512 * 7 * 7, n_classes)  # 全连接层得到的结果

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv11(x)
        x = self.conv111(x)
        x = self.conv2(x)
        x = self.conv22(x)
        x = self.conv222(x)
        x = self.conv3(x)
        x = self.conv33(x)
        x = self.conv333(x)
        x = self.conv4(x)
        x = self.conv44(x)
        x = self.conv444(x)
        x = self.conv5(x)
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
