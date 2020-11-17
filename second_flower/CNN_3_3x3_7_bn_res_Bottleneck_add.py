import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, need_pool = True):
        super(Bottleneck, self).__init__()
        isResAdd = need_pool
        self.need_pool = need_pool
        midplanes = int(outplanes / 4)
        self.conv1 = nn.Conv2d(inplanes, midplanes, 1)
        self.bn1 = nn.BatchNorm2d(midplanes)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(midplanes, midplanes, kernel_size, 1, 1)
        self.bn2 = nn.BatchNorm2d(midplanes)
        #self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(midplanes, outplanes, 1)
        self.bn3 = nn.BatchNorm2d(outplanes)

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
        out = self.relu1(out)

        out = self.conv3(out)
        out = self.bn3(out)
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
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 7, stride = 2, padding = 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.conv11 = Bottleneck(32, 32, 3, need_pool=False)
        self.conv111 = Bottleneck(32, 32, 3, need_pool=False)
        self.conv2 = Bottleneck(32, 64, 3)
        self.conv22 = Bottleneck(64, 64, 3, need_pool=False)
        self.conv222= Bottleneck(64, 64, 3, need_pool=False)
        self.conv2222= Bottleneck(64, 64, 3, need_pool=False)
        self.conv22222= Bottleneck(64, 64, 3, need_pool=False)
        self.conv3 = Bottleneck(64, 128, 3)
        self.conv33= Bottleneck(128, 128, 3, need_pool=False)
        self.conv333= Bottleneck(128, 128, 3, need_pool=False)
        self.conv3333= Bottleneck(128, 128, 3, need_pool=False)
        self.conv33333= Bottleneck(128, 128, 3, need_pool=False)
        self.conv333333= Bottleneck(128, 128, 3, need_pool=False)
        self.conv3333333= Bottleneck(128, 128, 3, need_pool=False)
        self.conv33333333= Bottleneck(128, 128, 3, need_pool=False)
        self.conv4 = Bottleneck(128, 256, 3)
        self.conv44= Bottleneck(256, 256, 3, need_pool=False)
        self.conv444= Bottleneck(256, 256, 3, need_pool=False)
        self.conv4444= Bottleneck(256, 256, 3, need_pool=False)
        self.conv44444= Bottleneck(256, 256, 3, need_pool=False)
        self.conv444444= Bottleneck(256, 256, 3, need_pool=False)
        self.conv4444444= Bottleneck(256, 256, 3, need_pool=False)
        self.conv44444444= Bottleneck(256, 256, 3, need_pool=False)

        self.conv5 = Bottleneck(256, 512, 3)
        self.conv55= Bottleneck(512, 512, 3, need_pool=False)
        self.conv555= Bottleneck(512, 512, 3, need_pool=False)
        self.conv5555= Bottleneck(512, 512, 3, need_pool=False)

        self.out = nn.Linear(512 * 7 * 7, n_classes)  # 全连接层得到的结果

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv11(x)
        x = self.conv111(x)
        x = self.conv2(x)
        x = self.conv22(x)
        x = self.conv222(x)
        x = self.conv2222(x)
        x = self.conv22222(x)
        x = self.conv3(x)
        x = self.conv33(x)
        x = self.conv333(x)
        x = self.conv3333(x)
        x = self.conv33333(x)
        x = self.conv333333(x)
        x = self.conv3333333(x)
        x = self.conv33333333(x)
        x = self.conv4(x)
        x = self.conv44(x)
        x = self.conv444(x)
        x = self.conv4444(x)
        x = self.conv44444(x)
        x = self.conv444444(x)
        x = self.conv4444444(x)
        x = self.conv44444444(x)
        x = self.conv5(x)
        x = self.conv55(x)
        x = self.conv555(x)
        x = self.conv5555(x)
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
