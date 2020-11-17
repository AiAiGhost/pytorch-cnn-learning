import torch
import torch.nn as nn
from torchvision import models

class CNN(nn.Module):
    def __init__(self, n_classes,pretrained):
        super(CNN, self).__init__()
        self.resnet = models.resnet34(pretrained)

        self.out = nn.Linear(1000, n_classes)  # 全连接层得到的结果

    def forward(self, x):
        x = self.resnet(x)
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

