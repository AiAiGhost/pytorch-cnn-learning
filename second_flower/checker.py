import torch
import torch.nn as nn
# import cv2
import settings
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
from PIL import Image



model_file_path = settings.get_log_path() + 'a_6_resnet18_false/cnn_20200917_132128.pkl'
img_path = "./../image_07653.jpg"

device = settings.get_device()

valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),#从中心开始裁剪
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#均值，标准差
    ])

DATASET_PATH = settings.get_dataset_path_second()
valid_datasets = datasets.ImageFolder(DATASET_PATH + "valid/", valid_transform)

img = Image.open(img_path)
#plt.imshow(img)
#plt.show()

inputs = valid_transform(img).unsqueeze(0)
inputs = inputs.to(device)
print(inputs)


model = torch.load(model_file_path)
print(model)

model.eval()
output = model(inputs)
print(output)
pred = torch.max(output.data, 1)
print(pred)
pred = pred[1].cpu()
#print(pred)
print("=======================:" + valid_datasets.classes[pred])
