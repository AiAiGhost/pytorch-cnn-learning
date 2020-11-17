import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
import settings
import funcs
from datetime import datetime
from tensorboardX import SummaryWriter

#为了多机测试数据相同，正式版不需要
#funcs.set_random_seed_same(8888)
funcs.set_random_seed_same(1000)

input_size = 224  #图像的总尺寸
num_classes = 102  #标签的种类数
num_epochs = 200  #训练的总循环周期
batch_size = 12  #一个撮（批次）的大小
learnrate=0.0001

pretrained=False

#############################1#############################
import CNN_1_cnn_1 as cnn
logs_path = settings.get_log_path() + 'g_1_cnn_1'

##############################2#############################
# import CNN_2_big_1 as cnn
# logs_path = settings.get_log_path() + 'g_2_big_1'

##############################3#############################
# import CNN_2_big_2 as cnn
# logs_path = settings.get_log_path() + 'g_2_big_2'

##############################4#############################
# import CNN_2_big_3 as cnn
# logs_path = settings.get_log_path() + 'g_2_big_3'

##############################5#############################
# import CNN_2_big_4 as cnn
# logs_path = settings.get_log_path() + 'g_2_big_4'

##############################6#############################
# import CNN_3_3x3_1 as cnn
# logs_path = settings.get_log_path() + 'g_3_3x3_1'

##############################7#############################
# import CNN_3_3x3_4_bn_class as cnn
# logs_path = settings.get_log_path() + 'g_3_3x3_4_bn_class'

##############################8#############################
# import CNN_3_3x3_6_bn_res_arelu as cnn
# logs_path = settings.get_log_path() + 'g_3_3x3_6_bn_res_arelu'

##############################9#############################
# import CNN_3_3x3_6_bn_res_x2 as cnn
# logs_path = settings.get_log_path() + 'g_3_3x3_6_bn_res_x2'

##############################9#############################
# import CNN_3_3x3_6_bn_res_x2_7x7_1 as cnn
# logs_path = settings.get_log_path() + 'g_3_3x3_6_bn_res_x2_7x7_1'

##############################9#############################
# import CNN_3_3x3_6_bn_res_x2_7x7_2 as cnn
# logs_path = settings.get_log_path() + 'g_3_3x3_6_bn_res_x2_7x7_2'

##############################9#############################
# import CNN_3_3x3_6_bn_res_x2_7x7_n1 as cnn
# logs_path = settings.get_log_path() + 'g_3_3x3_6_bn_res_x2_7x7_n1'

##############################9#############################
# import CNN_3_3x3_6_bn_res_x2_7x7_n2 as cnn
# logs_path = settings.get_log_path() + 'g_3_3x3_6_bn_res_x2_7x7_n2'

##############################11#############################
# import CNN_3_3x3_6_bn_res_x2_x3_n3 as cnn
# logs_path = settings.get_log_path() + 'g_3_3x3_6_bn_res_x2_x3_n3'

##############################10#############################
# import CNN_3_3x3_6_bn_res_x2_x5 as cnn
# logs_path = settings.get_log_path() + 'g_3_3x3_6_bn_res_x2_x5'

##############################11#############################
# import CNN_3_3x3_6_bn_res_x3 as cnn
# logs_path = settings.get_log_path() + 'g_3_3x3_6_bn_res_x3'

##############################12#############################
# import CNN_3_3x3_6_bn_res_x2_x17 as cnn
# logs_path = settings.get_log_path() + 'g_3_3x3_6_bn_res_x2_x17'

##############################13#############################
# import CNN_3_3x3_6_bn_res_x2_x17_bias as cnn
# logs_path = settings.get_log_path() + 'g_3_3x3_6_bn_res_x2_x17_bias'

##############################14#############################
# import CNN_3_3x3_6_bn_res_x2_x17_add as cnn
# logs_path = settings.get_log_path() + 'g_3_3x3_6_bn_res_x2_x17_add'

##############################15#############################
# import CNN_3_3x3_6_bn_res_x2_x17_add2 as cnn
# logs_path = settings.get_log_path() + 'g_3_3x3_6_bn_res_x2_x17_add2_no1x1'

##############################16#############################
# import CNN_3_3x3_7_bn_res_Bottleneck as cnn
# logs_path = settings.get_log_path() + 'g_3_3x3_7_bn_res_Bottleneck'

##############################17#############################
# import CNN_3_3x3_7_bn_res_Bottleneck_add as cnn
# logs_path = settings.get_log_path() + 'g_3_3x3_7_bn_res_Bottleneck_add'

##############################18#############################
# batch_size = 8
# import CNN_5_VGG16_1 as cnn
# logs_path = settings.get_log_path() + 'g_5_VGG16_1'

##############################19#############################
# batch_size = 8
# import CNN_5_VGG16_4_BN as cnn
# logs_path = settings.get_log_path() + 'g_5_VGG16_4_BN'

##############################20#############################
# import CNN_5_VGG16_4_BN_no_full as cnn
# logs_path = settings.get_log_path() + 'g_5_VGG16_4_BN_no_full'

##############################21#############################
# import CNN_5_VGG16_6_res as cnn
# logs_path = settings.get_log_path() + 'g_5_VGG16_6_res'

##############################22#############################
# batch_size = 8
# import CNN_5_VGG16_7_res_2x as cnn
# logs_path = settings.get_log_path() + 'g_5_VGG16_7_res_2x'

##############################23#############################
# import CNN_5_VGG16_8_res_bottlenet as cnn
# logs_path = settings.get_log_path() + 'g_5_VGG16_8_res_bottlenet'

# 请替换使用下面 cnn.CNN(num_classes,pretrained)
# pretrained=True
# import CNN_6_resnet50 as cnn
# logs_path = settings.get_log_path() + '6_resnet50_true'

# pretrained=True
# import CNN_6_resnet34 as cnn
# logs_path = settings.get_log_path() + '6_resnet34_true'

# pretrained=False
# import CNN_6_resnet50 as cnn
# logs_path = settings.get_log_path() + 'a_6_resnet50_false'

# pretrained=False
# import CNN_6_resnet34 as cnn
# logs_path = settings.get_log_path() + 'a_6_resnet34_false'

# pretrained=False
# import CNN_6_resnet18 as cnn
# logs_path = settings.get_log_path() + 'a_6_resnet18_false_zeng3'

# pretrained=False
# import CNN_6_resnet101 as cnn
# logs_path = settings.get_log_path() + 'a_6_resnet101_false'

# pretrained=True
# import CNN_7_googlenet_01 as cnn
# logs_path = settings.get_log_path() + '7_googlenet_01_true'

# pretrained=False
# import CNN_7_googlenet_01 as cnn
# logs_path = settings.get_log_path() + '7_googlenet_01_x_false'

# import CNN_7_self_google_01 as cnn
# logs_path = settings.get_log_path() + '7_self_google_01'

# import CNN_7_self_google_all_02 as cnn
# logs_path = settings.get_log_path() + '7_self_google_all_02'

# import CNN_7_self_google_all_02_1x1 as cnn
# logs_path = settings.get_log_path() + '7_self_google_all_02_1x1'

# import CNN_7_self_google_allx2_03 as cnn
# logs_path = settings.get_log_path() + '7_self_google_allx2_03'

# import CNN_7_self_google_allx2_03_1x1 as cnn
# logs_path = settings.get_log_path() + '7_self_google_allx2_03_1x1'

# import CNN_7_self_google_all_02_withx7 as cnn
# logs_path = settings.get_log_path() + '7_self_google_all_02_withx7'

# import CNN_7_self_google_all_02_big1x1 as cnn
# logs_path = settings.get_log_path() + '7_self_google_all_02_big1x1'

# import CNN_7_self_google_all_02_allx3x3 as cnn
# logs_path = settings.get_log_path() + '7_self_google_all_02_allx3x3'

# import CNN_7_self_google_all_02_large as cnn
# logs_path = settings.get_log_path() + '7_self_google_all_02_large'

# import CNN_7_self_google_allx2_03_big1x1 as cnn
# logs_path = settings.get_log_path() + '7_self_google_allx2_03_big1x1'

# import CNN_7_self_google_allx2_03_bigger1x1 as cnn
# logs_path = settings.get_log_path() + '7_self_google_allx2_03_bigger1x1'

# import CNN_8_self_google_mix_01 as cnn
# logs_path = settings.get_log_path() + '8_self_google_mix_01'

# import CNN_8_self_google_mix_02_add as cnn
# logs_path = settings.get_log_path() + '8_self_google_mix_02_add'

# import CNN_8_self_google_mix_01_big as cnn
# logs_path = settings.get_log_path() + '8_self_google_mix_01_big'

# import CNN_8_self_google_mix_02_add_ds as cnn
# logs_path = settings.get_log_path() + '8_self_google_mix_02_add_ds'

# import CNN_8_self_google_mix_02_add_ds_more as cnn
# logs_path = settings.get_log_path() + '8_self_google_mix_02_add_ds_more'



device = settings.get_device()
DATASET_PATH = settings.get_dataset_path() # the dataset file or root folder path.

net = cnn.CNN(num_classes)
#net = cnn.CNN(num_classes,pretrained)
net = net.to(device)

print("==========================================" + logs_path)
print(net)
print("参数总个数：{}  ".format(sum(x.numel() for x in net.parameters())))
print("==========================================" + logs_path)

# Creates writer2 object with auto generated file name
# The log directory will be something like 'runs/Aug20-17-20-33'
writer = SummaryWriter(logs_path)
dummy_input = torch.rand(10, 3, input_size, input_size).to(device)
#writer.add_graph(net, (dummy_input,) )

data_transform = transforms.Compose([
        #transforms.Resize(256),
        #transforms.CenterCrop(224)
        transforms.RandomRotation(90),#随机旋转，-45到45度之间随机选
        #transforms.Resize(300),
        transforms.RandomChoice([transforms.Resize(224),transforms.Resize(240),transforms.Resize(256),transforms.Resize(280),
                                 transforms.Resize(300),transforms.Resize(320),transforms.Resize(360),transforms.Resize(400), transforms.Resize(500)]),
        transforms.RandomCrop(224),
        #transforms.RandomResizedCrop(224),#随机裁剪
        transforms.RandomHorizontalFlip(p=0.5),#随机水平翻转 选择一个概率概率
        transforms.RandomVerticalFlip(p=0.5),#随机垂直翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),#参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
        #transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.2, hue=0.2),#参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
        transforms.RandomGrayscale(p=0.025),#概率转换成灰度率，3通道就是R=G=B
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#均值，标准差
    ])

valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),#从中心开始裁剪
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#均值，标准差
    ])

image_datasets = datasets.ImageFolder(DATASET_PATH + "train/", data_transform)
#print(image_datasets.classes)
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=True)

valid_datasets = datasets.ImageFolder(DATASET_PATH + "valid/", valid_transform)
#print(image_datasets.classes)
valid_loaders = torch.utils.data.DataLoader(valid_datasets, batch_size=batch_size, shuffle=False)

# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = optim.Adam(net.parameters(), lr=learnrate)  # 定义优化器，普通的随机梯度下降算法
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 40, gamma = 0.1, last_epoch=-1)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer,  milestones = [130,180], gamma = 0.1, last_epoch=-1)

def accuracy(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum().cpu()
    return rights, len(labels)

dt = datetime.now()
val_rightsrate_best = 0

# for epoch in range(num_epochs):
#     scheduler.step()
#     print("学习率：{},{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))

# 开始训练循环
for epoch in range(num_epochs):
    # 当前epoch的结果保存下来
    train_rights = []

    for batch_idx, (inputs, labels) in enumerate(dataloaders):
        #print(labels)
        inputs = inputs.to(device)
        labels = labels.to(device)
        # if len(labels) < batch_size :
        #     continue

        net.train()
        optimizer.zero_grad()
        output = net(inputs)
        loss = criterion(output, labels)
        right = accuracy(output, labels)
        train_rights.append(right)

        index = (batch_idx * batch_size + epoch * len(dataloaders.dataset))
        writer.add_scalar('loss', loss.data, global_step=index)

        if batch_idx % 10 == 0:
            print("[{},{},{}/{}]损失:{:.6f}% ======={}".
                  format(epoch,batch_idx,(index),len(dataloaders.dataset),loss.data,str((datetime.now() - dt))))

        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
            #rightsrate = 100. * train_r[0].numpy() / train_r[1]
            train_rightsrate = 100. * train_r[0] / train_r[1]
            #print(train_rights)

            net.eval()
            val_rights = []

            for (inputs, labels) in valid_loaders:
                inputs = inputs.to(device)
                labels = labels.to(device)
                output = net(inputs)
                right = accuracy(output, labels)
                val_rights.append(right)

            val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))
            val_rightsrate = 100. * val_r[0] / val_r[1]
            print("[{},{},{}/{}]损失:{:.6f}% ==正确率=测试集: {:.2f}%,验证集: {:.2f}%  ==={}".
                  format(epoch,batch_idx,(index),len(dataloaders.dataset),loss.data,(train_rightsrate),val_rightsrate,str((datetime.now() - dt))))

            if (val_rightsrate > val_rightsrate_best) :
                model_file_name = logs_path + '/cnn_' + datetime.strftime(dt, "%Y%m%d_%H%M%S") + '.pkl'
                torch.save(net, model_file_name)
                val_rightsrate_best = val_rightsrate
                print("保存模型:" + model_file_name)

            writer.add_scalar('acc', train_rightsrate, global_step=index)
            writer.add_scalar('acc_val', val_rightsrate, global_step=index)
            train_rights.clear()

    scheduler.step()
    print("学习率：{},{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))

