import warnings

import matplotlib.pyplot as plt
import numpy as np
import visdom
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import dataset
from model.GoogLeNet import *
from model.LeNet import *
from model.RestNet import *
from model.RestNet18 import *
from model.VGG import *

warnings.filterwarnings("ignore")
"""
根据自己显卡的显存调整batch_size
"""
batch_size = 32
learning_rate = 1e-3
device = torch.device('cuda:0')


def evaluate(model, loader, mode='val'):
    # 执行模式下的神经网络
    # 不记录梯度，不进行反向传播
    # 权重不会更新
    model.eval()
    correct = 0
    total = len(loader.dataset)
    step = 0
    count = int(total / loader.batch_size)
    with tqdm(total=count, desc=f'{mode} iteration :{count} ', postfix=dict,
              mininterval=0.5) as val_dataset:
        for image, label in loader:
            image, label = image.to(device), label.to(device)
            with torch.no_grad():
                logics = model(image)
                predict = logics.argmax(dim=1)
            correct += torch.eq(predict, label).sum().float().item()
            step += 1
            val_dataset.update(1)
    return correct / total


def train(model, model_name, dataset_path, epochs=100, batch_size=32, learning_rate=1e-3, use_visdom=False):
    # 选择模型
    assert isinstance(model, nn.Module)
    # 加载数据集
    train_db = dataset(dataset_path, 224, mode='train')
    val_db = dataset(dataset_path, 224, mode='val')
    test_db = dataset(dataset_path, 224, mode='test')
    # 设置dataloader
    train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_db, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_db, batch_size=batch_size, shuffle=True, num_workers=4)
    # 设置优化器，损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss().to(device)
    # 记录信息
    iteration = []
    train_loss_iter = []
    val_acc_iter = []
    best_weights = ''
    best_acc = 0
    global_step = 1
    # 训练进度监测，允许使用 visdom 或者 tqdm
    if use_visdom:
        viz = visdom.Visdom()
        viz.line([0], [-1], win='loss', opts=dict(title='loss'))
        viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))
    # 训练模型
    for epoch in range(epochs):
        # 模型进入训练模式，该模式下会记录梯度信息
        model.train()
        # 从dataloader中取出图片和标签
        # 神经网络识别图片给出预测值logics（经过softmax处理后称之为predict）
        # 预测值与标签值比较，损失函数使用交叉熵
        # 清空上次的梯度（本次计算的梯度依然保留）
        # 反向传播
        # 打印训练信息
        count_loss = 0
        with tqdm(total=len(train_loader),
                  desc=f'{model_name} Epoch:{epoch}/{epochs};Iteration :{len(train_loader)}',
                  postfix=dict, mininterval=0.5) as train_bar:
            for step, (image, label) in enumerate(train_loader):
                image, label = image.to(device), label.to(device)
                logics = model(image)
                loss = loss_function(logics, label)
                count_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()  # 反向传播
                optimizer.step()
                if use_visdom:
                    viz.line([loss.item()], [global_step], win='loss', update='append')
                global_step += 1
                train_bar.set_postfix(**{'loss': count_loss / (step + 1)})
                train_bar.update(1)
            # 验证集，使用评估模式，该模式下神经网络不会记录梯度
        val_acc = evaluate(model, val_loader)
        iteration.append(epoch + 1)
        train_loss_iter.append(count_loss / (step + 1))
        val_acc_iter.append(val_acc)
        # 输出验证信息
        print('val acc:%.4f' % val_acc)
        if use_visdom:
            viz.line([val_acc], [global_step], win='val_acc', update='append')
            # 保存网络状态
        if isinstance(model_name, nn.Module):
            model_name = 'unsetModelName'
        torch.save(model.state_dict(),
                   f'{model_name}-Epoch_{epoch + 1}-loss{round(count_loss / step + 1, 4)}-val acc_{round(val_acc, 4)}.pth')
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = f'{model_name}-Epoch_{epoch + 1}-loss{round(count_loss / step + 1, 4)}-val acc_{round(val_acc, 4)}.pth'
        print(f'save Epoch_{epoch + 1}-loss{round(count_loss / step + 1, 4)}-val acc_{round(val_acc, 4)}.pth')
    # 测试模型
    model.load_state_dict(torch.load(best_weights))
    print('loaded from check point!')
    test_acc = evaluate(model, test_loader, mode='test')
    print('test acc:', test_acc)
    with open('logs.txt', mode='a+') as f:
        f.write('############################\n')
        f.write(model_name + ' ' + dataset_path + '\n')
        f.write(str(iteration) + '\n')
        f.write(str(train_loss_iter) + '\n')
        f.write(str(val_acc_iter) + '\n')
        f.write(str(test_acc) + '\n')
        f.write(str(sum(x.numel() for x in model.parameters())) + '\n')


def draw(log_path):
    # 画图细节，与神经网络无关
    with open(log_path, mode='r+')as f:
        lines = f.readlines()
        content = []
        for index in range(len(lines)):
            if lines[index] == '############################\n':
                item = lines[index + 1:index + 7]
                content.append(item)
        fig, ax = plt.subplots(2, 2, figsize=(18, 12))
        font = {'family': 'Times New Roman',
                'weight': 'normal',
                'size': 18,
                }
        ax_train_loss = ax[0, 0]
        ax_val_loss = ax[0, 1]
        ax_test_acc = ax[1, 0]
        ax_parameter = ax[1, 1]

        ax_train_loss.set_xlabel('iter', font)
        ax_train_loss.set_ylabel('train_loss', font)
        ax_train_loss.set_title('Train Loss', font=font)

        ax_val_loss.set_xlabel('iter', font)
        ax_val_loss.set_ylabel('val_loss', font)
        ax_val_loss.set_title('Val Loss', font=font)

        ax_test_acc.set_xlabel('model', font)
        ax_test_acc.set_ylabel('test_acc', font)
        ax_test_acc.set_title('Test Accuracy', font=font)

        ax_parameter.set_xlabel('model', font)
        ax_parameter.set_ylabel('model_parameter', font)
        ax_parameter.set_title('Model parameter', font=font)

        model_names = []
        color = ['r', 'g', 'b', 'y', 'p']
        test_accs = []
        parameters = []
        for item in content:
            model_name, dataset_name = item[0][:-1].split(' ')
            iteration = eval(item[1][:-1])
            train_loss = eval(item[2][:-1])
            val_loss = eval(item[3][:-1])
            test_accs.append(float(item[4][:-1]))
            parameters.append(float(item[5][:-1]))
            ax_train_loss.plot(iteration, train_loss, label=model_name, color=color[content.index(item)])
            ax_val_loss.plot(iteration, val_loss, label=model_name, color=color[content.index(item)])
            model_names.append(model_name)
        model_num = len(model_names)
        x_coor = np.arange(model_num)
        ax_test_acc.bar(model_names, test_accs)
        ax_test_acc.set_xticks(x_coor, model_names)
        ax_parameter.bar(model_names, parameters)
        ax_parameter.set_xticks(x_coor, model_names)
        for x, y in zip(x_coor, test_accs):
            ax_test_acc.text(x, y, '%.2f' % y, ha='center', va='bottom', fontsize=10)
        for x, y in zip(x_coor, parameters):
            ax_parameter.text(x, y + 0.5, '%f' % y, ha='center', va='bottom', fontsize=10)
        ax_train_loss.legend()
        ax_val_loss.legend()
        plt.show()


if __name__ == '__main__':
    with open('logs.txt', mode='w+') as f:
        f.truncate(0)
    device = torch.device('cuda:0')
    LeNet = LeNet(5).to(device)
    GoogLeNet = GoogLeNet(5, aux_logits=False).to(device)  # 这样子训练GooLeNet不会达到最好的状态
    VGG16 = vgg(model_name='vgg16', num_classes=5).to(device)
    ResNet18 = ResNet18(5).to(device)
    ResNet50 = resnet50(num_classes=5).to(device)
    model_name = [[LeNet, 'LeNet'], [GoogLeNet, 'GoogLeNet'], [VGG16, 'VGG16'], [ResNet18, 'ResNet18'],
                  [ResNet50, 'ResNet50']]
    for item in model_name:
        train(item[0], item[1], 'PetImages', epochs=100)
    draw('logs.txt')
