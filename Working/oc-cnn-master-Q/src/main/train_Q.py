import torch
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision.models as models
import torchvision.transforms as transforms
import pickle as cp
import matplotlib.pyplot as plt
from subprocess import call
import visdom
from Config import *
from classifier import *
import cv2
import scipy.io
from scipy import misc
from sklearn import svm
from sklearn import metrics
import joblib
from sklearn.metrics import accuracy_score
import sys
import copy
import h5py
import time
import pickle
import os
import random
import argparse
import numpy as np
from light_cnn import *
from oneClassDataset import OneClassDataset
from classifier import classifier_nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from Config import *
import VGG_FACE_torch


def getExtractor(model_type, pre_trained_flag):
    if model_type == 'alexnet':
        model = torchvision.models.alexnet(pretrained=pre_trained_flag)
        new_classifier = nn.Sequential(*list(model.classifier.children())[:-2])
        model.classifier = new_classifier
    elif model_type == 'vgg16':
        model = torchvision.models.vgg16(pretrained=pre_trained_flag)
        new_classifier = nn.Sequential(*list(model.classifier.children())[:-3])
        model.classifier = new_classifier
    elif model_type == 'vgg19':
        model = torchvision.models.vgg19(pretrained=pre_trained_flag)
        new_classifier = nn.Sequential(*list(model.classifier.children())[:-3])
        model.classifier = new_classifier
    elif model_type == 'vgg16bn':
        model = torchvision.models.vgg16_bn(pretrained=pre_trained_flag)
        new_classifier = nn.Sequential(*list(model.classifier.children())[:-3])
        model.classifier = new_classifier
    elif model_type == 'vgg19bn':
        model = torchvision.models.vgg19_bn(pretrained=pre_trained_flag)
        new_classifier = nn.Sequential(*list(model.classifier.children())[:-3])
        model.classifier = new_classifier
    elif model_type == 'resnet18':
        model = torchvision.models.resnet18(pretrained=pre_trained_flag)
        model.fc = nn.Sequential()
    elif model_type == 'resnet34':
        model = torchvision.models.resnet34(pretrained=pre_trained_flag)
        model.fc = nn.Sequential()
    elif model_type == 'resnet50':
        model = torchvision.models.resnet50(pretrained=pre_trained_flag)
        model.fc = nn.Sequential()
    elif model_type == 'vggface':
        model = VGG_FACE_torch.VGG_FACE_torch
        model.load_state_dict(torch.load('VGG_FACE.pth'))
        model = model[:-3]
    elif model_type == 'lightcnn':
        model = LightCNN_29Layers_v2(num_classes=80013)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load('LightCNN_29Layers_V2_checkpoint.pth')['state_dict'])
        new_model = nn.Sequential(*list(model.module.children())[:-1])
    else:
        raise argparse.ArgumentTypeError(
            'models supported in this version of code are alexnet, vgg16, vgg19, vgg16bn, vgg19bn. \n '
            'Enter model_type as one fo this argument')
    return model


def addGaussianNoise(FeatureVectors, input_label, mean=0, std=0.01, shuffle=True):
    # 一个正常输入对应一个噪声
    noiseLabel = torch.zeros(len(FeatureVectors))
    gaussian_data = np.random.normal(mean, std, FeatureVectors.shape)
    gaussian_data = torch.from_numpy(gaussian_data)
    data = torch.cat((FeatureVectors, gaussian_data), dim=0)
    labels = torch.cat((input_label, noiseLabel))
    # 原论文没有的操作
    # 将输入和噪声打乱
    if shuffle:
        tmp = []
        for index, tensor in enumerate(data):
            tmp.append([tensor, labels[index]])
        random.shuffle(tmp)
        data = [torch.unsqueeze(i[0], dim=0) for i in tmp]
        labels = [i[1] for i in tmp]
        data = torch.cat(data, dim=0)
    return data, labels


def getClassifier(featureDimension):
    return classifier_nn(featureDimension)


def OCCNN_train(params):
    datasetPath = params.datasetPath
    positiveClass = params.positiveClass
    extractor = params.extractor
    extractor_pretrain = params.extractor_pretrain
    batch_size = params.batch_size
    num_workers = params.num_workers
    featureDimension = params.D
    lr = params.lr
    epochs = params.epochs
    gpu = params.gpu_flag
    #####################################################################
    train_loader = DataLoader(
        OneClassDataset(path=datasetPath, positiveClass=positiveClass, mode='train'),
        batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(
        OneClassDataset(path=datasetPath, positiveClass=positiveClass, mode='val'),
        batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(
        OneClassDataset(path=datasetPath, positiveClass=positiveClass, mode='test'),
        batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = getExtractor(extractor, extractor_pretrain)
    classifier = getClassifier(featureDimension)
    InstanceNormal = nn.InstanceNorm1d(1, affine=False)
    relu = nn.ReLU()
    loss_function = nn.CrossEntropyLoss()
    model_optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=lr)

    if gpu:
        device = torch.device('cuda:0')
        model = model.to(device)
        classifier = classifier.to(device)
        relu = relu.to(device)
        loss_function = loss_function.to(device)

    # 记录信息
    iteration = []
    train_loss_iter = []
    val_acc_iter = []
    best_weights = ''
    best_acc = 0
    global_step = 1

    for epoch in range(epochs):
        model.train()
        count_loss = 0
        with tqdm(total=len(train_loader),
                  desc=f'{extractor} Epoch:{epoch}/{epochs};Iteration :{len(train_loader)}',
                  postfix=dict, mininterval=0.5) as train_bar:
            for step, (image, label) in enumerate(train_loader):
                image, label = image.to(device), label.to(device)
                # 给图像添加高斯噪声

                # 提取特征向量,进行像素上的归一化
                featureVectors = model(image)
                featureVectors = featureVectors.view(batch_size, 1, featureDimension)
                featureVectors = InstanceNormal(featureVectors)
                featureVectors = featureVectors.view(batch_size, featureDimension)
                # 特征向量添加高斯噪声
                data, labels = addGaussianNoise(featureVectors, label, shuffle=False)
                data = relu(data)
                out = classifier(data)
                # 梯度清零
                model_optimizer.zero_grad()
                classifier_optimizer.zero_grad()
                # 计算loss
                loss = loss_function(out, labels)
                count_loss += loss.item()
                # 反向传播
                loss.backward()
                # 更新参数
                model_optimizer.step()
                classifier_optimizer.step()
                global_step += 1
                train_bar.set_postfix(**{'loss': count_loss / (step + 1)})
                train_bar.update(1)
            # 验证集，使用评估模式，该模式下神经网络不会记录梯度
        val_acc = evaluate(model, classifier, val_loader)
        iteration.append(epoch + 1)
        train_loss_iter.append(count_loss / (step + 1))
        val_acc_iter.append(val_acc)
        # 输出验证信息
        print('val acc:%.4f' % val_acc)

        # 保存网络状态
        if val_acc > best_acc:
            best_acc = val_acc
            if os.path.exists(best_weights):
                # 删除之前保存的权值文件
                os.remove(best_weights)
            torch.save(model.state_dict(),
                       f'{extractor}-Epoch_{epoch + 1}-loss{round(count_loss / step + 1, 4)}-val acc_{round(val_acc, 4)}.pth')
            best_weights = f'{extractor}-Epoch_{epoch + 1}-loss{round(count_loss / step + 1, 4)}-val acc_{round(val_acc, 4)}.pth'
            print(f'save Epoch_{epoch + 1}-loss{round(count_loss / step + 1, 4)}-val acc_{round(val_acc, 4)}.pth')


def evaluate(extractor, classifier, val_loader):
    extractor.eval()
    classifier.eval()
    device = torch.device('cuda:0')
    count_loss = 0
    relu = nn.ReLU()
    correct = 0
    total = len(val_loader)

    with tqdm(total=len(val_loader),
              desc=f';Iteration :{len(val_loader)}',
              postfix=dict, mininterval=0.5) as train_bar:
        for step, (image, label) in enumerate(val_loader):
            image, label = image.to(device), label.to(device)
            # 测试不用加噪声
            featureVectors = extractor(image)
            # data, labels = addGaussianNoise(featureVectors, label, shuffle=False)
            data = relu(featureVectors)
            out = classifier(data)
            # 计算loss
            predict = out.argmax(dim=1)
            correct += torch.eq(predict, label).float().sum().item()
            train_bar.set_postfix(**{'loss': count_loss / (step + 1)})
            train_bar.update(1)
    return correct / (total * val_loader.batch_size)


if __name__ == '__main__':
    image = torch.randn((32, 3, 28, 28))
    # image_val_ex = torch.randn((1, 3, 224, 224))
    # modelvgg16 = getExtractor('vgg16', True)
    # modelvgg19 = getExtractor('vgg19', True)
    # modelAlex = getExtractor('alexnet', True)
    #
    # modelResNet = getExtractor('resnet18', True)
    # out1 = modelResNet(image)
    # modelResNet = torchvision.models.resnet34(True)
    # modelResNet = torchvision.models.resnet50(True)
    OCCNN_train(trainArgs())
    out = image.view(32, 20)
    print(out.shape)
