from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, CIFAR10
import csv
import glob
import os
import random
import copy
import torch
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms


class myDataset(Dataset):
    def __init__(self, dataset='MNIST', datasetPath='.'):
        super(myDataset, self).__init__()
        self.dataset = dataset
        self.datasetPath = datasetPath
        self.label_image_train = {}
        self.label_image_val = {}
        self.label_image_test = {}
        self.label2text = {}
        self.transform = False
        self.addGaussianNoise = False
        # 若要添加高斯噪声，指定均值和方差
        self.mean = None
        self.std = None
        for i in range(10):
            self.label_image_train[i] = []
            self.label_image_val[i] = []
            self.label_image_test[i] = []
            self.label2text[i] = ''
        self.images = []
        self.labels = []
        self.datasetMode = 'train'
        self.positive_class = 0
        self.negative_positive_proportion = 1
        if dataset == 'MNIST':
            self.initMNIST(datasetPath)
            self.size = 28
        elif dataset == 'CIFAR10':
            self.initCIFAR10(datasetPath)
            self.size = 32
        self.tf = transforms.Compose([
            lambda x: x.convert('RGB'),  # 扩展为RGB通道
            transforms.Resize((int(self.size * 1.2), int(self.size * 1.2))),  # 后期会进行中心裁剪，所以图片先resize到1.25倍
            transforms.RandomRotation(15),  # 数据增强部分，随机旋转
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),

        ])

    def __getitem__(self, item):
        image = self.images[item]
        label = self.labels[item]
        if self.transform:
            image = self.tf(image)
        if self.addGaussianNoise:
            image = self.addGaussian(image, self.mean, self.std)
        return image, label

    def __len__(self):
        assert len(self.images) == len(self.labels)
        return len(self.images)

    def setMode(self, mode, positive_class=0, negative_positive_proportion=1, transform=False, addGaussianNoise=False,
                mean=0, std=0.01):
        """
        negative_positive_proportion :负类：正类比例：x:1,x--> [1,9]
        """
        self.images = []
        self.labels = []
        self.datasetMode = mode
        self.positive_class = positive_class
        if positive_class not in self.label_image_train:
            print('选取的正类不存在')
            exit(-1)
        if self.datasetMode == 'train':
            self.images = self.label_image_train[positive_class]
            self.labels = [1 for _ in self.images]
        elif self.datasetMode == 'val':
            tmp = []
            for label in self.label_image_val:
                images = self.label_image_val[label]
                if label != positive_class:
                    mark_label = 0
                else:
                    mark_label = 1
                for image in images:
                    tmp.append([image, mark_label])
            random.shuffle(tmp)
            self.images = [item[0] for item in tmp]
            self.labels = [item[1] for item in tmp]
        elif self.datasetMode == 'test':
            if negative_positive_proportion > len(self.label_image_test) - 1:
                print('没有这么多负样本')
                exit(-1)
            tmp = []
            positive_images = self.label_image_test[positive_class]
            for image in positive_images:
                tmp.append([image, 1])
            negative_labels = []
            while negative_positive_proportion > 0:
                choose = random.sample(list(self.label_image_test), 1)
                if choose[0] != positive_class and choose[0] not in negative_labels:
                    negative_labels.append(choose[0])
                    negative_positive_proportion -= 1
            for negative_label in negative_labels:
                images = self.label_image_test[negative_label]
                for image in images:
                    tmp.append([image, 0])
            random.shuffle(tmp)
            self.images = [item[0] for item in tmp]
            self.labels = [item[1] for item in tmp]
        elif self.datasetMode == 'normal':
            for label in self.label_image_train:
                images = self.label_image_train[label]
                for image in images:
                    self.images.append(image)
                    self.labels.append(label)
            for label in self.label_image_val:
                images = self.label_image_val[label]
                for image in images:
                    self.images.append(image)
                    self.labels.append(label)
            for label in self.label_image_test:
                images = self.label_image_test[label]
                for image in images:
                    self.images.append(image)
                    self.labels.append(label)
        self.transform = transform
        self.addGaussianNoise = addGaussianNoise
        self.mean = mean
        self.std = std
        if self.addGaussianNoise:
            assert self.mean is not None and self.std is not None

    def addGaussian(self, image, mean, std):
        img = np.asarray(image)
        noise = np.random.normal(mean, std, img.shape)
        noise = torch.from_numpy(noise)
        image += noise
        return image

    def denormalize(self, image):
        assert torch.is_tensor(image)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x = image * std + mean
        return x

    def initMNIST(self, datasetPath, train_val_proportion=0.9):
        mnist_train = MNIST(root=datasetPath, train=True, download=True)
        mnist_test = MNIST(root=datasetPath, train=False, download=True)
        train_total_num = mnist_train.__len__()
        train_num = int(train_total_num * train_val_proportion)
        for index in range(train_num):
            image, label = mnist_train[index]
            self.label_image_train[label].append(image)
        for index in range(train_num, train_total_num):
            image, label = mnist_train[index]
            self.label_image_val[label].append(image)
        for image, label in mnist_test:
            self.label_image_test[label].append(image)

    def initCIFAR10(self, datasetPath, train_val_proportion=0.9):
        cifar10_train = CIFAR10(root=datasetPath, train=True, download=True)
        cifar10_test = CIFAR10(root=datasetPath, train=False, download=True)
        train_total_num = cifar10_train.__len__()
        train_num = int(train_total_num * train_val_proportion)
        for index in range(train_num):
            image, label = cifar10_train[index]
            self.label_image_train[label].append(image)
        for index in range(train_num, train_total_num):
            image, label = cifar10_train[index]
            self.label_image_val[label].append(image)
        for image, label in cifar10_test:
            self.label_image_test[label].append(image)


if __name__ == '__main__':
    dataset = myDataset('MNIST', '.')
    # dataset.setMode('train', 3)
    # train_dataset = copy.deepcopy(dataset)
    # train_loader = DataLoader(train_dataset)
    # print(train_loader.__len__())
    # dataset.setMode('test', 3, negative_positive_proportion=7)
    # test_dataset = copy.deepcopy(dataset)
    # test_loader = DataLoader(test_dataset)
    # print('test loader:', test_loader.__len__())
    # print('train loader:', train_loader.__len__())
    dataset.setMode('normal', transform=True, addGaussianNoise=True, mean=0, std=1)
    image, label = dataset.__getitem__(28891)
    assert torch.is_tensor(image)
    image = dataset.denormalize(image.clone().detach())
    image = np.transpose(image, (1, 2, 0))
    plt.imshow(image)
    plt.show()
    dataset.setMode('normal', transform=True, addGaussianNoise=False, mean=0, std=0.1)
    image, label = dataset.__getitem__(28891)
    assert torch.is_tensor(image)
    image = dataset.denormalize(image.clone().detach())
    image = np.transpose(image, (1, 2, 0))
    plt.imshow(image)
    plt.show()
