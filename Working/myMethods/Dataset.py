from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from Tools import *


class myDataset(Dataset):
    def __init__(self, dataset='MNIST', datasetPath='.'):
        super(myDataset, self).__init__()
        self.dataset_train_val = None
        self.dataset_test = None
        self.datasetPath = datasetPath
        self.label_image_train = {}
        self.label_image_val = {}
        self.label_image_test = {}
        self.label2text = {}
        for i in range(10):
            self.label_image_train[i] = []
            self.label_image_val[i] = []
            self.label_image_test[i] = []
        self.images = []
        self.labels = []
        self.datasetMode = 'train'
        self.positive_class = 0
        self.negative_positive_proportion = 1
        if dataset == 'MNIST':
            self.initMNIST(datasetPath)
        elif dataset == 'CIFAR10':
            self.initCIFAR10(datasetPath)

    def __getitem__(self, item):
        if self.datasetMode == 'train' or self.datasetMode == 'val':
            image, _ = self.dataset_train_val[self.images[item]]
        elif self.datasetMode == 'test':
            image, _ = self.dataset_test[self.images[item]]
        label = self.labels[item]
        self.default_tf = transforms.Compose([
            lambda x: x.convert('RGB'),  # 扩展为RGB通道
            transforms.ToTensor(),  # 转换为tensor数据
        ])
        # 基础变换，转换为RGB通道，转换为tensor类型
        sourceImage = self.default_tf(image)
        return sourceImage, label

    def __len__(self):
        assert len(self.images) == len(self.labels)
        return len(self.images)

    def setMode(self, mode, positive_class=0, np_proportion=1):
        """
        negative_positive_proportion :负类：正类比例：1:x --> 1:1~1:9
        mode : train val test
        """
        self.images = []
        self.labels = []
        self.datasetMode = mode
        self.positive_class = positive_class
        if positive_class not in self.label_image_train:
            print('选取的正类不存在')
            exit(-1)
        if self.datasetMode == 'train':
            self.images += self.label_image_train[positive_class]
            self.labels += [1 for _ in self.images]
        elif self.datasetMode == 'val':
            for class_id in self.label_image_val:
                images = self.label_image_val[class_id]
                self.images += [image for image in images]
                if class_id == positive_class:
                    self.labels += [1 for _ in images]
                else:
                    self.labels += [0 for _ in images]
        elif self.datasetMode == 'test':
            if np_proportion > len(self.label_image_test) - 1:
                print('没有这么多负样本')
                exit(-1)
            labels = [positive_class]
            while np_proportion > 0:
                choose = random.sample(list(self.label_image_test), 1)
                if choose[0] != positive_class and choose[0] not in labels:
                    labels.append(choose[0])
                    np_proportion -= 1
            for label in labels:
                images = self.label_image_test[label]
                self.images += [image for image in images]
                if label != positive_class:
                    self.labels += [0 for _ in images]
                else:
                    self.labels += [1 for _ in images]
        elif self.datasetMode == 'normal':
            for label in self.label_image_train:
                images = self.label_image_train[label]
                self.images += [image for image in images]
                self.labels += [label for _ in images]
            for label in self.label_image_val:
                images = self.label_image_val[label]
                self.images += [image for image in images]
                self.labels += [label for _ in images]
            for label in self.label_image_test:
                images = self.label_image_test[label]
                self.images += [image for image in images]
                self.labels += [label for _ in images]

    def initMNIST(self, datasetPath, train_val_proportion=0.9):
        mnist_train = MNIST(root=datasetPath, train=True, download=True)
        mnist_test = MNIST(root=datasetPath, train=False, download=True)
        train_total_num = mnist_train.__len__()
        train_num = int(train_total_num * train_val_proportion)
        for index in range(train_num):
            _, label = mnist_train[index]
            self.label_image_train[label].append(index)
        for index in range(train_num, train_total_num):
            _, label = mnist_train[index]
            self.label_image_val[label].append(index)
        for index in range(len(mnist_test)):
            _, label = mnist_test[index]
            self.label_image_test[label].append(index)
        self.dataset_train_val = mnist_train
        self.dataset_test = mnist_test

    def initCIFAR10(self, datasetPath, train_val_proportion=0.9):
        cifar10_train = CIFAR10(root=datasetPath, train=True, download=True)
        cifar10_test = CIFAR10(root=datasetPath, train=False, download=True)
        train_total_num = cifar10_train.__len__()
        train_num = int(train_total_num * train_val_proportion)
        for index in range(train_num):
            _, label = cifar10_train[index]
            self.label_image_train[label].append(index)
        for index in range(train_num, train_total_num):
            _, label = cifar10_train[index]
            self.label_image_val[label].append(index)
        for index in range(len(cifar10_test)):
            image, label = cifar10_test[index]
            self.label_image_test[label].append(index)
        self.dataset_train_val = cifar10_train
        self.dataset_test = cifar10_test


if __name__ == '__main__':
    dataset = myDataset('CIFAR10', '.')
    dataset.setMode('normal')
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)
    dataset.setMode('val', positive_class=0)
    for image, labels in dataloader:
        print(labels)
    dataset.setMode('test', positive_class=0, np_proportion=2)
    for image, labels in dataloader:
        print(labels)
    pass
