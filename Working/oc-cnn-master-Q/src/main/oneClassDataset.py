import os
import csv
import glob
import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class OneClassDataset(Dataset):
    def __init__(self, path, positiveClass, mode='train', transform=None, resize=28):
        """
        path：数据集的路径
        positiveClass:指定正类
        当mode为train模式时，dataset只会从正类那里加载数据，正类的标签始终为1
        当mode为val或者test模式时，dataset会从目录下随机加载正类与负类，正类标签为1，负类标签为0
        """
        super(OneClassDataset, self).__init__()
        self.path = path
        self.image_label = {}
        self.class_label = {}
        self.images = []
        self.labels = []
        self.classes = []
        self.mode = mode
        self.transform = transform
        self.resize = resize
        self.positiveClass = positiveClass
        # 列出目录下所有类别
        for name in sorted(os.listdir(self.path)):
            if not os.path.isdir(os.path.join(self.path, name)):
                continue
            self.class_label[name] = len(self.class_label.keys())
            self.classes.append(name)
        # 正类为1，其余为负类，置0
        for class_ in self.class_label:
            if class_ != positiveClass:
                self.class_label[class_] = 0
        if positiveClass in self.class_label:
            self.class_label[positiveClass] = 1
        else:
            print('没有找到指定的正类')
            exit(1)
        self.loadImages()
        pass

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        imgPath, label = self.images[idx], self.labels[idx]
        img = Image.open(imgPath).convert('RGB')
        if self.transform:
            img = self.transform(img)
        else:
            tf = transforms.Compose([
                transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
                transforms.RandomRotation(15),
                transforms.CenterCrop(self.resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            img = tf(img)
        label = torch.tensor(label)
        return img, label

    def loadImages(self):
        images = []
        for className in self.class_label:
            images += glob.glob(os.path.join(self.path, className, '*.png'))
            images += glob.glob(os.path.join(self.path, className, '*.jpg'))
            images += glob.glob(os.path.join(self.path, className, '*.jpeg'))
        for img_path in images:
            className = img_path.split(os.sep)[-2]
            label = self.class_label[className]
            self.image_label[img_path] = label
            # 训练模式下只加载正类
            if self.mode == 'train':
                if className == self.positiveClass:
                    self.images.append(img_path)
                    self.labels.append(label)
            else:
                self.images.append(img_path)
                self.labels.append(label)


if __name__ == '__main__':
    dataset = OneClassDataset(r'D:\Code\machineLearning\pyTorch\Dataset\1001 Abnormal Image Dataset', 'Car', mode='val',
                              resize=224)
    img, label = dataset.__getitem__(5)
    print(label)
