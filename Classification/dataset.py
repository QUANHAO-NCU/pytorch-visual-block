import csv
import glob
import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class dataset(Dataset):
    def __init__(self, path, resize, mode, train=0.6, val=0.2, test=0.2):
        super(dataset, self).__init__()
        self.path = path
        self.resize = resize
        self.name2label = {}  # 类别编码
        self.name = []
        for name in sorted(os.listdir(path)):
            if not os.path.isdir(os.path.join(path, name)):
                continue
            self.name2label[name] = len(self.name2label.keys())
            self.name.append(name)
        self.class_num = len(self.name2label)
        # image, tips_label
        self.images, self.labels = self.load_csv('images.csv')
        assert (train + val + test) == 1
        if mode == 'train':
            self.images = self.images[:int(train * len(self.images))]
            self.labels = self.labels[:int(train * len(self.labels))]
        elif mode == 'val':
            self.images = self.images[int(train * len(self.images)):int((train + val) * len(self.images))]
            self.labels = self.labels[int(train * len(self.labels)):int((train + val) * len(self.labels))]
        else:  # mode == 'test'
            self.images = self.images[int((1 - (train + val)) * len(self.images)):]
            self.labels = self.labels[int((1 - (train + val)) * len(self.labels)):]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        transform = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),  # 打开图片
            transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),  # 后期会进行中心裁剪，所以图片先resize到1.25倍
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img = transform(img)
        label = torch.tensor(label)
        return img, label

    def shuffle_img_label(self):
        img_label = {}
        for index, img_path in enumerate(self.images):
            img_label[img_path] = self.labels[index]

        random.shuffle(self.images)
        match_label = []
        for img_path in self.images:
            match_label.append(img_label[img_path])
        self.labels = match_label

    def load_csv(self, filename):
        images = []
        for name in self.name2label:
            images += glob.glob(os.path.join(self.path, name, '*.png'))
            images += glob.glob(os.path.join(self.path, name, '*.jpg'))
            images += glob.glob(os.path.join(self.path, name, '*.jpeg'))
        random.shuffle(images)
        with open(os.path.join(self.path, filename), mode='w', newline='') as f:
            writer = csv.writer(f)
            for img_path in images:
                name = img_path.split(os.sep)[-2]
                label = self.name2label[name]
                writer.writerow([img_path, label])
            print('writen into csv file:', filename)
        # read from csv file

        images, labels = [], []
        with open(os.path.join(self.path, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img_path, label = row
                label = int(label)
                images.append(img_path)
                labels.append(label)
        assert len(images) == len(labels)
        return images, labels


    def denormalize(self, x_hat):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x = x_hat * std + mean
        return x


    def get_item_with_path(self, path):
        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),  # string path= > image data
            transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img = tf(path)
        return img
