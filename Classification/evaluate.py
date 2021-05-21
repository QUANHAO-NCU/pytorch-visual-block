import warnings

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import transforms

from dataset import dataset
from dods_cats_classification.model.RestNet18 import ResNet18


def denormalize(x_hat):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
    std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
    x = x_hat * std + mean
    return x


warnings.filterwarnings("ignore")

device = torch.device('cuda:0')
weights_path = 'RestNet18-Epoch_50-loss1.0383-val acc_0.8712.pth'
model = ResNet18(5).to(device)
print('load weights...')
model.load_state_dict(torch.load(weights_path))

dataset_path = 'pokemon_5'
data = dataset(dataset_path, 224, 'test')

tf = transforms.Compose([
    lambda x: Image.open(x).convert('RGB'),
    transforms.Resize((int(data.resize * 1.25), int(data.resize * 1.25))),
    transforms.RandomRotation(15),
    transforms.CenterCrop(data.resize),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
plt.axis('off')
while True:
    img = input('输入图像路径:')
    try:
        image = tf(img).cuda()
        image = torch.unsqueeze(image, dim=0)
    except:
        print('打开失败!')
        continue
    else:
        logics = model(image)
        predict = logics.argmax(dim=1)
        image = torch.squeeze(image, dim=0)
        image = denormalize(image.cpu())
        image = image.detach().numpy()
        image = np.transpose(image, [1, 2, 0])
        plt.imshow(image)
        class_name = data.name[predict[0]]
        confidence = torch.softmax(logics[0], dim=0)[predict[0]]
        plt.text(0, -10, f'class {class_name},confidence{confidence}',fontsize=18)
        plt.show()
