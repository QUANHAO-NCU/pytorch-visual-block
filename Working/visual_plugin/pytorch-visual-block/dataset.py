from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms


class myDataLoader():
    def __init__(self):
        super(myDataLoader, self).__init__()
        self.train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]), download=True)
        self.test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]), download=True)

    def get_data_set(self, mode='train', batch_size=64, num_workers=1, shuffle=True):
        global data_loader
        if mode == 'train':
            data_loader = DataLoader(self.train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        elif mode == 'val':
            data_loader = DataLoader(self.val, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        elif mode == 'test':
            data_loader = DataLoader(self.test, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return data_loader
