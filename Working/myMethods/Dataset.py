from torch.utils.data import Dataset


class myDataset(Dataset):
    def __init__(self):
        super(myDataset, self).__init__()

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass
