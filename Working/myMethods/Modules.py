from torch import nn


class Encoder(nn.Module):
    def __init__(self, out_dimension: int):
        super(Encoder, self).__init__()

    def forward(self, x):
        pass


class Decoder(nn.Module):
    def __init__(self, in_dimension: int):
        super(Decoder, self).__init__()

    def forward(self, x):
        pass


class Classifier(nn.Module):
    def __init__(self, in_dimension: int, out_dimension: int):
        super(Classifier, self).__init__()
        self.in_dim = in_dimension
        self.out_dim = out_dimension
        assert self.in_dim > 512
        self.process = nn.Sequential(
            nn.InstanceNorm1d(self.in_dim),
            nn.Linear(self.in_dim, 512),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, self.out_dim)
        )

    def forward(self, x):
        return self.process(x)
