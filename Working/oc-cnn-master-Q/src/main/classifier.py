import torch.nn as nn


class classifier_nn(nn.Module):

    def __init__(self, Dimension):
        super(classifier_nn, self).__init__()
        self.fc1 = nn.Linear(Dimension, 2)

    def forward(self, x):
        return self.fc1(x)
