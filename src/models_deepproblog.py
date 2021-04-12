import torch
import numpy as np
import torch.nn as nn
from problog.logic import term2list
from torch.autograd import Variable

from Global import SCREEN_HEIGHT, SCREEN_WIDTH
from data_handling import get_dataset

dataset_y = get_dataset(data_dir='../data/sorted_y')
dataset_x = get_dataset(data_dir='../data/sorted_x')


class PONG_Net_Y(nn.Module):
    def __init__(self):
        super(PONG_Net_Y, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1),  # 3 30 16 -> 16 30 16
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*30*16, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, SCREEN_HEIGHT),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16*30*16)
        x = self.classifier(x)
        return x


def neural_predicate_y(network, i):
    """ Function responsible for extracting the data from the data pointer passed in the query.
    The data pointer can either point to:
        *   a dataset
            => in which case, we simply fetch the tensor
        *   a list of values representing an image
            => in which case, we have to transform the list to an array of the correct shape and type
            and then transform this array to a tensor.
    The resulting tensor is then fed to the passed network and the probability distribution of the output is returned.
    The cache of the network is then cleared since a cache lookup is more expensive than a forward-prop."""
    from_dataset = (str(i.functor) == "dataset_id")
    if from_dataset:
        i = int(i.args[0])
        d, _ = dataset_y[i]
    else:
        d = term2list(i)
        d = np.asarray(d, dtype=np.float32)
        d = d.reshape((3, SCREEN_HEIGHT, SCREEN_WIDTH))
        d = torch.from_numpy(d)
    d = Variable(d.unsqueeze(0))
    output = network.net(d)
    network.clear()  # clear cached input query, since lookup is more expensive than propagation through the CNN
    return output.squeeze(0)


class PONG_Net_X(nn.Module):
    def __init__(self):
        super(PONG_Net_X, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1),  # 3 30 16 -> 16 30 16
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*30*16, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, SCREEN_WIDTH-2),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16*30*16)
        x = self.classifier(x)
        return x


def neural_predicate_x(network, i):
    from_dataset = (str(i.functor) == "dataset_id")
    if from_dataset:
        i = int(i.args[0])
        d, _ = dataset_x[i]
    else:
        d = term2list(i)
        d = np.asarray(d, dtype=np.float32)
        d = d.reshape((3, SCREEN_HEIGHT, SCREEN_WIDTH))
        d = torch.from_numpy(d)
    d = Variable(d.unsqueeze(0))
    output = network.net(d)
    network.clear()  # clear cached input query, since lookup is more expensive than propagation through the CNN
    return output.squeeze(0)
