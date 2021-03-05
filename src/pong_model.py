import torch
import numpy as np
import torch.nn as nn
from problog.logic import term2list
from torch.autograd import Variable

from Global import SCREEN_HEIGHT, SCREEN_WIDTH
from data_handling import get_dataset

dataset = get_dataset()
OUTPUTS = SCREEN_HEIGHT  # outputs = ball positions on y-axis


class PONG_Net(nn.Module):
    def __init__(self):
        super(PONG_Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=1),  # 3 30 16 -> 8 30 16
            nn.MaxPool2d(2, 2),  # 8 30 16 -> 8 15 8
            nn.ReLU(True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2),  # 8 15 8 -> 16 14 7
            nn.MaxPool2d(2, 2, ceil_mode=True),  # 16 14 7 -> 16 7 4
            nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2),  # 16 7 4 -> 32 6 3
            nn.MaxPool2d(2, 2, ceil_mode=True),  # 32 6 3 -> 32 3 2
            nn.ReLU(True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 3 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, OUTPUTS),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 32 * 3 * 2)
        x = self.classifier(x)
        return x


def neural_predicate(network, i):
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
        d, _ = dataset[i]
    else:
        d = term2list(i)
        d = np.asarray(d, dtype=np.float32)
        d = d.reshape((3, SCREEN_HEIGHT, SCREEN_WIDTH))
        d = torch.from_numpy(d)
    d = Variable(d.unsqueeze(0))
    output = network.net(d)
    network.clear()  # clear cached input query, since lookup is more expensive than propagation through the CNN
    return output.squeeze(0)
