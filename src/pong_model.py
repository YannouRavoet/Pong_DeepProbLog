import torch
import numpy as np
import torch.nn as nn
from problog.logic import term2list
from torch.autograd import Variable
from generate_data import get_dataset

dataset = get_dataset()


class PONG_Net(nn.Module):
    def __init__(self, N=28):
        super(PONG_Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3), # 3 28 28 -> 6 26 26
            nn.MaxPool2d(2, 2),  # 6 26 26 -> 6 13 13
            nn.ReLU(True),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=2),  # 6 13 13 -> 16 12 12
            nn.MaxPool2d(2, 2),  # 16 12 12 -> 16 6 6
            nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),  # 16 6 6 -> 32 4 4
            nn.MaxPool2d(2, 2),  # 32 4 4 -> 32 2 2
            nn.ReLU(True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 2 * 2, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, N),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 32 * 2 * 2)
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
    The resulting tensor is then fed to the passed network and the probability distribution of the output is returned."""
    from_dataset = (str(i.functor) == "dataset_id")
    if from_dataset:
        i = int(i.args[0])
        d, _ = dataset[i]
    else:
        d = term2list(i)
        d = np.asarray(d, dtype=np.float32)
        d = d.reshape((3, 28, 28))
        d = torch.from_numpy(d)
    d = Variable(d.unsqueeze(0))
    output = network.net(d)
    return output.squeeze(0)
