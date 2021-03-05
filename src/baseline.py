import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data_handling import get_dataset, split_dataset_pytorch
from logger import Logger

OUTPUTS = 3  # 3 possible actions


class PONG_PytorchNet(nn.Module):
    def __init__(self):
        super(PONG_PytorchNet, self).__init__()
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
            nn.LogSoftmax(1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 32 * 3 * 2)
        x = self.classifier(x)
        return x


def test_pytorch(model, testset):
    confusion = np.zeros((OUTPUTS, OUTPUTS), dtype=np.uint32)  # First index actual, second index predicted
    correct = 0
    n = 0
    for d, l in testset:
        d = Variable(d.unsqueeze(0))
        outputs = model.forward(d)
        _, out = torch.max(outputs.data, 1)
        c = int(out.squeeze())
        confusion[l, c] += 1
        if c == l:
            correct += 1
        n += 1
    acc = correct / n
    print(f'Accuracy: {acc}')
    return acc


def train_pytorch(epochs=1, test_iter=500, log_iter=50):
    BATCH_SIZE = 2
    dataset = get_dataset()
    trainset, testset = split_dataset_pytorch(dataset)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

    model = PONG_PytorchNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()

    step = 0
    test_iter = test_iter / BATCH_SIZE
    log_iter = log_iter / BATCH_SIZE
    running_loss = 0.0
    log = Logger()

    for epoch in range(epochs):
        for data in trainloader:
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if step % log_iter == 0:
                print('Iteration: ', step * BATCH_SIZE, '\tAverage Loss: ', running_loss / log_iter)
                log.log('loss', step * BATCH_SIZE, running_loss / log_iter)
                running_loss = 0
            if step % test_iter == 0:
                Acc = test_pytorch(model, testset)
                log.log('Accuracy', step * BATCH_SIZE, Acc)
            step += 1
    return log
