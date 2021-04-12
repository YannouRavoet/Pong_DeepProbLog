import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data_handling import get_dataset, split_dataset_pytorch
from logger import Logger

OUTPUTS = 3  # 3 possible actions


class PONG_PyTorchNet(nn.Module):
    def __init__(self):
        super(PONG_PyTorchNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1),  # 3 30 16 -> 16 30 16
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 30 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, OUTPUTS),
            nn.LogSoftmax(1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * 30 * 16)
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
    BATCH_SIZE = 1
    dataset = get_dataset(data_dir='../data/sorted_pytorch')
    trainset, testset = split_dataset_pytorch(dataset)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

    model = PONG_PyTorchNet()
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
    Acc = test_pytorch(model, testset)
    log.log('Accuracy', step * BATCH_SIZE, Acc)
    return log
