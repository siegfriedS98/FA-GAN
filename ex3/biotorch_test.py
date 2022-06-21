import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.datasets import VisionDataset
from ex3.TestModel import TestModel
import ex3.util as utils
from ex3.constants import USE_SIGMOID
from biotorch.module.biomodule import BioModule


class IdentityTransform(object):
    def __call__(self, tensor):
        return tensor


if __name__ == '__main__':
    transforms1 = transforms.Compose([
        transforms.ToTensor(),
        IdentityTransform() if USE_SIGMOID else transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset: VisionDataset
    num_epochs: int
    dims: int
    dims_total: int

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms1,
    )
    num_epochs = 300
    dims = 28
    dims_total = 28 * 28

    dataloader = DataLoader(dataset, batch_size=64,
                            drop_last=True,
                            shuffle=True)

    device = torch.device("cpu")

    model = BioModule(TestModel(in_features=28*28, out_features=10), mode='fa', output_dim=10)

    # Optimizers and loss
    lr = 0.00002
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    #optimizer = optim.RMSprop(model.parameters())
    criterion = nn.BCELoss()

    print("Starting Training...")

    for epoch in range(1, num_epochs + 1):
        for batch_idx, (data, targets) in enumerate(dataloader):
            data = data.to(device)
            batch_size = data.shape[0]
            true_label = torch.zeros(64, 10)
            for i in range(64):
                target = targets[i]
                true_label[i][target] = 1

            model.zero_grad()
            predicted = model(data.view(data.shape[0], -1), targets=true_label, loss_function=criterion).to(device)
            loss = criterion(predicted, true_label).to(device)
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch}/{num_epochs}]  Batch {batch_idx + 1}/{len(dataloader)} \
                            Loss : {loss:.4f}')

        total = 0
        right = 0
        for batch_idx, (data, targets) in enumerate(dataloader):
            data = data.to(device)
            true_label = torch.zeros(64, 10)
            for i in range(64):
                target = targets[i]
                true_label[i][target] = 1
            predicted = model(data.view(data.shape[0], -1), targets=true_label, loss_function=criterion, ).to(device)
            for j in range(64):
                prediction = torch.argmax(predicted[j])
                actual = targets[j]
                if actual == prediction:
                    right += 1
                total += 1

        print("Accuracy: " + str(right/total))




