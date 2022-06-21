import torch
import torch.nn as nn
import torch.nn.functional as F
from biotorch.layers.fa import Conv2d, Linear


class Discriminator(nn.Module):
    '''
        Discriminator model
    '''
    def __init__(self, in_features, out_features, mode, sig=False):
        super(Discriminator, self).__init__()

        #torch.manual_seed(3)

        self.hidden_0 = nn.Sequential(
            nn.Linear(in_features + mode, 1024),
            nn.LeakyReLU(0.2) if not sig else nn.Sigmoid(),
            nn.Dropout(0.5),
        )
        self.hidden_1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2) if not sig else nn.Sigmoid(),
            nn.Dropout(0.4)
        )
        self.hidden_2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2) if not sig else nn.Sigmoid(),
            nn.Dropout(0.4)
        )

        self.out = nn.Sequential(
            torch.nn.Linear(256, out_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden_0(x)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.out(x)
        return x

