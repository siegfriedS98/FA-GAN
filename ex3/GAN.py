from ex3.Generator import Generator
from ex3.Discriminator import Discriminator
import torch
import torch.nn as nn


class GAN(nn.Module):

    def __init__(self, in_features_g, out_features_g, in_features_d, out_features_d, mode, discriminator, generator):
        super().__init__()

        torch.manual_seed(3)
        self.generator = generator

        self.discriminator = discriminator

    def forward(self, x):
        if x.size()[1] == 100:
            x = self.generator(x)
        x = self.discriminator(x)
        return x
