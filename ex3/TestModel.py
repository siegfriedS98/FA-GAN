import torch
import torch.nn as nn
import torch.nn.functional as F
from biotorch.layers.fa import Conv2d, Linear


class TestModel(nn.Module):
    '''
        TestModel
    '''
    def __init__(self, in_features, out_features):
        torch.manual_seed(3)
        super(TestModel, self).__init__()
        self.hidden_0 = nn.Sequential(Linear(in_features=in_features, out_features=400), nn.Tanh())
        self.hidden_1 = nn.Sequential(Linear(in_features=400, out_features=400), nn.Tanh())
        self.hidden_2 = nn.Sequential(Linear(in_features=400, out_features=400), nn.Tanh())
        #self.out = nn.Sequential(Linear(in_features=256, out_features=out_features), nn.Sigmoid())
        self.out = nn.Sequential(Linear(in_features=400, out_features=out_features), nn.Sigmoid())



    def forward(self, x):
        x = self.hidden_0(x)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.out(x)
        return x

