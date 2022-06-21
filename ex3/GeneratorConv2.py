import torch.nn as nn


class GeneratorConv2(nn.Module):

    def __init__(self, input_dim=100, output_dim=2, input_size=28):
        super(GeneratorConv2, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.UpsamplingBilinear2d(size=(28, 28)),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(size=(28*2, 28*2)),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, self.output_dim, kernel_size=4, stride=2, padding=0),
            nn.Tanh(),
        )
        pass

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        #print("Generator1: " + str(x.shape))
        x = self.deconv(x)
        #print("Generator2: " + str(x.shape))

        return x
