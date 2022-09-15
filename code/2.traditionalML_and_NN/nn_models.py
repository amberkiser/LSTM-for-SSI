import torch
import torch.nn as nn


class NeuralNetModule(nn.Module):
    def __init__(self, input_size, h1_size, h2_size, output_size):
        super(NeuralNetModule, self).__init__()
        self.input_layer = nn.Linear(input_size, h1_size)
        self.layer1 = nn.Linear(h1_size, h2_size)
        self.layer2 = nn.Linear(h2_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.input_layer(x)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.relu(out)
        out = self.layer2(out)
        return out.squeeze()
