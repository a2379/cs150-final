"""
validator.py defines a simple feedforward neural network (ValidatorNet)
used to distinguish between predictions from two different models. It outputs
a binary classification indicating which model a given sample likely came from.
"""

import torch.nn as nn


# --------------------------- Validator Network Class --------------------------
class ValidatorNet(nn.Module):
    """
    A simple 2-layer feedforward neural network for binary classification.

    ValidatorNet is used in an adversarial validation setup to identify
    whether a given embedding originated from Model A or Model B.
    """

    def __init__(self, input_dim):
        super(ValidatorNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
