import torch
import torch.nn as nn


class ValidatorNet(nn.Module):
    def __init__(self, input_dim):
        super(ValidatorNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)

    # def __init__(self, input_size, hidden_size=128):
    #     super(ValidatorNet, self).__init__()
    #     self.net = nn.Sequential(
    #         nn.Linear(input_size * 2, hidden_size),
    #         nn.ReLU(),
    #         nn.Linear(hidden_size, 1),
    #         nn.Sigmoid(),  # Outputs probability that A is better than B
    #     )

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    # def forward(self, a_preds, b_preds):
    #     # a_preds and b_preds are (batch_size, seq_len, vocab_size)
    #     # Flatten and concatenate
    #     a_flat = a_preds.view(a_preds.size(0), -1)
    #     b_flat = b_preds.view(b_preds.size(0), -1)
    #     x = torch.cat([a_flat, b_flat], dim=1)
    #     return self.net(x)
