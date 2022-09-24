import torch.nn as nn


class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions.
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            # nn.ReLU(),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            # nn.ReLU(),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)  # (B, 1) -> (B)
        return x
