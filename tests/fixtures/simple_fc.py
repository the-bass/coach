import torch.nn as nn
from torch_state_control.nn import StatefulModule


class SimpleFC(StatefulModule):

    def __init__(self, name, directory):
        super().__init__(name, directory)

        self.layer = nn.Linear(
            in_features=14,
            out_features=1,
            bias=True
        )

    def forward(self, input):
        return self.layer(input)
