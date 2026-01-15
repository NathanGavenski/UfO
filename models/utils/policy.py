from typing import Tuple

from benchmark.methods.policies import MlpWithAttention, MLP
from torch import nn, zeros, Tensor, exp


def get_policy(desc: str) -> nn.Module:
    if desc == "MlpPolicy":
        return Policy
    elif desc == "MlpWithAttention":
        return PolicyWithAttention
    else:
        raise Exception(f"{desc} is not a viable network")


class Policy(MLP):
    def __init__(self, in_dim, out_dim, activation) -> None:
        super().__init__(in_dim, out_dim, activation)
        self.log_stds = nn.Parameter(zeros(1, out_dim))

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        x = x.float()
        mean = self.layers(x)
        std = exp(self.log_stds)
        return mean, std


class PolicyWithAttention(MlpWithAttention):
    def __init__(self, in_dim, out_dim, activation) -> None:
        super().__init__(in_dim, out_dim, activation)
        self.log_stds = nn.Parameter(zeros(1, out_dim))

    def forward(self, x:Tensor) -> Tuple[Tensor]:
        x = x.float()
        mean = self.layers(x)
        std = exp(self.log_stds)
        return mean, std
