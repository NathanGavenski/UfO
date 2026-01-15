from benchmark.methods.policies import MlpWithAttention
from benchmark.methods.policies import SelfAttn1D
from torch import nn


class Policy(MlpWithAttention):
    def __init__(self, in_dim, out_dim, activation, att) -> None:
        super().__init__(in_dim, out_dim, activation)

        out = max(400, in_dim * 2)

        if att:
            self.layers = nn.Sequential(
                nn.Linear(in_dim, out),
                activation(),

                SelfAttn1D(out),
                nn.Linear(out, out),
                activation(),

                SelfAttn1D(out),
                nn.Linear(out, out),
                activation(),

                nn.Linear(out, out),
                activation(),

                nn.Linear(out, out_dim),
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(in_dim, out),
                activation(),

                nn.Linear(out, out),
                activation(),

                nn.Linear(out, out),
                activation(),

                nn.Linear(out, out),
                activation(),

                nn.Linear(out, out_dim),
            )
