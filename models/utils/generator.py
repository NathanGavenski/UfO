from torch import nn
from benchmark.methods.policies.mlp import MLP, MlpWithAttention


def get_generator(disc: str) -> nn.Module:
    if disc == "MlpPolicy":
        return MLP
    elif disc == "MlpWithAttention":
        return MlpWithAttention
    else:
        raise Exception(f"{disc} is not a viable network")
