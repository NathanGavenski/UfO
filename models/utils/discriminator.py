import torch
from torch import no_grad, Tensor, log
from torch import nn
from torch.nn import functional as F
from benchmark.methods.policies.mlp import MLP, MlpWithAttention


def get_discriminator(disc: str) -> nn.Module:
    if disc == "MlpPolicy":
        return MLP
    elif disc == "MlpWithAttention":
        return MlpWithAttention
    elif disc == "LSTM":
        return DiscLSTM
    else:
        raise Exception(f"{disc} is not a viable network")


def compute_reward(discriminator: nn.Module, signature: Tensor) -> Tensor:
    with no_grad():
        return discriminator(signature)


class DiscLSTM(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, num_layer, output_dim, device):
        super().__init__()

        self.device = device

        self.hidden_dim = hidden_dim
        self.layer_dim = num_layer
        self.output_dim = output_dim
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layer,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, output_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        seq_size: list[int] = 0,
        h0: torch.Tensor = None,
        c0: torch.Tensor = None,
        output_inner: bool = False
    ):  # (batch, seq, features)
        x = x.float()

        # Initialize hidden state with zeros
        if h0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        h0 = h0.to(self.device)

        # Initialize cell state
        if c0 is None:
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        c0 = c0.to(self.device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        b, s, f = out.shape
        out = out.reshape(-1, self.hidden_dim)
        classes = self.fc(out)
        classes = classes.reshape(b, s, self.output_dim)

        if output_inner:
            return classes, (hn, cn)
        return classes
