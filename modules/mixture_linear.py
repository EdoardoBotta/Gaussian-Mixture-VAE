import torch
import torch.nn as nn

class MixtureLinear(nn.Module):
    def __init__(self, in_features, out_features, mixture_components):
        super(MixtureLinear, self).__init__()

        empty = torch.empty(mixture_components, in_features, out_features)
        self.Pi = nn.Parameter(nn.init.kaiming_normal_(empty))

    def forward(self, x, z, return_Pi=False):
        #Pi = torch.einsum("bcd,eb->ecd", self.Pi, z)
        #mu = torch.einsum("bij,bj->bi", Pi, x)
        x_next = torch.einsum("bcd,eb,ed->ec", self.Pi, z, x)
        if return_Pi:
            Pi = torch.einsum("bcd,eb->ecd", self.Pi, z)
            return x_next, Pi
        return x_next