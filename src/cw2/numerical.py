import torch
import torch.nn as nn

from torch.nn import Tanh
from torch.autograd import Function


class ParamTanh(nn.Module):
    def __init__(self, delta1=0.0, delta2=0.0, delta3=0.0, delta4=0.0):
        super(ParamTanh, self).__init__()
        self.delta1 = delta1
        self.delta2 = delta2
        self.delta3 = delta3
        self.delta4 = delta4

        self.C1 = 2.0
        self.C2 = 1.0
        self.C3 = 2.0
        self.C4 = 1.0

    def forward(self, x: torch.Tensor):
        c1 = self.C1 + self.delta1
        c2 = self.C2 + self.delta2
        c3 = self.C3 + self.delta3
        c4 = self.C4 + self.delta4

        out = (c1 / (c2 + torch.exp(-c3*x))) - c4
        return out


def n_tanh(x: torch.Tensor, approx_order=20):
    coeffs_from_mathematica = [
        0., 1., 0., -1/3, 0., 2/15, 0., -17/315, 0., 62/2835,
        0., -1382/155925, 0., 21844/6081075, 0., -929569/638512875,
        0., 6404582/10854718875, 0., -443861162/1856156927625
    ]
    terms = [
        coeff * x**i
        for i, coeff in enumerate(coeffs_from_mathematica)
        if coeff != 0 and i < approx_order
    ]
    return sum(terms)


class TanhFunction(Function):
    @staticmethod
    # ctx is the first argument to forward
    def forward(ctx, input):
        # The forward pass can use ctx.
        ctx.save_for_backward(input)
        output = n_tanh(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        pass


class NTanh(nn.Module):
    def __init__(self):
        super(NTanh, self).__init__()

    def forward(self, input: torch.Tensor):
        out = TanhFunction.apply(input.clamp(-3, 3))
        return out


if __name__ == "__main__":

    torch.manual_seed(1137)

    x = torch.ones((3,))

    numerical = n_tanh(x)
    pytorch = Tanh()(x)
    param=ParamTanh(0.1,0.1,0.1,0.1)(x)
    print(f'Taylor: {numerical}')
    print(f'nn: {pytorch}')
    print(f'param: {param}')
