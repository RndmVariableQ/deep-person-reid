from __future__ import absolute_import

import torch
import torch.nn.functional as F
from torch import nn, autograd


class OIM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, lut, momentum=0.5):
        ctx.lut = lut
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(lut.t())
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        # grad_inputs = None
        # if self.needs_input_grad[0]:
        grad_inputs = grad_outputs.mm(ctx.lut)
        for x, y in zip(inputs, targets):
            ctx.lut[y] = ctx.momentum * ctx.lut[y] + (1. - ctx.momentum) * x
            ctx.lut[y] /= ctx.lut[y].norm()
        return grad_inputs, None, None, None


# class OIM(autograd.Function):
#     def __init__(self, lut, momentum=0.5):
#         super(OIM, self).__init__()
#         self.lut = lut
#         self.momentum = momentum
#
#     @staticmethod
#     def forward(self, inputs, targets):
#         self.save_for_backward(inputs, targets)
#         outputs = inputs.mm(self.lut.t())
#         return outputs
#
#     @staticmethod
#     def backward(self, grad_outputs):
#         inputs, targets = self.saved_tensors
#         grad_inputs = None
#         if self.needs_input_grad[0]:
#             grad_inputs = grad_outputs.mm(self.lut)
#         for x, y in zip(inputs, targets):
#             self.lut[y] = self.momentum * self.lut[y] + (1. - self.momentum) * x
#             self.lut[y] /= self.lut[y].norm()
#         return grad_inputs, None


def oim(inputs, targets, lut, momentum=0.5):
    return OIM.apply(inputs, targets, lut, momentum)


class OIMLoss(nn.Module):
    def __init__(self, num_features, num_classes, scalar=1.0, momentum=0.5,
                 weight=None, size_average=True):
        super(OIMLoss, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.momentum = momentum
        self.scalar = scalar
        self.weight = weight
        self.size_average = size_average

        self.register_buffer('lut', torch.zeros(num_classes, num_features))

    def forward(self, inputs, targets):
        inputs = oim(inputs, targets, self.lut, momentum=self.momentum)
        inputs *= self.scalar
        loss = F.cross_entropy(inputs, targets, weight=self.weight,
                               reduction='mean')
        return loss, inputs


if __name__ == '__main__':
    criterion_oim = OIMLoss(256, 768,
                            scalar=30, momentum=0.5)
    feat = torch.rand(5, 256, requires_grad=True)
    l, i = criterion_oim(inputs=feat, targets=torch.rand(5).type(torch.long))
    l.backward()
    print(feat, l)
