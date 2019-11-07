import torch


class SegmentConsensus(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.tensor = x
        output = x.mean(dim=1, keepdim=True)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        shape = ctx.tensor.size()
        grad_in = grad_output.expand(shape) / float(shape[1])

        return grad_in


class ConsensusModule(torch.nn.Module):
    def forward(self, input):
        return SegmentConsensus.apply(input)
