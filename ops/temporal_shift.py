import torch
import torch.nn as nn


class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div)
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=3):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div

        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift future frames
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift past frames
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)


if __name__ == '__main__':
    tsm = TemporalShift(nn.Sequential(), n_segment=3, n_div=8)

    print('=> Testing GPU...')
    tsm.cuda()
    # test forward
    with torch.no_grad():
        x = torch.rand(3, 8, 1, 1).cuda()
        y = tsm(x)
        print(x)
        print(y)

    print('Test passed.')
