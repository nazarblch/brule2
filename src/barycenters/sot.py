from torch import nn, Tensor
import torch
from loss.weighted_was import PairwiseCost


class SOT(nn.Module):
    def __init__(self, max_iters: int=100, reg=1e-2):
        super(SOT, self).__init__()
        self.reg = reg
        self.max_iters = max_iters
        self.dist = PairwiseCost(lambda t1, t2: (t1 - t2).pow(2).sum(dim=-1).sqrt())

    def sinkhorn(self, a: Tensor, b: Tensor, M: Tensor):

        device = a.device
        # init data
        dim_a = a.shape[1]
        dim_b = b.shape[1]
        batch_size = a.shape[0]
        assert a.shape[0] == b.shape[0]
        a = a.view(batch_size, dim_a, 1).type(torch.float64)
        b = b.view(batch_size, dim_b, 1).type(torch.float64)

        u = torch.ones((batch_size, dim_a, 1), device=device, dtype=torch.float64) / dim_a
        v = torch.ones((batch_size, dim_b, 1), device=device, dtype=torch.float64) / dim_b

        K = torch.exp(-M.type(torch.float64) / self.reg)
        Kt = K.transpose(1, 2)

        cpt = 0

        P = (u.reshape((batch_size, -1, 1)) * K * v.reshape((batch_size, 1, -1)))

        while cpt < self.max_iters:
            uprev = u
            vprev = v

            KtU = torch.bmm(Kt, u)
            v = b / KtU

            KV = K.bmm(v)
            u = a / KV

            if (torch.any(KtU == 0)
                    or torch.any(torch.isnan(u)) or torch.any(torch.isnan(v))
                    or torch.any(torch.isinf(u)) or torch.any(torch.isinf(v))):
                print('Warning: numerical errors at iteration', cpt)
                u = uprev
                v = vprev
                break

            cpt = cpt + 1

            if cpt % 10 == 0:
                P_new = (u.reshape((batch_size, -1, 1)) * K * v.reshape((batch_size, 1, -1)))
                if (P - P_new).abs().max() < 0.00001:
                    P = P_new
                    break
                else:
                    P = P_new

        return P.type(torch.float32)

    def forward(self, x1: Tensor, x2: Tensor):

        if x1.shape.__len__() == 2:
            x1 = x1[None, ]
            x2 = x2[None, ]

        M = self.dist(x1, x2)
        Mnorm = M / M.max(dim=1)[0].max(dim=1)[0].view(-1, 1, 1)

        N = x1.shape[1]
        B = x1.shape[0]
        a = torch.ones(B, N, dtype=torch.float64).cuda()
        a /= a.sum(dim=1, keepdim=True)
        P = self.sinkhorn(a, a, Mnorm)

        return (M * P).sum().item()