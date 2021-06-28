import numpy as np
import ot
import pykeops
from geomloss import SamplesLoss
from geomloss.sinkhorn_samples import cost_routines, cost_formulas
from geomloss.utils import distances
from scipy import stats
import torch
from torch.distributions import Bernoulli, Uniform
from matplotlib import pyplot as plt
# pykeops.clean_pykeops()

def sample(n, p, b):

    def dist(b, n, p):
        params = torch.ones(b, n, p, device="cuda")
        s = Bernoulli(params * 0.5).sample() - 0.5
        return s

    X = dist(b, n, p)
    X = X.sum(dim=1) / np.sqrt(n)

    X1 = dist(1, 1000000, p)
    V = X1[0].t().mm(X1[0]) / (1000000 - 1)

    G = torch.distributions.MultivariateNormal(torch.zeros(p, device="cuda"), V)
    Y: torch.Tensor = torch.cat([G.sample()[None, :] for i in range(b)])

    return X.max(-1)[0].cpu().numpy(), Y.max(-1)[0].cpu().numpy()


def ecdf(x):
    x = np.sort(x)

    def result(v):
        return np.searchsorted(x, v, side='right') / x.size

    return result


def Pdiff(x, y):

    x0 = min(x.min(), y.min())
    x1 = max(x.max(), y.max())
    dx = (x1 - x0) / 100

    Fx = ecdf(x)
    Fy = ecdf(y)

    diff = 0

    for i in range(1, 100):
        xi = x0 + dx * i
        diff = max(np.abs(Fx(xi) - Fy(xi)), diff)

    return diff


# for k in [2, 3, 4, 5, 6, 7, 8, 9]:
#     cost_routines[k] = (lambda x,y : distances(x,y)**k)



def maxPdiff(n, p):
    x = np.array([])
    y = np.array([])
    for i in range(30):
        xt, yt = sample(n, p, 30000)
        x = np.concatenate([x, xt])
        y = np.concatenate([y, yt])

    return Pdiff(x, y)

def WL(X, Y, L):

  blur = 0.01
  scaling = 0.99
  loss = SamplesLoss("sinkhorn", blur=blur, scaling=scaling, debias=True, p=L, backend="tensorized")

  n = X.shape[0]

  # HX, _ = np.histogramdd(X, bins=100)
  # HY, _ = np.histogramdd(Y, bins=100)
  #
  # nonzero = np.where(HX > -1)
  # nonzero = np.concatenate([a[::, np.newaxis] for a in nonzero], axis=-1)
  # nonzero = 2 * 3.1622 * (nonzero + 0.5) / 100

  # M = ot.dist(X, Y)
  # M = M ** (L/2)
  p = np.ones(n) / n
  # D = ot.emd2(p, p, M, processes=40, numItermax=100000)
  D = loss.forward(torch.from_numpy(p).type(torch.float32).cuda(),
                   torch.from_numpy(X).type(torch.float32).cuda(),
                   torch.from_numpy(p).type(torch.float32).cuda(),
                   torch.from_numpy(Y).type(torch.float32).cuda()).item()
  return D ** (1/L)

res = []

for k in [25, 50, 75, 100, 150, 200]:

    print("gen max 2")
    w = maxPdiff(100, k)
    res.append(w)

    print(k, w)

print(res)


