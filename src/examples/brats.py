import torch
from torch.utils import data

from barycenters.image import createK, barycenter_debiased_2d, ot_dist, w2_dist
from dataset.brats import FilterDataset, BraTS2D
from dataset.lazy_loader import BraTSLoader, data_sampler, sample_data
from matplotlib import pyplot as plt
from skimage import draw

loader = BraTSLoader()
tt, seg = next(loader.loader_train_inf)

P = seg[:, 0].cuda()

P = P / P.sum(dim=[1, 2], keepdim=True)


K = createK(256, 0.002)

print("tut")

b = barycenter_debiased_2d(P, K, tol=1e-6, maxiter=500)
b = (b > b.mean()).float()
b = b / b.sum(dim=[0, 1], keepdim=True)


P1 = torch.cat([b[None,]]).sum(0)

plt.imshow(P1.cpu().numpy())
plt.show()