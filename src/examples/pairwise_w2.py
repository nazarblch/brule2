

from dataset.d300w import ThreeHundredW

import torch
from dataset.probmeasure import ProbabilityMeasureFabric, ProbabilityMeasure, UniformMeasure2D01
from dataset.toheatmap import heatmap_to_measure
from loss.weighted_was import WasLoss

from parameters.path import Paths

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.set_device(device)

from torch.utils.data import Subset

image_size = 256
batch_size = 1
padding = 68

# Download 300w from https://ibug.doc.ic.ac.uk/resources/300-W/
dataset_train = Subset(ThreeHundredW(f"{Paths.default.data()}/300w", train=True, imwidth=500, crop=15), range(100))
file1 = open('w300graph100.txt', 'w')

for i in range(len(dataset_train)):
    for j in range(len(dataset_train)):

        if i == j:
            continue

        landmarks = dataset_train[i]["meta"]["keypts_normalized"].cuda().view(1, 68, 2)
        mes1 = UniformMeasure2D01(torch.clamp(landmarks, max=1))

        landmarks = dataset_train[j]["meta"]["keypts_normalized"].cuda().view(1, 68, 2)
        mes2 = UniformMeasure2D01(torch.clamp(landmarks, max=1))

        w2 = WasLoss(scaling=0.98, blur=0.0000001)(mes1, mes2)

        print(i, j, w2.item())
        file1.write(f"{i},{j},{w2.item()}\n")


file1.close()




