import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import pandas as pd


class CardioDataset(Dataset):
    def __init__(self, csv_path, train=True, transform=None):
        super().__init__()
        self.path = "/raid/data/ibespalov/CHAZOV_dataset/"
        self.csv = pd.read_csv(csv_path)
        if train:
            self.csv = self.csv[self.csv["fold"] != 5]
        else:
            self.csv = self.csv[self.csv["fold"] == 5]
        self.transform = transform

    def __getitem__(self, index):
        path_to_image = self.csv.iloc[index]['images']
        path_to_keypts = self.csv.iloc[index]['keypts']
        img = np.array(Image.open(os.path.join(self.path, path_to_image + ".tiff")))[:, :, 0:3] #
        keypts = np.load(os.path.join(self.path, path_to_keypts + ".npy")) # numpy - [1, 200, 2]
        keypts = [keypts[0, :, 0], keypts[0, :, 1], 0, 1]
        dict_transforms = self.transform(image=img, keypoints=[keypts])
        image, keypts = dict_transforms["image"], dict_transforms["keypoints"]

        kp_x, kp_y = keypts[0][0] / image.shape[-1], keypts[0][1] / image.shape[-1]
        keypoints_new = torch.cat([torch.tensor(kp_x)[..., None], torch.tensor(kp_y)[..., None]], dim=1)

        return {"image": image, "keypoints": keypoints_new}

    def __len__(self):
        return len(self.csv)
