import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import pandas as pd

from dataset.toheatmap import ToGaussHeatMap, heatmap_to_measure


class CardioDataset(Dataset):
    def __init__(self, csv_path, train=True, transform=None):
        super().__init__()
        self.path = "/raid/data/ibespalov/CHAZOV_dataset/"
        self.csv = pd.read_csv(csv_path)
        if train:
            self.csv = self.csv[self.csv["fold"] != 4]
        else:
            self.csv = self.csv[self.csv["fold"] == 4]
        self.transform = transform

    def __getitem__(self, index):
        path_to_image = self.csv.iloc[index]['images']
        path_to_keypts = self.csv.iloc[index]['keypts']
        img = np.array(Image.open(os.path.join(self.path, path_to_image + ".tiff")))[:, :, 0:3] #
        keypts = np.load(os.path.join(self.path, path_to_keypts + ".npy")) # numpy - [1, 200, 2]
        keypts = keypts[0]
        # keypts = keypts[np.lexsort((keypts[:, 0], keypts[:, 1]))]
        keypts = [keypts[:, 0], keypts[:, 1], 0, 1]
        dict_transforms = self.transform(image=img, keypoints=[keypts])
        image, keypts = dict_transforms["image"], dict_transforms["keypoints"]

        kp_x, kp_y = keypts[0][0] / image.shape[-1], keypts[0][1] / image.shape[-1]
        keypoints_new = torch.cat([torch.tensor(kp_x)[..., None], torch.tensor(kp_y)[..., None]], dim=1)

        return {"image": image, "keypoints": keypoints_new}

    def __len__(self):
        return len(self.csv)


LM_EXTENSIONS = [".npy"]

def is_lm_file(filename):
    return any(filename.endswith(extension) for extension in LM_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_lm_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class LandmarksDataset(Dataset):
    def __init__(self, path: str, transform=None):
        super().__init__()
        self.path = path
        self.transform = transform
        self.data = make_dataset(path)

    def __getitem__(self, index):
        keypts = np.load(self.data[index]) # numpy - [200, 2]
        # keypts = keypts[np.lexsort((keypts[:, 0], keypts[:, 1]))]
        keypts = [keypts[:, 0], keypts[:, 1], 0, 1]
        dict_transforms = self.transform(image=np.zeros((256, 256, 3)), keypoints=[keypts])
        keypts = dict_transforms["keypoints"]
        kp_x, kp_y = keypts[0][0], keypts[0][1]
        keypts_new = torch.cat([
            torch.tensor(kp_x, dtype=torch.float32)[..., None],
            torch.tensor(kp_y, dtype=torch.float32)[..., None]],
            dim=1
        )
        return keypts_new  #torch.tensor - [200, 2]

    def __len__(self):
        return len(self.data)


class LandmarksDatasetAugment(Dataset):
    def __init__(self, path: str, transform=None):
        super().__init__()
        self.path = path
        self.transform = transform
        self.data = make_dataset(path)
        self.heatmapper = ToGaussHeatMap(256, 1)

    def __getitem__(self, index):
        keypts = np.load(self.data[index]) # numpy - [200, 2]
        hm = self.heatmapper.forward(torch.tensor(keypts)[None,])[0]
        transformed = self.transform(
            image=np.zeros_like(np.array(hm.permute(1, 2, 0))),
            mask=np.array(hm.permute(1, 2, 0))
        )
        mask = transformed["mask"]
        coord, p = heatmap_to_measure(mask.permute(2, 0, 1)[None])
        return coord[0].type(torch.float32)  #torch.tensor - [200, 2]

    def __len__(self):
        return len(self.data)

