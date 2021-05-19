import os.path
import torch
from torch.utils.data import Dataset, Sampler
import nibabel as nib
import numpy as np
from typing import Callable

from dataset.cardio_keypts import make_dataset


class BraTS2D(Dataset):

    def __init__(self, path_dir: str, transform):

        self.transform = transform
        self.step = 5

        directories = [d for d in os.listdir(path_dir) if os.path.isdir(os.path.join(path_dir, d))]
        self.mri_files_t1, self.mri_files_t2, self.seg_files = [], [], []
        for i in directories:
            for s in range(1, 156, 5):
                mri_file_t1 = os.path.join(path_dir, i) + f"/{i}_s{s}_t1.npy"
                mri_file_t2 = os.path.join(path_dir, i) + f"/{i}_s{s}_t2.npy"
                seg_file = os.path.join(path_dir, i) + f"/{i}_s{s}_seg.npy"
                if os.path.isfile(mri_file_t1) and os.path.isfile(mri_file_t2) and os.path.isfile(seg_file):
                    self.mri_files_t1.append(mri_file_t1)
                    self.mri_files_t2.append(mri_file_t2)
                    self.seg_files.append(seg_file)


    def __getitem__(self, index):
        # print(index)
        t1 = np.load(self.mri_files_t1[index])[:, :, np.newaxis]
        t2 = np.load(self.mri_files_t2[index])[:, :, np.newaxis]
        blank = np.zeros_like(t1)
        seg = np.load(self.seg_files[index])[:, :, np.newaxis]

        image = np.concatenate([t1, t2, t1], axis=-1)

        transformed = self.transform(
            image=image,
            mask=seg
        )

        transformed["mask"] = transformed["mask"].permute(2, 0, 1).type(torch.float32)
        transformed["mask"] = (transformed["mask"] > 0).float()
        transformed["image"] = transformed["image"].type(torch.float32)

        return transformed["image"],\
               transformed["mask"]

    def __len__(self):
        return len(self.seg_files)


class BraTS3D(Dataset):

    def __init__(self, path_dir: str, transform):

        self.transform = transform
        self.step = 5

        directories = [d for d in os.listdir(path_dir) if os.path.isdir(os.path.join(path_dir, d))]
        self.mri_files_t1, self.mri_files_t2, self.seg_files = [], [], []
        for i in directories:
            mri_file_t1 = os.path.join(path_dir, i) + f"/{i}_t1.nii"
            mri_file_t2 = os.path.join(path_dir, i) + f"/{i}_t2.nii"
            seg_file = os.path.join(path_dir, i) + f"/{i}_seg.nii"
            if os.path.isfile(mri_file_t1) and os.path.isfile(mri_file_t2) and os.path.isfile(seg_file):
                self.mri_files_t1.append(mri_file_t1)
                self.mri_files_t2.append(mri_file_t2)
                self.seg_files.append(seg_file)


    def __getitem__(self, index):
        # print(index)
        t1 = nib.load(self.mri_files_t1[index]).get_fdata()
        t2 = nib.load(self.mri_files_t2[index]).get_fdata()
        seg = nib.load(self.seg_files[index]).get_fdata()

        mask = np.concatenate([t1, t2, seg], axis=-1)

        transformed = self.transform(
            image=np.zeros_like(mask),
            mask=mask
        )

        K = seg.shape[-1]

        transformed["mask"] = transformed["mask"].permute(2, 0, 1).type(torch.float32)

        return transformed["mask"][0: K: self.step, :, :], \
               transformed["mask"][K: 2*K: self.step, :, :], \
               transformed["mask"][2*K: 3*K: self.step, :, :]

    def __len__(self):
        return len(self.seg_files)


class Dataset3DTo2D(Dataset):

    def __init__(self, dataset: Dataset):

        self.NC = dataset[0][0].shape[0]
        self.dataset = dataset
        self.tmp = None
        self.tmp_index = None

    def fast_index(self, index):
        if index != self.tmp_index:
            self.tmp = self.dataset[index]
            self.tmp_index = index
        return self.tmp

    def __getitem__(self, index):
        # print(index)
        d3_object = self.fast_index(index // self.NC)
        return tuple(el[index % self.NC][None, ] for el in d3_object)

    def __len__(self):
        return self.dataset.__len__() * self.NC


class FilterDataset(Dataset):

    def __init__(self, data_source, condition: Callable, load=False):
        super().__init__()
        self.data_source = data_source
        self.ids = []

        if not load:
            for i in range(len(self.data_source)):
                if condition(self.data_source[i]):
                    print(i)
                    self.ids.append(i)
            self.save()
        else:
            self.load()

    def __getitem__(self, index):
        # print(self.ids[index], self.data_source[self.ids[index]][-1].sum())
        return self.data_source[self.ids[index]]

    def __len__(self):
        return len(self.ids)

    def save(self):
        np.save("ids.npy", np.asarray(self.ids))

    def save_data(self):
        for i in self.ids:
            np.save("ids.npy", np.asarray(self.ids))

    def load(self):
        self.ids = np.load("ids.npy").tolist()


class MasksDataset(Dataset):

    def __init__(self, path: str, transform=None):
        super().__init__()
        self.path = path
        self.transform = transform
        self.data = make_dataset(path)

    def __getitem__(self, index):
        mask = np.load(self.data[index])

        transformed = self.transform(
            image=np.zeros_like(mask),
            mask=mask
        )

        # transformed["mask"] = (transformed["mask"] > 0).float()

        return transformed["mask"].type(torch.float32)

    def __len__(self):
        return len(self.data)






