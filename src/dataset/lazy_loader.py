from typing import Optional

import albumentations
import torch
from torch import nn, Tensor
from torch.utils import data
from torch.utils.data import DataLoader

from dataset.cardio_dataset import ImageMeasureDataset, ImageDataset, CelebaWithLandmarks
from dataset.cardio_keypts import CardioDataset, LandmarksDataset
from dataset.d300w import ThreeHundredW
from dataset.MAFL import MAFLDataset
from albumentations.pytorch.transforms import ToTensorV2 as AlbToTensor, ToTensorV2

from parameters.path import Paths



def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

class W300DatasetLoader:

    batch_size = 8
    test_batch_size = 16

    def __init__(self):
        self.dataset_train = ThreeHundredW(f"{Paths.default.data()}/300w", train=True, imwidth=500, crop=15)

        self.loader_train = data.DataLoader(
            self.dataset_train,
            batch_size=W300DatasetLoader.batch_size,
            sampler=data_sampler(self.dataset_train, shuffle=True, distributed=False),
            drop_last=True,
            num_workers=20
        )

        self.loader_train_inf = sample_data(self.loader_train)

        self.test_dataset = ThreeHundredW(f"{Paths.default.data()}/300w", train=False, imwidth=500, crop=15)

        self.test_loader = data.DataLoader(
            self.test_dataset,
            batch_size=W300DatasetLoader.test_batch_size,
            drop_last=False,
            num_workers=20
        )

        print("300 W initialize")
        print(f"train size: {len(self.dataset_train)}, test size: {len(self.test_dataset)}")

        self.test_loader_inf = sample_data(self.test_loader)


class CelebaWithKeyPoints:

    image_size = 256
    batch_size = 8

    @staticmethod
    def transform():
        return albumentations.Compose([
            albumentations.HorizontalFlip(),
            albumentations.Resize(CelebaWithKeyPoints.image_size, CelebaWithKeyPoints.image_size),
            # albumentations.ElasticTransform(p=0.5, alpha=100, alpha_affine=1, sigma=10),
            albumentations.ShiftScaleRotate(p=0.5, rotate_limit=10),
            albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            AlbToTensor()
        ])

    def __init__(self):

        print("init calaba with masks")

        dataset = ImageMeasureDataset(
            f"{Paths.default.data()}/celeba",
            f"{Paths.default.data()}/celeba_masks",
            img_transform=CelebaWithKeyPoints.transform()
        )

        print("dataset size: ", len(dataset))

        self.loader = data.DataLoader(
            dataset,
            batch_size=CelebaWithKeyPoints.batch_size,
            sampler=data_sampler(dataset, shuffle=True, distributed=False),
            drop_last=True,
            num_workers=20
        )

        print("batch size: ", CelebaWithKeyPoints.batch_size)

        self.loader = sample_data(self.loader)


class Celeba:

    image_size = 256
    batch_size = 8

    transform = albumentations.Compose([
            albumentations.HorizontalFlip(),
            albumentations.Resize(CelebaWithKeyPoints.image_size, CelebaWithKeyPoints.image_size),
            # albumentations.ElasticTransform(p=0.5, alpha=50, alpha_affine=1, sigma=10),
            albumentations.ShiftScaleRotate(p=0.5, rotate_limit=10, scale_limit=(-0.1, 0.3)),
            albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            AlbToTensor()
    ])

    def __init__(self):
        print("init calaba")

        dataset = ImageDataset(
            f"{Paths.default.data()}/celeba",
            img_transform=Celeba.transform
        )

        print("dataset size: ", len(dataset))

        self.loader = data.DataLoader(
            dataset,
            batch_size=Celeba.batch_size,
            sampler=data_sampler(dataset, shuffle=True, distributed=False),
            drop_last=True,
            num_workers=Celeba.batch_size
        )

        print("batch size: ", Celeba.batch_size)

        self.loader = sample_data(self.loader)


class Cardio:

    image_size = 256
    batch_size = 4
    test_batch_size = 4

    transforms = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])

    def __init__(self):
        path = "/raid/data/ibespalov/CHAZOV_dataset/folds4chamb.csv"
        self.dataset_train = CardioDataset(path, train=True, transform=Cardio.transforms)

        self.loader_train = data.DataLoader(
            self.dataset_train,
            batch_size=Cardio.batch_size,
            sampler=data_sampler(self.dataset_train, shuffle=True, distributed=False),
            drop_last=True,
            num_workers=20
        )

        self.loader_train_inf = sample_data(self.loader_train)

        self.test_dataset = CardioDataset(path, train=False, transform=Cardio.transforms)

        print("train:", self.dataset_train.__len__())
        print("test:", self.test_dataset.__len__())

        self.test_loader = data.DataLoader(
            self.test_dataset,
            batch_size=Cardio.test_batch_size,
            drop_last=False,
            num_workers=20
        )


class CardioLandmarks:

    batch_size = 4

    transforms = albumentations.Compose([
        ToTensorV2()
    ])

    def __init__(self):
        path = "/raid/data/cardio/lm"
        self.dataset_train = LandmarksDataset(path, transform=CardioLandmarks.transforms)

        self.loader_train = data.DataLoader(
            self.dataset_train,
            batch_size=CardioLandmarks.batch_size,
            sampler=data_sampler(self.dataset_train, shuffle=True, distributed=False),
            drop_last=True,
            num_workers=20
        )

        self.loader_train_inf = sample_data(self.loader_train)

        print("CardioLandmarks")
        print("train:", self.dataset_train.__len__())


class W300Landmarks:

    batch_size = 4

    transforms = albumentations.Compose([
        ToTensorV2()
    ])

    def __init__(self, N = 3148):
        path = f"{Paths.default.data()}/w300_bc_{N}"
        self.dataset_train = LandmarksDataset(path, transform=W300Landmarks.transforms)

        self.loader_train = data.DataLoader(
            self.dataset_train,
            batch_size=W300Landmarks.batch_size,
            sampler=data_sampler(self.dataset_train, shuffle=True, distributed=False),
            drop_last=True,
            num_workers=20
        )

        self.loader_train_inf = sample_data(self.loader_train)

        print("CardioLandmarks")
        print("train:", self.dataset_train.__len__())


class MAFL:

    batch_size = 8
    test_batch_size = 32

    def __init__(self):
        dataset_train = MAFLDataset(f"{Paths.default.data()}", split="train", target_type="landmarks")

        self.loader_train = data.DataLoader(
            dataset_train,
            batch_size=MAFL.batch_size,
            sampler=data_sampler(dataset_train, shuffle=True, distributed=False),
            drop_last=True,
            num_workers=20
        )

        self.loader_train_inf = sample_data(self.loader_train)

        self.test_dataset = MAFLDataset(f"{Paths.default.data()}", split="test", target_type="landmarks")

        self.test_loader = data.DataLoader(
            self.test_dataset,
            batch_size=MAFL.test_batch_size,
            drop_last=False,
            num_workers=20
        )

        print("MAFL initialize")
        print(f"train size: {len(dataset_train)}, test size: {len(self.test_dataset)}")

        self.test_loader_inf = sample_data(self.test_loader)


class LazyLoader:

    w300_save: Optional[W300DatasetLoader] = None
    celeba_kp_save: Optional[CelebaWithKeyPoints] = None
    celeba_save: Optional[Celeba] = None
    cardio_save: Optional[Cardio] = None
    cardio_lm_save: Optional[CardioLandmarks] = None
    w300_lm_save: Optional[CardioLandmarks] = None
    celebaWithLandmarks: Optional[CelebaWithLandmarks] = None
    mafl_save: Optional[MAFL] = None


    @staticmethod
    def w300() -> W300DatasetLoader:
        if not LazyLoader.w300_save:
            LazyLoader.w300_save = W300DatasetLoader()
        return LazyLoader.w300_save

    @staticmethod
    def mafl() -> MAFL:
        if not LazyLoader.mafl_save:
            LazyLoader.mafl_save = MAFL()
        return LazyLoader.mafl_save

    @staticmethod
    def celeba_with_kps():
        if not LazyLoader.celeba_kp_save:
            LazyLoader.celeba_kp_save = CelebaWithKeyPoints()
        return LazyLoader.celeba_kp_save

    @staticmethod
    def celeba():
        if not LazyLoader.celeba_save:
            LazyLoader.celeba_save = Celeba()
        return LazyLoader.celeba_save

    @staticmethod
    def cardio():
        if not LazyLoader.cardio_save:
            LazyLoader.cardio_save = Cardio()
        return LazyLoader.cardio_save

    @staticmethod
    def cardio_landmarks():
        if not LazyLoader.cardio_lm_save:
            LazyLoader.cardio_lm_save = CardioLandmarks()
        return LazyLoader.cardio_lm_save\

    @staticmethod
    def w300_landmarks(n: int):
        if not LazyLoader.w300_lm_save:
            LazyLoader.w300_lm_save = W300Landmarks(n)
        return LazyLoader.w300_lm_save

    @staticmethod
    def celeba_test(batch_size=1):
        if not LazyLoader.celebaWithLandmarks:
            LazyLoader.celebaWithLandmarks = sample_data(DataLoader(
                CelebaWithLandmarks(),
                batch_size=batch_size,
                drop_last=True))
        return LazyLoader.celebaWithLandmarks
