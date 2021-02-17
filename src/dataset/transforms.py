from albumentations import *
from albumentations.pytorch import ToTensor


def pre_transform(resize):
    transforms = []
    transforms.append(Resize(resize, resize))
    return Compose(transforms)

def post_transform():
    return Compose([
        ToTensor()
    ])

def mix_transform(resize):
    return Compose([
        pre_transform(resize=resize),
        HorizontalFlip(),
        post_transform(),
        OneOf([
            RandomContrast(),
            RandomGamma(),
            RandomBrightness(),
        ], p=0.3),
    ])