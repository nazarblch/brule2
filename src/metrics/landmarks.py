from dataset.lazy_loader import LazyLoader
from loss.weighted_was import OTWasDist


def verka_cardio_w2(enc):
    sum_loss = 0
    n = len(LazyLoader.cardio().test_dataset)
    for i, batch in enumerate(LazyLoader.cardio().test_loader):
        data = batch['image'].cuda()
        landmarks_ref = batch["keypoints"].cuda()
        pred = enc(data)["mes"].coord
        sum_loss += OTWasDist().forward(pred, landmarks_ref).sum().item()
    print("test loss: ", sum_loss / n)
    return sum_loss / n


def verka_300w(enc):
    sum_loss = 0
    n = len(LazyLoader.w300().test_dataset)
    for i, batch in enumerate(LazyLoader.w300().test_loader):
        data = batch['data'].cuda()
        landmarks = batch["meta"]["keypts_normalized"].cuda()
        pred = enc(data)["mes"].coord
        eye_dist = landmarks[:, 45] - landmarks[:, 36]
        eye_dist = eye_dist.pow(2).sum(dim=1).sqrt()
        sum_loss += ((pred - landmarks).pow(2).sum(dim=2).sqrt().mean(dim=1) / eye_dist).sum().item()
    print("test loss: ", sum_loss / n)
    return sum_loss / n


def verka_300w_w2(enc):
    sum_loss = 0
    n = len(LazyLoader.w300().test_dataset)
    for i, batch in enumerate(LazyLoader.w300().test_loader):
        data = batch['data'].cuda()
        landmarks = batch["meta"]["keypts_normalized"].cuda()
        pred = enc(data)["mes"].coord
        eye_dist = landmarks[:, 45] - landmarks[:, 36]
        eye_dist = eye_dist.pow(2).sum(dim=1).sqrt()
        sum_loss += (OTWasDist().forward(pred, landmarks) / eye_dist).sum().item()
    print("test loss: ", sum_loss / n)
    return sum_loss / n


def verka_human(enc):
    sum_loss = 0
    n = len(LazyLoader.human36().test_dataset)
    for i, batch in enumerate(LazyLoader.human36().test_loader):
        data = batch['A'].cuda()
        landmarks = batch["paired_B"].cuda()
        pred = enc(data)["mes"].coord
        # eye_dist = landmarks[:, 45] - landmarks[:, 36]
        # eye_dist = eye_dist.pow(2).sum(dim=1).sqrt()
        sum_loss += ((pred - landmarks).pow(2).sum(dim=2).sqrt().mean(dim=1) ).sum().item()
    print("test loss: ", sum_loss / n)
    return sum_loss / n