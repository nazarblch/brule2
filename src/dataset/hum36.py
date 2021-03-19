import os.path

import torch
from PIL import Image
import torchvision.transforms as transforms
import skimage.io
import skimage.transform
import random
import numpy as np
import scipy.io
from torch.utils import data

from dataset import utils
from torch.utils.data.dataset import Dataset

def proc_im(image, mask, apply_mask=True):
    # read image
    image = skimage.io.imread(image)
    image = skimage.img_as_float(image).astype(np.float32)
    if not apply_mask:
        return image

    mask = skimage.io.imread(mask)
    mask = skimage.img_as_float(mask).astype(np.float32)

    return image * mask[..., None]


def get_transform(channels=3):
    mean = 0.5
    std = 0.5
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize([mean] * channels,
                                           [std] * channels)]
    return transforms.Compose(transform_list)



class SimpleHuman36mDatasetSingle(object):
    def __init__(self, root, sample_window=[5, 30], activities=None,
                 actors=None, split_sequence='full', subsampled_size=None, subsample_seed=None):

        self.root = root
        self.sample_window = sample_window

        self.ordered_stream = None

        # load dataset
        self.sequences = []
        for actor in actors:
            sequences = os.listdir(os.path.join(root, actor, 'BackgroudMask'))
            sequences = sorted(sequences)
            for activity in activities:
                activity_sequences = [s for s in sequences if s.lower().startswith(activity.lower())]
                for seq in activity_sequences:
                    frames = os.listdir(os.path.join(root, actor, 'BackgroudMask', seq))
                    frames = [int(os.path.splitext(x)[0]) for x in frames]
                    frames = sorted(frames)
                    if split_sequence == 'full':
                        pass
                    elif split_sequence == 'first_half':
                        frames = frames[:len(frames) // 2]
                    elif split_sequence == 'second_half':
                        frames = frames[len(frames) // 2:]
                    else:
                        raise ValueError()
                    self.sequences.append({'frames': frames, 'actor': actor, 'activity_sequence': seq})

        if subsampled_size:
            sequences_ = []
            rnd = random.Random(subsample_seed)
            for _ in range(subsampled_size):
                seq = rnd.choice(self.sequences).copy()
                seq['frames'] = [rnd.choice(seq['frames'])]
                sequences_.append(seq)
            self.sequences = sequences_


    def get_pair(self, sequence, frame1, frame2):
        def get_single(sequence, frame):
            mat_file = os.path.join(self.root, sequence['actor'], 'Landmarks', sequence['activity_sequence'], str(frame) + '.mat')
            mat = scipy.io.loadmat(mat_file)
            landmarks = mat['keypoints_2d'] * 128.0
            return {
                'image': os.path.join(self.root, sequence['actor'], 'WithBackground', sequence['activity_sequence'], str(frame) + '.jpg'),
                'mask': os.path.join(self.root, sequence['actor'], 'BackgroudMask', sequence['activity_sequence'], str(frame) + '.png'),
                'landmarks': landmarks
                }
        return get_single(sequence, frame1), get_single(sequence, frame2)


    def get_ordered_stream(self):
        if self.ordered_stream is None:
            self.ordered_stream = []
            for sequence in self.sequences:
                step = self.sample_window[1]
                for i in range(0, len(sequence['frames']), step):
                    frame = sequence['frames'][i]
                    self.ordered_stream.append((sequence, frame))
        return self.ordered_stream


    def get_item(self, index):
        ordered_stream = self.get_ordered_stream()
        # print(len(ordered_stream))
        sequence, frame = ordered_stream[index]
        return self.get_pair(sequence, frame, frame)


    def sample_item(self):
        sequence = random.choice(self.sequences)
        length = len(sequence['frames'])
        start = random.randint(0, length - self.sample_window[0] - 1)
        end = random.randint(
            start + self.sample_window[0],
            min(start + self.sample_window[1], length - 1))
        return self.get_pair(
            sequence, sequence['frames'][start],
            sequence['frames'][end])


    def num_samples(self):
        return len(self.get_ordered_stream())



class SimpleHuman36mDataset(Dataset):


    def initialize(self, data_path, subset='train', sample_window=[5, 30], skeleton_subset_size=0, skeleton_subset_seed=None, output_nc=3, use_mask = False):

        self.load_images = True
        self.root = data_path

        self.use_mask = use_mask
        self.fineSize = 128

        activities = ['directions', 'discussion', 'greeting', 'posing',
                      'waiting', 'walking']
        train_actors = ['S%d' % i for i in [1, 5, 6, 7, 8, 9]]
        val_actors = ['S%d' % i for i in [11]]
        test_actors = val_actors

        if subset == 'train':
            actors = train_actors
        elif subset == 'val':
            actors = val_actors
        elif subset == 'test':
            actors = test_actors
        else:
            raise ValueError()

        if subset == 'train':
            order_stream = True
            split_sequence = 'first_half'
        elif subset == 'val':
            order_stream = True
            split_sequence = 'full'
            order_stream = True
        elif subset == 'test':
            order_stream = True
            split_sequence = 'full'
        else:
            ValueError()

        self.dataset = SimpleHuman36mDatasetSingle(
            self.root, sample_window=sample_window,
            activities=activities, actors=actors,
            split_sequence=split_sequence)

        if subset == 'train':
            self.skeleton_dataset = SimpleHuman36mDatasetSingle(
                self.root, sample_window=[5, 30],
                activities=activities, actors=actors,
                split_sequence='second_half',
                subsampled_size=skeleton_subset_size,
                subsample_seed=skeleton_subset_seed)
        else:
            self.skeleton_dataset = self.dataset

        self.len = self.dataset.num_samples()
        self.ordered_stream = order_stream

        self.A_transform = get_transform()
        self.B_transform = get_transform(channels=output_nc)

    def _get_sample(self, dataset, index, load_image=True):
        if self.ordered_stream:
            source, target = dataset.get_item(index)
        else:
            source, target = dataset.sample_item()
        landmarks = utils.swap_xy_points(source['landmarks'])
        future_landmarks = utils.swap_xy_points(target['landmarks'])

        landmarks = landmarks.astype('float32')
        future_landmarks = future_landmarks.astype('float32')

        if load_image:
            future_image = proc_im(source['image'], source['mask'], apply_mask=self.use_mask)
            source_image = proc_im(target['image'], target['mask'], apply_mask=self.use_mask)
        else:
            future_image = None
            source_image = None


        return source_image, future_image, source['image'], target['image'], landmarks, future_landmarks,


    def __getitem__(self, index):
        # sample
        cond_A_img, A_img, cond_A_path, A_paths, paired_cond_B, paired_B = self._get_sample(self.dataset, index)

        # sample B
        _, _, _, _, _, B = self._get_sample(self.skeleton_dataset, index, load_image=False)

        # normalize keypoints
        paired_cond_B = utils.normalize_points(
            paired_cond_B, self.fineSize, self.fineSize)
        paired_B = utils.normalize_points(
            paired_B, self.fineSize, self.fineSize)
        B = utils.normalize_points(
            B, self.fineSize, self.fineSize)

        if self.load_images:
            A = self.A_transform(A_img)
            cond_A = self.A_transform(cond_A_img)

        data = {'B': torch.from_numpy(B),
                'paired_cond_B': torch.from_numpy(paired_cond_B),
                'paired_B': torch.from_numpy(paired_B),
                'A_paths': A_paths, 'cond_A_path': cond_A_path}
        if self.load_images:
            data.update({'A': A, 'cond_A': cond_A})
        return data


    def __len__(self):
        return self.len

    def name(self):
        return 'UnalignedDataset'
