
"""
This is where the dataloaders and defined for the HD and SC datasets.
"""
import logging
import os
from pathlib import Path
import csv
import numpy as np

import torch
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchaudio_augmentations import ComposeMany
from torchaudio_augmentations import Gain
from torchaudio_augmentations import Noise
from torchaudio_augmentations import PolarityInversion
from torchaudio_augmentations import RandomApply
from torchaudio_augmentations import Reverb
import math
import pandas as pd
import librosa

logger = logging.getLogger(__name__)

def read_csv_to_dict(csv_path):
    # Initialize an empty dictionary
    path_to_random_number = {}

    # Open the CSV file for reading
    with open(csv_path, mode='r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the header row
        for row in csvreader:
            # Assuming the first column is the file path and the second column is the random number
            file_path = row[0]
            random_number = int(row[1])  # Convert string to integer
            end_time=int(row[2])
            path_to_random_number[file_path] = [random_number, end_time]

    return path_to_random_number


class hundred_words_Dataset_with_startpoint(Dataset):

    def __init__(self, folder, info_path=None, transform=None, classes=None):
        all_classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)) and not d.startswith('_')]
        #for c in classes[2:]:
        #    assert c in all_classes


        class_to_idx = {classes[i]: i for i in range(len(classes))}
        for c in all_classes:
            if c not in class_to_idx:
                class_to_idx[c] = 0

        data = []
        for c in all_classes:
            d = os.path.join(folder, c)
            target = class_to_idx[c]
            time_info = read_csv_to_dict(os.path.join(d, f"begin_end_time.csv"))
            for f in os.listdir(d):
                if f.endswith('.wav'):
                    path = os.path.join(d, f)
                    data.append((path, target, time_info[path]))

        self.classes = classes
        self.data = data
        self.transform = transform
        self.time = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, target, time_info = self.data[index]
        data = {'path': path, 'target': target, 'time_info' : time_info}
        x, sample_rate = torchaudio.load(path)
        if self.transform is not None:
            x = self.transform(x).squeeze(dim=0)
        
        
        length = int(self.time * sample_rate)
        begin_time, _ = data['time_info']
        x = x.squeeze()
        raw_audio = torch.nn.functional.pad(x.squeeze(), (begin_time, length - len(x.squeeze())-begin_time), "constant")

        # Compute acoustic features
        x = torchaudio.compliance.kaldi.fbank(raw_audio.unsqueeze(0), num_mel_bins=40)
        # x = librosa.feature.mfcc(y=x.numpy(), sr=sample_rate, n_mfcc=40, hop_length=160, win_length=400)
        # x = torch.tensor(x, device="cuda")
        time_to_frame_ratio = x.shape[0]/sample_rate
        frame_info = [np.min(np.floor(time_to_frame_ratio * time_info[0])-1, 0), np.ceil(time_to_frame_ratio * time_info[1])-1]
        return x, target, frame_info, raw_audio

    def make_weights_for_balanced_classes(self):
        """adopted from https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3"""

        nclasses = len(self.classes)
        count = np.zeros(nclasses)
        for item in self.data:
            count[item[1]] += 1

        N = float(sum(count))
        weight_per_class = N / count
        weight = np.zeros(len(self))
        for idx, item in enumerate(self.data):
            weight[idx] = weight_per_class[item[1]]
        return weight   
    
    def generateBatch(self, batch):

        xs, ys, timeinfo, raw_audio = zip(*batch)
        xlens = torch.tensor([x.shape[0] for x in xs])
        xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
        ys = torch.LongTensor(ys)
        timeinfo = torch.LongTensor(timeinfo)
        raw_audio = torch.nn.utils.rnn.pad_sequence(raw_audio, batch_first=True)

        return xs, timeinfo, ys, raw_audio
    
def load_100words(
    dataset_name,
    data_folder,
    split,
    batch_size,
    shuffle=True,
    use_augm=False,
    min_snr=0.0001,
    max_snr=0.9,
    p_noise=0.1,
    workers=4,
):
    """
    This function creates a dataloader for a given split of
    the HD or SC dataset.

    Arguments
    ---------
    dataset_name : str
        The name of the dataset, either hd or sc.
    data_folder : str
        Path to folder containing the desired dataset.
    split : str
        Split of the desired dataset, must be either "train" or "test" for hd
        and "training", "validation" or "testing" for sc.
    batch_size : int
        Number of examples in a single generated batch.
    shuffle : bool
        Whether to shuffle examples or not.
    use_augm : bool
        Whether to perform data augmentation or not.
    min_snr, max_snr : float
        Minimum and maximum amounts of noise if augmentation is used.
    p_noise : float in (0, 1)
        Probability to apply noise if augmentation is used, i.e.,
        proportion of examples to which augmentation is applied.
    workers : int
        Number of workers.
    """
    CLASS_path = os.path.join(data_folder, "All_keywords.csv")
    CLASSES = list(pd.read_csv(CLASS_path)['Classes'])
    # Data augmentation
    if use_augm and split == "training":
        transforms = [
            RandomApply([PolarityInversion()], p=0.8),
            RandomApply([Noise(min_snr, max_snr)], p_noise),
            RandomApply([Gain()], p=0.3),
            RandomApply([Reverb(sample_rate=16000)], p=0.6),
        ]
        transf = ComposeMany(transforms, num_augmented_samples=1)
    else:
        transf = lambda x: x.unsqueeze(dim=0)

    if split not in ["train", "valid", "test"]:
        raise ValueError(f"Invalid split name {split}")


    if split == "train":
        train_path = os.path.join(data_folder, 'Train')
        dataset = hundred_words_Dataset_with_startpoint(train_path, 
                                                        transform=transf,
                                      classes=CLASSES)

    elif split == "valid":
        valid_path = os.path.join(data_folder, 'Valid')
        dataset = hundred_words_Dataset_with_startpoint(valid_path, transform=transf,
                                        classes=CLASSES)
    else:
        split = "testing"
        test_path = os.path.join(data_folder, 'Test')
        dataset = hundred_words_Dataset_with_startpoint(test_path, transform=transf,
                                        classes=CLASSES)

    logging.info(f"Number of examples in {dataset_name} {split} set: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.generateBatch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
    )
    return loader
