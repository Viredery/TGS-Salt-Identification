import os
import time
import numpy as np
import pandas as pd
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def dummy_augment(index, image, mask):
    return index, image, mask


class TsgDataset(Dataset):
    def __init__(self, root, image_ids, augment, mode='train'):
        self.root = root
        self.image_ids = image_ids
        self.mode = mode
        self.augment = augment
        
        self.depths = pd.read_csv('../data/depths.csv', index_col='id')
        
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image = cv2.imread(os.path.join(self.root, 'images/{}.png'.format(image_id)),
                           cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        
        if self.mode in ['train', 'valid']:
            mask = cv2.imread(os.path.join(self.root, 'masks/{}.png'.format(image_id)),
                              cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
            label = 0 if np.sum(mask == 1) == 0 else 1
        elif self.mode in ['test']:
            mask = np.array([])
            label = None

        depth = self.depths.loc[image_id]['z']
        return self.augment(index, image, mask, label)

    def __len__(self):
        return len(self.image_ids)
