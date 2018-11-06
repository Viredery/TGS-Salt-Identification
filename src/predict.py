import numpy as np
import pandas as pd
from time import time
from sklearn.model_selection import StratifiedKFold

import torchvision
import torch
import torch.nn as nn 
import torch.nn.functional as F


from transform import *
from data import TsgDataset
from model import UNetResNet34

from metrics import intersection_over_union_thresholds
from utils import RLenc

NAME = 'UNetResNet34'



def valid_augment_padding(index, image, mask, label):
    image = do_center_pad_to_factor(image, factor=32)
    mask = (mask >= 0.5).astype(np.float32)
    return index, image, mask, label


def valid_augment_resize(index, image, mask, label):
    image = cv2.resize(image, (128, 128))
    mask = (mask >= 0.5).astype(np.float32)
    return index, image, mask, label


def do_eval(net, dataset, mode='padding'):
    net.set_mode('eval')

    probs = np.zeros((len(dataset), 101, 101))
    truths = np.zeros((len(dataset), 101, 101))

    for i in range(len(dataset)):
        with torch.no_grad():
            index, image, y_mask, y_label = dataset[i]

            hflip_image = np.array(image)[:, ::-1]
            images = np.array([image, hflip_image])
            images = torch.Tensor(images).cuda()

            logit_fuse, logit_pixel, logit_image = net(images)
            prob = logit_fuse.sigmoid()

            prob = prob.cpu().data.numpy().squeeze()
            mask = prob[0]
            hflip_mask = prob[1][:, ::-1]
            prob = (mask + hflip_mask) / 2

            if mode == 'padding':
                prob = prob[13:128-14, 13:128-14]
            else:
                prob = cv2.resize(prob, (101, 101))

            probs[i, :, :] = prob
            truths[i, :, :] = y_mask

    iou = intersection_over_union_thresholds(
        np.int32(truths >= 0.5), np.int32(probs >= 0.5))

    return probs, iou


def do_eval_with_pros(probs, dataset):
    truths = np.zeros((len(dataset), 101, 101))

    for i in range(len(dataset)):
        index, image, y_mask, y_labels = dataset[i]
        truths[i, :, :] = y_mask

    iou = intersection_over_union_thresholds(
        np.int32(truths >= 0.5), np.int32(probs >= 0.5))

    return iou


def do_test(net, dataset, mode='padding'):
    net.set_mode('eval')
    probs = np.zeros((len(dataset), 101, 101))

    for i in range(len(dataset)):
        with torch.no_grad():
            index, image, y_mask, y_labels = dataset[i]

            hflip_image = np.array(image)[:, ::-1]
            images = np.array([image, hflip_image])
            images = torch.Tensor(images).cuda()

            logit_fuse, logit_pixel, logit_image = net(images)
            prob = logit_fuse.sigmoid()
            
            prob = prob.cpu().data.numpy().squeeze()
            mask = prob[0]
            hflip_mask = prob[1][:, ::-1]
            prob = (mask + hflip_mask) / 2

            if mode == 'padding':
                prob = prob[13:128-14, 13:128-14]
            else:
                prob = cv2.resize(prob, (101, 101))

            probs[i, :, :] = prob
    return probs


data = pd.read_csv('../data/data_ids_with_class.csv')
data_ids = data['id'].values
data_class = data['class'].values

test_ids = pd.read_csv('../data/test_ids.csv')['id'].values
test_probs = np.zeros((len(test_ids), 101, 101))

kfold = 5
test_num = [0, 1, 2, 3, 4]
fold_num = len(test_num)
cv = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=1337)

cycle_list = [2, 4, 3, 5, 5]

mode_list = ['padding']
mode_num = len(mode_list)

for mode in mode_list:
    if mode not in {'padding', 'resize'}:
        raise ValueError('{} is not defined!'.format(mode))

for cv_num, (train_idx, val_idx) in enumerate(cv.split(data_ids, data_class)):
    
    print('cv:', cv_num)
    if cv_num not in test_num:
        continue

    train_ids, val_ids = data_ids[train_idx], data_ids[val_idx]
    fold_probs = np.zeros((len(val_ids), 101, 101))

    for mode in mode_list:
        print('---mode: ', mode)
        if mode == 'padding':
            test_dataset = TsgDataset(root='../data/test', image_ids=test_ids,
                                      augment=valid_augment_padding, mode='test')

            valid_dataset = TsgDataset(root='../data/train', image_ids=val_ids,
                                       augment=valid_augment_padding, mode='valid')
        elif mode == 'resize':
            test_dataset = TsgDataset(root='../data/test', image_ids=test_ids,
                                      augment=valid_augment_resize, mode='test')

            valid_dataset = TsgDataset(root='../data/train', image_ids=val_ids,
                                       augment=valid_augment_resize, mode='valid')

        cycle_probs = np.zeros((len(valid_dataset), 101, 101))

        net = UNetResNet34().cuda()

        net.load_state_dict(torch.load('../weights/{}_{}_lovasz_loss_clr_{}_{}.th'.format(
            NAME, cv_num, cycle_list[cv_num], mode)))

        probs, iou = do_eval(net, valid_dataset, mode=mode)
        cycle_probs += probs
        fold_probs += probs
        print('------cycle {}, valid iou: {}'.format(cycle_list[cv_num], iou))
        test_probs += do_test(net, test_dataset, mode=mode)



test_probs /= fold_num

pred_dict = {idx: RLenc(np.where(test_probs[i] >= 0.5, 1, 0)) for i, idx in enumerate(test_ids)}
sub = pd.DataFrame.from_dict(pred_dict, orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('./{}_cv.csv'.format(NAME))


    
