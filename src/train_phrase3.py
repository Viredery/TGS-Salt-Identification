
from include import *
import numpy as np
import pandas as pd
import math
from time import time
from sklearn.model_selection import StratifiedKFold

import torchvision
import torch
import torch.nn as nn 
import torch.nn.functional as F


from transform import *
from data import TsgDataset
from model import UNetResNet34
from loss import *
from metrics import intersection_over_union_thresholds

NAME = 'UNetResNet34'
BATCH_SIZE = 16
MODE = 'padding'

if MODE not in {'padding', 'resize'}:
    raise ValueError('{} is not defined!'.format(MODE))

def valid_augment(index, image, mask, label):

    if MODE == 'padding':
        image = do_center_pad_to_factor(image, factor=32)
    else:
        image = cv2.resize(image, (128, 128))

    mask = (mask >= 0.5).astype(np.float32)
    return index, image, mask, label


def train_augment(index, image, mask, label):

    if np.random.rand() < 0.5:
        image, mask = randomHorizontalFlip(image, mask)
        
    if np.random.rand() < 0.5:
        image, mask = do_random_shift_scale_crop_pad2(image, mask, 0.1)
        
    if np.random.rand() < 0.5:
        c = np.random.choice(3)
        if c==0:
            image, mask = do_random_shift_scale_crop_pad2(image, mask, 0.125)
        if c==1:
            image, mask = do_elastic_transform2(image, mask, grid=10,
                                                distort=np.random.uniform(0,0.1))
        if c==2:
            image, mask = do_shift_scale_rotate2(image, mask, dx=0, dy=0, scale=1,
                                                 angle=np.random.uniform(0,10))

    if np.random.rand() < 0.5:
        c = np.random.choice(3)
        if c==0:
            image = do_brightness_shift(image, np.random.uniform(-0.05, +0.05))
        if c==1:
            image = do_brightness_multiply(image, np.random.uniform(1-0.05, 1+0.05))
        if c==2:
            image = do_gamma(image, np.random.uniform(1-0.05, 1+0.05))

    if MODE == 'padding':
        image, mask = do_center_pad_to_factor2(image, mask, factor=32)
    else:
        image = cv2.resize(image, (128, 128))
        mask = cv2.resize(mask, (128, 128))

    mask = (mask >= 0.5).astype(np.float32)
    return index, image, mask, label


def do_eval(net, dataset):
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

            if MODE == 'padding':
                prob = prob[13:128-14, 13:128-14]
            else:
                prob = cv2.resize(prob, (101, 101))

            probs[i, :, :] = prob
            truths[i, :, :] = y_mask

    iou = intersection_over_union_thresholds(
        np.int32(truths >= 0.5), np.int32(probs >= 0.5))

    return iou

def criterion(logit_fuse, logit_pixel, logit_image, truth_pixel, truth_image):

    # classification loss
    loss_image = F.binary_cross_entropy_with_logits(logit_image, truth_image, reduction='elementwise_mean')

    # non-empty loss
    loss_pixel = lovasz_loss_per_image(logit_pixel, truth_pixel)
    loss_pixel = loss_pixel * truth_image
    loss_pixel = loss_pixel.sum() / (truth_image.sum() + 1e-12)

    # all images loss
    loss_fuse = lovasz_hinge(logit_fuse, truth_pixel)

    return loss_image * 0.05 + loss_pixel * 0.5 + loss_fuse * 1

data = pd.read_csv('../data/data_ids_with_class.csv')
data_ids = data['id'].values
data_class = data['class'].values


kfold = 5
train_num = [0, 1, 2, 3, 4]
cv = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=1337)

print('MODE: ', MODE)
for cv_num, (train_idx, val_idx) in enumerate(cv.split(data_ids, data_class)):

    print('cv:', cv_num)

    if cv_num not in train_num:
        continue
    f = open('../logs/{}_{}_lovasz_loss_clr_{}.txt'.format(NAME, cv_num, MODE), 'w+')
    f.close()

    train_ids, val_ids = data_ids[train_idx], data_ids[val_idx]

    train_dataset = TsgDataset(root='../data/train', image_ids=train_ids, augment=train_augment, mode='train')
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    valid_dataset = TsgDataset(root='../data/train', image_ids=val_ids, augment=valid_augment, mode='valid')

    net = UNetResNet34()
    net.cuda()
    net.load_state_dict(torch.load('../weights/{}_{}_lovasz_loss_{}.th'.format(NAME, cv_num, MODE)))


    max_lr = 1e-4
    min_lr = 1e-7
    optimizer = torch.optim.Adam(net.parameters(), lr=max_lr)

    num_epochs = 360
    circle_size = 60
    num_circles = int(num_epochs / circle_size)
    epoch = 0

    best_iou_metric = np.zeros(num_circles)
    best_loss_metric = np.ones(num_circles)
    best_train_iou_metric = np.ones(num_circles)

    tic = time()

    while epoch < num_epochs:
        
        train_loss = 0
        
        epoch_items = len(train_loader)

        for i, (indices, images, y_masks, y_labels) in enumerate(train_loader):
            net.set_mode('train')

            optimizer.zero_grad()

            images = images.float().cuda()
            y_masks = y_masks.float().cuda()
            y_labels = y_labels.float().cuda()

            logit_fuse, logit_pixel, logit_image = net(images)

            logit_fuse = logit_fuse.squeeze()
            logit_pixel = logit_pixel.squeeze()
            logit_image = logit_image.squeeze()

            loss = criterion(logit_fuse, logit_pixel, logit_image, y_masks, y_labels)

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            epoch_ = epoch % circle_size
            lr = min_lr + (max_lr - min_lr) / 2. * (1 + math.cos(
                (epoch_ * epoch_items + i) / (circle_size * epoch_items) * math.pi))
            
            optimizer =  torch.optim.Adam(net.parameters(), lr=lr)
            
        train_loss = train_loss / len(train_loader)
        eval_iou = do_eval(net, valid_dataset)

        print('[%03d] duration: %.2f train_loss: %.4f valid_iou: %.4f' % (
            epoch+1, time() - tic, train_loss, eval_iou))

        print('the current learning rate is {}'.format(lr))

        num_circle = int(epoch / circle_size)
        if eval_iou > best_iou_metric[num_circle]:
            best_iou_metric[num_circle] = eval_iou
            print('saving the best eval_iou model in clr {}. iou:{}'.format(num_circle, eval_iou))
            torch.save(net.state_dict(), '../weights/{}_{}_lovasz_loss_clr_{}_{}.th'.format(NAME,
                                                                                            cv_num,
                                                                                            num_circle,
                                                                                            MODE))

        with open('../logs/{}_{}_lovasz_loss_clr_{}.txt'.format(NAME, cv_num, MODE), 'a+') as f:
            f.write('[%03d] train_loss: %.4f,  '
                    'valid_iou: %.4f\n' % (epoch+1, train_loss, eval_iou))

        epoch += 1
    