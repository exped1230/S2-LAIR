# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import torchvision
import numpy as np
import math

from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from semilearn.datasets.utils import split_ssl_data


mean, std = {}, {}
mean['cifar10'] = [0.485, 0.456, 0.406]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]
mean['age'] = [0.485, 0.456, 0.406]

std['cifar10'] = [0.229, 0.224, 0.225]
std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]
std['age'] = [0.229, 0.224, 0.225]


def get_adience(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    
    #data_dir = os.path.join(data_dir, name.lower())
    # dset = getattr(torchvision.datasets, name.upper())
    # dset = dset(data_dir, train=True, download=True)
    #data, targets = dset.data, dset.targets
    base_dataset = torchvision.datasets.ImageFolder(data_dir)
    imgs = np.array(base_dataset.imgs)
    data, target = imgs[:, 0], imgs[:, 1]
    target = [int(element) for element in target]
    
    # data=[]
    # target=[]

    # for i in range(5):
    #     if i==args.fold_id:
    #         continue
    #     fold_path='fold_{:}_data'.format(i)
    #     base_dataset = torchvision.datasets.ImageFolder(data_dir+'/{:}'.format(fold_path))
    #     imgs = np.array(base_dataset.imgs)
    #     data_part, target_part = imgs[:, 0], imgs[:, 1]
    #     target_part = [int(element) for element in target_part]
    #     data.extend(data_part)
    #     target = target_part + target
    print(len(data),len(target))
    crop_size = args.img_size
    img_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['age'], std['age'])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(mean['age'], std['age'])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean['age'], std['age'])
    ])


    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(args, data, target, num_classes, 
                                                                lb_num_labels=num_labels,
                                                                ulb_num_labels=args.ulb_num_labels,
                                                                lb_imbalance_ratio=args.lb_imb_ratio,
                                                                ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                include_lb_to_ulb=include_lb_to_ulb)
    
    lb_count = [0 for _ in range(num_classes)]
    ulb_count = [0 for _ in range(num_classes)]
    for c in lb_targets:
        lb_count[c] += 1
    for c in ulb_targets:
        ulb_count[c] += 1
    print("lb count: {}".format(lb_count))
    print("ulb count: {}".format(ulb_count))
    # lb_count = lb_count / lb_count.sum()
    # ulb_count = ulb_count / ulb_count.sum()
    # args.lb_class_dist = lb_count
    # args.ulb_class_dist = ulb_count

    if alg == 'fullysupervised':
        lb_data = data
        lb_targets = target
        # if len(ulb_data) == len(data):
        #     lb_data = ulb_data 
        #     lb_targets = ulb_targets
        # else:
        #     lb_data = np.concatenate([lb_data, ulb_data], axis=0)
        #     lb_targets = np.concatenate([lb_targets, ulb_targets], axis=0)
    
    # output the distribution of labeled data for remixmatch
    # count = [0 for _ in range(num_classes)]
    # for c in lb_targets:
    #     count[c] += 1
    # dist = np.array(count, dtype=float)
    # dist = dist / dist.sum()
    # dist = dist.tolist()
    # out = {"distribution": dist}
    # output_file = r"./data_statistics/"
    # output_path = output_file + str(name) + '_' + str(num_labels) + '.json'
    # if not os.path.exists(output_file):
    #     os.makedirs(output_file, exist_ok=True)
    # with open(output_path, 'w') as w:
    #     json.dump(out, w)

    lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, None, False)

    ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_strong, False)

    base_dataset = torchvision.datasets.ImageFolder(args.test_data_dir)
    imgs = np.array(base_dataset.imgs)
    test_data, test_target = imgs[:, 0], imgs[:, 1]
    test_target = [int(element) for element in test_target]
    print(len(test_target))

    eval_dset = BasicDataset(alg, test_data, test_target, num_classes, transform_val, False, None, False)

    return lb_dset, ulb_dset, eval_dset
