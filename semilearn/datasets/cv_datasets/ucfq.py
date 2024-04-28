# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import torchvision
import numpy as np
import math
from scipy.io import loadmat
from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from semilearn.datasets.utils import split_ssl_data
import random

mean, std = {}, {}
mean['cifar10'] = [0.485, 0.456, 0.406]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]
mean['ucfq'] = [0.485, 0.456, 0.406]

std['cifar10'] = [0.229, 0.224, 0.225]
std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]
std['ucfq'] = [0.229, 0.224, 0.225]

def score2label(dmos):
        if 0<=dmos<=300:
           return 0
        if 300<dmos<=700:
           return 1
        if 700<dmos<=1200:
           return 2
        if 1200<dmos<=2300:
           return 3       
        if dmos>2300:
           return 4     
  
def get_ucfq(args, alg, name, num_labels, num_classes, data_dir='/home/ubuntu/wwc/dataset/ucf-q/train', include_lb_to_ulb=True):
    data = []
    target = []
    # with open(data_dir+'/image_labels.txt', 'r') as file:
    #     # 逐行读取文件内容
    #     for line in file:
    #         # 处理每一行的数据
    #         line = line.split(',')
    #         image = line[0]
    #         score = int(line[1])
    #         data.append('/home/ubuntu/wwc/dataset/jhu_crowd_v2.0/train/images/'+image+'.jpg')
    #         target.append(score2label(score))
    for i in range(int(len(os.listdir(data_dir))/2)):
        mat = '/home/ubuntu/jgl/datasets/UCF-Q/Train/'+'img_{:04}_ann.mat'.format(i+1)
        img_path = '/home/ubuntu/jgl/datasets/UCF-Q/Train/'+'img_{:04}.jpg'.format(i+1)
        anno = loadmat(mat)
        num = anno['annPoints'].shape
        target.append(score2label(num[0]))
        data.append(img_path)   
        
            
    # random.seed(args.seed)
    # data_target_pairs = list(zip(all_data, all_target))
    # random.shuffle(data_target_pairs)
    # train_size = int(0.8 * len(data_target_pairs))
    # test_size = len(data_target_pairs) - train_size
    # train_pairs = data_target_pairs[:train_size]
    # test_pairs = data_target_pairs[train_size:]
    
    # data = [pair[0] for pair in train_pairs]
    # target = [pair[1] for pair in train_pairs]
    # test_data = [pair[0] for pair in test_pairs]
    # test_target = [pair[1] for pair in test_pairs]
    print(len(data),len(target))
    crop_size = args.img_size
    img_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['ucfq'], std['ucfq'])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(mean['ucfq'], std['ucfq'])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean['ucfq'], std['ucfq'])
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

    # base_dataset = torchvision.datasets.ImageFolder(args.test_data_dir)
    # imgs = np.array(base_dataset.imgs)
    # test_data, test_target = imgs[:, 0], imgs[:, 1]
    # test_target = [int(element) for element in test_target]
    test_data = []
    test_target = []
    for i in range(int(len(os.listdir(args.test_data_dir))/2)):
        mat = '/home/ubuntu/jgl/datasets/UCF-Q/Test/'+'img_{:04}_ann.mat'.format(i+1)
        img_path = '/home/ubuntu/jgl/datasets/UCF-Q/Test/'+'img_{:04}.jpg'.format(i+1)
        anno = loadmat(mat)
        num = anno['annPoints'].shape
        test_target.append(score2label(num[0]))
        test_data.append(img_path)   


    eval_dset = BasicDataset(alg, test_data, test_target, num_classes, transform_val, False, None, False)

    return lb_dset, ulb_dset, eval_dset