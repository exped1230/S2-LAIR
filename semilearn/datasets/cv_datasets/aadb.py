"""
Photo Aesthetics Ranking Network with Attributes and Content Adaptation. ECCV 2016
A aesthetics and attributes database (AADB) which contains:

===> 0) 10,000 images in total, 8458 (training) and 1,000 (testing)
with
===> 1) aesthetic scores
and
===> 2) meaningful attributes assigned to each image
by multiple human rater.

paper reference: https://www.ics.uci.edu/~fowlkes/papers/kslmf-eccv16.pdf
code reference: https://github.com/aimerykong/deepImageAestheticsAnalysis

Training images: https://drive.google.com/uc?export=download&id=1Viswtzb77vqqaaICAQz9iuZ8OEYCu6-_ (2 GB)
Test images: https://drive.google.com/uc?export=download&id=115qnIQ-9pl5Vt06RyFue3b6DabakATmJ (212 MB)
Labels: https://github.com/aimerykong/deepImageAestheticsAnalysis/raw/master/AADBinfo.mat (175 KB)

Preprocess part:
1) unzip AADB_newtest_originalSize.zip
2) unzip datasetImages_originalSize.zip
"""
import os
import json
import torchvision
import numpy as np
import math

from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from semilearn.datasets.utils import split_ssl_data
import scipy.io as scio
from PIL import Image
from torchvision.datasets import VisionDataset

from semilearn.datasets.cv_datasets.common.utils import check_if_file_exists, join
from semilearn.datasets.cv_datasets.factory import DatasetFactory

mean, std = {}, {}

mean['aadb'] = [0.485, 0.456, 0.406]
std['aadb'] = [0.229, 0.224, 0.225]
class AADBBaseDataset(VisionDataset):
    """
    AADBBaseDataset that contains all the elements in AADB datasets,
    which can be inherited into the following datasets:
    1) aesthetic binary classification datasets
    2) aesthetic score regression datasets

    The elements in AVABaseDataset include:
    1) all images
    3) averaged aesthetic scores
    """

    def __init__(self, root, split='train', transforms=None):
        super().__init__(root=root, transforms=transforms)
        assert split in ['train', 'test'], 'Got unsupported split: `%s`' % split
        self.split = split

        if self.split == 'train':
            self.image_root = join(self.root, 'datasetImages_originalSize')
        else:
            self.image_root = join(self.root, 'AADB_newtest_originalSize')

        self.split_and_label_root = join(self.root, 'AADBinfo.mat')

        check_if_file_exists(self.split_and_label_root)

        self._images, self.scores = self._process_split_and_label()

        self._targets = None

    @property
    def images(self):
        return self._images

    @property
    def targets(self):
        return self._targets

    def __getitem__(self, index):
        image_id = self.images[index]
        target = self.targets[index]
        image = Image.open(join(self.image_root, image_id)).convert('RGB')

        if self.transforms:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.images)

    def _process_split_and_label(self):
        def squeeze_and_to_list(_x):
            return _x.squeeze().tolist()

        def squeeze_and_retrieve_and_to_list(_x):
            tmp = squeeze_and_to_list(_x)
            return [t[0] for t in tmp]
        def squeeze_and_retrieve_and_to_train_list(_x):
            tmp = squeeze_and_to_list(_x)
            return [join('/home/ubuntu/jgl/datasets/aadb/datasetImages_originalSize', t[0]) for t in tmp]
        def squeeze_and_retrieve_and_to_test_list(_x):
            tmp = squeeze_and_to_list(_x)
            return [join('/home/ubuntu/jgl/datasets/aadb/AADB_newtest_originalSize', t[0]) for t in tmp]
        data = scio.loadmat(self.split_and_label_root)
        for k, v in data.items():
            if k in ['testScore', 'trainScore']:
                data[k] = squeeze_and_to_list(data[k])
            # if k in ['testNameList', 'trainNameList']:
            #     data[k] = squeeze_and_retrieve_and_to_list(data[k])
            if k in ['trainNameList']:
                data[k] = squeeze_and_retrieve_and_to_train_list(data[k])
            if k in ['testNameList']:
                data[k] = squeeze_and_retrieve_and_to_test_list(data[k])
            # if k in ['testNameList', 'testScore']:
            #     print("===> k: ", k)
            #     print("===> data[k]: ", data[k])

        if self.split == 'train':
            return data['trainNameList'], data['trainScore']
        else:
            return data['testNameList'], data['testScore']

def get_label(target):
    ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1)]

    if target==0:
        return 0
    for i in range(len(ranges)):
        if ranges[i][0] < target <= ranges[i][1]:
            return i

@DatasetFactory.register('AADBClassificationDataset')
class AADBClassificationDataset(AADBBaseDataset):
    """
    AADBClassificationDataset that used for binary aesthetic classification.
    The binary label is obtained by setting the label to 1 when the aesthetic scores > 0.5
    and setting to 0 otherwise.
    """

    def __init__(self, root, split='train', transforms=None):
        super().__init__(root=root, split=split, transforms=transforms)
        #self._targets = [1 if s > 0.5 else 0 for s in self.scores]
        self._targets = [get_label(s) for s in self.scores]


@DatasetFactory.register('AADBRegressionDataset')
class AADBRegressionDataset(AADBBaseDataset):
    """
    AADBRegressionDataset that used for aesthetic score regression,
    since the ranking in aesthetic assessment is often very important.
    """

    def __init__(self, root, split='train', transforms=None):
        super().__init__(root=root, split=split, transforms=transforms)
        self._targets = self.scores


def get_aadb(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    
    # data_dir = os.path.join(data_dir, name.lower())
    # dset = getattr(torchvision.datasets, name.upper())
    # dset = dset(data_dir, train=True, download=True)
    # data, targets = dset.data, dset.targets

    base_dataset = AADBClassificationDataset(root=data_dir, split='train')
    data = np.array(base_dataset.images)
    target=base_dataset.targets
    print(len(target))
    
    
    crop_size = args.img_size
    img_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['aadb'], std['aadb'])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(mean['aadb'], std['aadb'])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean['aadb'], std['aadb'])
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

    base_dataset = AADBClassificationDataset(root=data_dir, split='test')
    test_data = np.array(base_dataset.images)
    test_target=base_dataset.targets

    eval_dset = BasicDataset(alg, test_data, test_target, num_classes, transform_val, False, None, False)

    return lb_dset, ulb_dset, eval_dset