#from __future__ import print_function, division

import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import os.path

import cv2
import torchvision

def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

        self.targets = []
        for img in imgs:
            self.targets.append(img[1])

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class ImageList_idx(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)

class ImageList_idx_(Dataset):
    def __init__(self, image_list, labels=None, transform=None, transform_list=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.transform_list = transform_list
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

        self.targets = []
        for img in imgs:
            self.targets.append(img[1])

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform_list is not None:
            img_transformed = []
            for transform in self.transform_list:
                img_transformed.append(transform(img))
            img = torch.stack(img_transformed)
        else:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)

class ImageList_noisy_idx(Dataset):
    def __init__(self, image_list, labels=None, transform=None, transform_list=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.transform_list = transform_list
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

        self.path = []
        self.targets = []
        for (path, target) in imgs:
            self.path.append(path)
            self.targets.append(target)

    def __getitem__(self, index):
        # path, target = self.imgs[index]
        path, target = self.path[index], self.targets[index]
        img = self.loader(path)
        if self.transform_list is not None:
            img_transformed = []
            for transform in self.transform_list:
                img_transformed.append(transform(img))
            # img = torch.stack(img_transformed)
            img = img_transformed
        else:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return [[img, target], index]

    def __len__(self):
        return len(self.imgs)


class ImageList_aug_idx(Dataset):
    def __init__(self, image_list, labels=None, transform=None, transform_list=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.transform_list = transform_list
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

        self.path = []
        self.targets = []
        for (path, target) in imgs:
            self.path.append(path)
            self.targets.append(target)

    def __getitem__(self, index):
        # path, target = self.imgs[index]
        path, target = self.path[index], self.targets[index]
        img = self.loader(path)

        img_transformed = []
        # img_transformed2 = []
        weak_augmented1 = self.transform_list[0](img)
        img_transformed.append(weak_augmented1)
        weak_augmented2 = self.transform_list[1](img)
        img_transformed.append(weak_augmented2)
        weak_augmented = torch.stack(img_transformed)

        strong_augmented1 = self.transform_list[2](img)
        # img_transformed2.append(strong_augmented1)
        strong_augmented2 = self.transform_list[3](img)
        # img_transformed2.append(strong_augmented2)
        # strong_augmented = torch.stack(img_transformed2)


        if self.target_transform is not None:
            target = self.target_transform(target)

        return weak_augmented, target, index, strong_augmented1, strong_augmented2

    def __len__(self):
        return len(self.imgs)