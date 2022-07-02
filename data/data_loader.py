import cv2
import os
import pandas as pd
import numpy as np
from utils.utils import make_mask, get_labels
import albumentations as albu
from torch.utils.data import Dataset


class CloudDataset(Dataset):
    def __init__(self, df: pd.DataFrame = None, path="./dataset", datatype: str = 'train', img_ids: np.array = None,
                preprocessing=None):
        self.df = df
        if datatype == 'train':
            self.data_folder = f"{path}/train_images"
            self.transforms = self.train_augmentations()
        elif datatype == "valid":
            self.data_folder = f"{path}/train_images"
            self.transforms = self.test_augmentations()
        elif datatype == "test":
            self.data_folder = f"{path}/test_images"
            self.transforms = self.test_augmentations()
        else:
            raise Exception("Invalid datatype")

        self.img_ids = img_ids
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        mask = make_mask(self.df, image_name)
        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img = preprocessed['image']
            mask = preprocessed['mask']
        return img, mask

    def __len__(self):
        return len(self.img_ids)

    def train_augmentations(self):
        train_transform = [
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.ShiftScaleRotate(scale_limit=0.20, rotate_limit=10, shift_limit=0.1, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
            albu.GridDistortion(p=0.5),
            albu.Resize(320, 640),
            albu.Normalize(),
        ]
        return albu.Compose(train_transform)

    def test_augmentations(self):
        test_transform = [
            albu.Resize(320, 640),
            albu.Normalize(),
        ]
        return albu.Compose(test_transform)


class CloudDatasetNoResize(Dataset):
    def __init__(self, df: pd.DataFrame = None, path="./dataset", datatype: str = 'train', img_ids: np.array = None,
                preprocessing=None):
        self.df = df
        if datatype == 'train':
            self.data_folder = f"{path}/train_images"
            self.transforms = self.train_augmentations()
        elif datatype == "valid":
            self.data_folder = f"{path}/train_images"
            self.transforms = self.test_augmentations()
        elif datatype == "test":
            self.data_folder = f"{path}/test_images"
            self.transforms = self.test_augmentations()
        else:
            raise Exception("Invalid datatype")

        self.img_ids = img_ids
        self.preprocessing = preprocessing
        self.resize_image = self.image_aug()
        self.resize_mask = self.mask_aug()

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        mask = make_mask(self.df, image_name)
        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        resized = self.resize_image(image=img)
        img = resized["image"]
        mask = augmented['mask']
        resized = self.resize_mask(image=mask)
        mask = resized["image"]
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img = preprocessed['image']
            mask = preprocessed['mask']
        return img, mask

    def __len__(self):
        return len(self.img_ids)

    def train_augmentations(self):
        train_transform = [
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.ShiftScaleRotate(scale_limit=0.20, rotate_limit=10, shift_limit=0.1, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
            albu.GridDistortion(p=0.5),
            albu.Normalize(),
        ]
        return albu.Compose(train_transform)

    def test_augmentations(self):
        test_transform = [
            albu.Normalize(),
        ]
        return albu.Compose(test_transform)

    def image_aug(self):
        return albu.Compose([albu.Resize(1408, 2176)])

    def mask_aug(self):
        return albu.Compose([albu.Resize(352, 544)])


class CloudDatasetClassification(Dataset):
    def __init__(self, df: pd.DataFrame = None, path="./dataset", datatype: str = 'train', img_ids: np.array = None,
                preprocessing=None):
        self.df = df
        if datatype == 'train':
            self.data_folder = f"{path}/train_images"
            self.transforms = self.train_augmentations()
        elif datatype == "valid":
            self.data_folder = f"{path}/train_images"
            self.transforms = self.test_augmentations()
        elif datatype == "test":
            self.data_folder = f"{path}/test_images"
            self.transforms = self.test_augmentations()
        else:
            raise Exception("Invalid datatype")

        self.img_ids = img_ids
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        labels = get_labels(self.df, image_name)
        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.transforms(image=img)
        img = augmented['image']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img)
            img = preprocessed['image']
        return img, labels

    def __len__(self):
        return len(self.img_ids)

    def train_augmentations(self):
        train_transform = [
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.ShiftScaleRotate(scale_limit=0.20, rotate_limit=10, shift_limit=0.1, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
            albu.GridDistortion(p=0.5),
            albu.Resize(320, 640),
            albu.Normalize(),
        ]
        return albu.Compose(train_transform)

    def test_augmentations(self):
        test_transform = [
            albu.Resize(320, 640),
            albu.Normalize(),
        ]
        return albu.Compose(test_transform)


class CloudDatasetTTA(Dataset):
    def __init__(self, df: pd.DataFrame = None, path="./dataset", datatype: str = 'train', img_ids: np.array = None,
                preprocessing=None):
        self.df = df
        if datatype == "valid":
            self.data_folder = f"{path}/train_images"
            self.transforms = self.test_augmentations()
        elif datatype == "test":
            self.data_folder = f"{path}/test_images"
            self.transforms = self.test_augmentations()
        else:
            raise Exception("Invalid datatype")

        self.img_ids = img_ids
        self.preprocessing = preprocessing
        self.h_flip = albu.HorizontalFlip(p=1)
        self.v_flip = albu.VerticalFlip(p=1)

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        mask = make_mask(self.df, image_name)
        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']

        img_hflip = self.h_flip(image=img)["image"]
        img_vflip = self.v_flip(image=img)["image"]

        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img = preprocessed['image']
            img_hflip = self.preprocessing(image=img_hflip, mask=mask)["image"]
            img_vflip = self.preprocessing(image=img_vflip, mask=mask)["image"]
            mask = preprocessed['mask']
        images = {
            "img":img,
            "img_hflip": img_hflip,
            "img_vflip": img_vflip
        }
        return images, mask

    def __len__(self):
        return len(self.img_ids)

    def test_augmentations(self):
        test_transform = [
            albu.Resize(320, 640),
            albu.Normalize(),
        ]
        return albu.Compose(test_transform)
