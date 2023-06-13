import glob
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

# Dataset Metadata
RGB_MEAN = [0.51442681, 0.43435301, 0.33421855]
RGB_STD = [0.24099932, 0.246478, 0.23652802]
INPUT_SIZE = (384, 384)

TRAIN_TRANSFORMATION = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(45),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=RGB_MEAN, std=RGB_STD)
])

VAL_TRANSFORMATION = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=RGB_MEAN, std=RGB_STD)
])


class ImageFolderDataset(Dataset):
    """Strutured Image Folder Dataset

    Parameters
    ----------
    root_dir : str
        Path to image root directory
    transform : Optional[torchvision.transforms.Compose], optional
        Data augmentation pipeline, by default None
    """

    def __init__(self, root_dir: str, stage: Optional[str] = None, transform: Optional[transforms.Compose] = None):
        if stage is not None:
            self.root_dir = os.path.join(root_dir, stage)
        else:
            self.root_dir = root_dir
        if not os.path.exists(root_dir):
            raise RuntimeError(f'Path to dataset is not valid')
        self.labels_name = os.listdir(self.root_dir)
        self.labels_name.sort()
        self.list_images =  glob.glob(f'{self.root_dir}/**/*.jpg')
        self.transform = transform

    def __len__(self) -> int:
        """Get number of images."""
        return len(self.list_images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get sample for the current idx."""
        img_path = self.list_images[idx]

        # NOTE: cv2 read image as BGR.
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert class name to label 0-10
        # NOTE: root_path/class_name/image_file.jpg
        class_name = img_path.split('/')[-2]
        label_index = self.labels_name.index(class_name)
        label_index = [label_index]

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # If no transformations, convert to C,H,W
        # Normalize to [0,1]
        if isinstance(image, np.ndarray):
            image = np.transpose(image, (1, 2, 0))
            image = torch.from_numpy()
            image /= 255.0

        return image, torch.tensor(label_index, dtype=torch.int64)


class Food101LitDatamodule(LightningDataModule):
    """LightningDataModule for Food101 Data Pipeline.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html

    Parameters
    ----------
    data_dir : str, optional
        FiftyOne dataset directory, by default 'data/'
    input_size : List[int], optional
        Input model size, by default [384, 384]
    batch_size : int, optional
        Number of training batch size, by default 64
    num_workers : int, optional
        Number of worksers to process data, by default 0
    pin_memory : bool, optional
        Enable memory pinning, by default False
    """

    def __init__(
        self,
        data_dir: str = 'data/',
        input_size: Tuple[int, int] = (384, 384),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        """Get number of classes."""
        return 10

    def setup(self, stage: Optional[str] = None):
        """Load the data with specified stage."""
        if stage in ['train', 'fit', None] and self.data_train is None:
            self.data_train = ImageFolderDataset(
                root_dir=self.hparams.data_dir, stage='train', transform=TRAIN_TRANSFORMATION)
            if len(self.data_train) == 0:
                raise ValueError('Train dataset is empty.')
        if stage in ['validation', 'test', 'fit', None]:
            if self.data_val is None:
                self.data_val = ImageFolderDataset(
                    root_dir=self.hparams.data_dir, stage='valid', transform=VAL_TRANSFORMATION)
                if len(self.data_val) == 0:
                    raise ValueError('Validation dataset is empty.')
            if self.data_test is None:
                self.data_test = ImageFolderDataset(
                    root_dir=self.hparams.data_dir, stage='valid', transform=VAL_TRANSFORMATION)
                if len(self.data_test) == 0:
                    raise ValueError('Test dataset is empty.')
        if stage == 'predict':
            if self.data_test is None:
                self.data_predict = ImageFolderDataset(
                    root_dir=self.hparams.data_dir, transform=VAL_TRANSFORMATION)
                if len(self.data_predict) == 0:
                    raise ValueError('Predict dataset is empty.')

    def train_dataloader(self):
        """Get train dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        """Get validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        """Get test dataloader."""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
