from typing import Dict, Optional

import torch
import numpy as np
import pytorch_lightning as pl
from yacs.config import CfgNode

from .dataset import Dataset
from .human import HumanData
from .robot import RobotData



def to_lower(x: Dict) -> Dict:
    """
    Convert all dictionary keys to lowercase
    Args:
      x (dict): Input dictionary
    Returns:
      dict: Output dictionary with all keys converted to lowercase
    """
    return {k.lower(): v for k, v in x.items()}


def create_dataset(
    cfg: CfgNode, dataset_cfg: CfgNode, train: bool = True, **kwargs
) -> Dataset:
    """
    Instantiate a dataset from a config file.
    Args:
        cfg (CfgNode): Model configuration file.
        dataset_cfg (CfgNode): Dataset configuration info.
        train (bool): Variable to select between train and val datasets.
    """

    dataset_type = Dataset.registry[dataset_cfg.TYPE]
    print(dataset_cfg)
    return dataset_type(cfg, **to_lower(dataset_cfg), train=train, **kwargs)



class MixedDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: CfgNode, dataset_cfg: CfgNode, train: bool = True) -> None:
        dataset_list = cfg.DATASETS.TRAIN if train else cfg.DATASETS.VAL
        datasets = [
            create_dataset(cfg, dataset_cfg[dataset], train=train)
            for dataset, v in dataset_list.items()
        ]
        self.datasets = datasets

        weights = np.array([v.WEIGHT for dataset, v in dataset_list.items()])
        self.weights = weights / weights.sum()  # normalize
        self.lengths = [len(d) for d in datasets]
        self.total_length = sum(self.lengths)

    def __len__(self):
        return max(self.total_length, 1000)

    def __getitem__(self, idx):
        dataset_idx = np.random.choice(len(self.datasets), p=self.weights)
        return self.datasets[dataset_idx][idx % self.lengths[dataset_idx]]


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg: CfgNode, dataset_cfg: CfgNode) -> None:
        """
        Initialize LightningDataModule for HAMER training
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode containing necessary dataset info.
            dataset_cfg (CfgNode): Dataset configuration file
        """
        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load datasets necessary for training
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode containing necessary dataset info.
        """
        shuffle = 250
        epoch = 5_000
        if self.train_dataset is None:
            self.train_dataset = MixedDataset(self.cfg, self.dataset_cfg, train=True)
            self.val_dataset = MixedDataset(self.cfg, self.dataset_cfg, train=False)

    def train_dataloader(self) -> Dict:
        """
        Setup training data loader.
        Returns:
            Dict: Dictionary containing image and mocap data dataloaders
        """
        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            self.cfg.bs,
            drop_last=True,
            shuffle=True,
            num_workers=10,
            prefetch_factor=2,
            pin_memory=True,
        )
        return train_dataloader

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Setup val data loader.
        Returns:
            torch.utils.data.DataLoader: Validation dataloader
        """
        val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            self.cfg.test_bs,
            drop_last=True,
            num_workers=10,
        )
        return val_dataloader    