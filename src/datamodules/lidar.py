# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:43:26 2023

@author: cleli
"""

import logging
from src.datamodules.base import BaseDataModule
from src.datasets.graphdataset import MyOwnDataset
from torch.utils.data import DataLoader

import pytorch_lightning as pl


log = logging.getLogger(__name__)


class LidarDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, val_dir, test_dir, batch_size=32):
        super(LidarDataModule, self).__init__()
        self.batch_size = batch_size
        self.train_dataset = MyOwnDataset(root=train_dir)
        self.val_dataset = MyOwnDataset(root=val_dir)
        self.test_dataset = MyOwnDataset(root=test_dir)
 
    def setup(self, stage=None):
        # Pas besoin de diviser manuellement les données, car elles sont déjà dans des dossiers séparés.
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

train_dir = "C:/Users/cleli/OneDrive/Documents/PIR/datasets/train"
val_dir = "C:/Users/cleli/OneDrive/Documents/PIR/datasets/val"
test_dir = "C:/Users/cleli/OneDrive/Documents/PIR/datasets/test"

# Instantiate the data module
    data_module = LidarDataModule(train_dir=train_dir, val_dir=val_dir, test_dir=test_dir, batch_size=32)
   

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = str(pyrootutils.setup_root(__file__, pythonpath=True))
    cfg = omegaconf.OmegaConf.load(root + "/configs/datamodule/lidar.yaml")
    cfg.data_dir = root + "/data"
    _ = hydra.utils.instantiate(cfg)


