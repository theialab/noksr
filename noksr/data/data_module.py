from importlib import import_module
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Sampler, DistributedSampler, Dataset
import pytorch_lightning as pl
from arrgh import arrgh


class DataModule(pl.LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        self.dataset = getattr(import_module('noksr.data.dataset'), data_cfg.data.dataset)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_set = self.dataset(self.data_cfg, "train")
            self.val_set = self.dataset(self.data_cfg, "val")
        if stage == "test" or stage is None:
            self.val_set = self.dataset(self.data_cfg, self.data_cfg.model.inference.split)
        if stage == "predict" or stage is None:
            self.test_set = self.dataset(self.data_cfg, "test")

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.data_cfg.data.batch_size, shuffle=True, pin_memory=True,
                          collate_fn=_sparse_collate_fn, num_workers=self.data_cfg.data.num_workers, drop_last=True)

    def val_dataloader(self):          
        return DataLoader(self.val_set, batch_size=1, pin_memory=True, collate_fn=_sparse_collate_fn,
                          num_workers=self.data_cfg.data.num_workers)

    def test_dataloader(self):
        return DataLoader(self.val_set, batch_size=1, pin_memory=True, collate_fn=_sparse_collate_fn,
                          num_workers=self.data_cfg.data.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=1, pin_memory=True, collate_fn=_sparse_collate_fn,
                          num_workers=self.data_cfg.data.num_workers)


def _sparse_collate_fn(batch):
    if "gt_geometry" in batch[0]:
        """ for dataset with ground truth geometry """
        data = {}
        xyz = []
        point_features = []
        gt_geometry_list = []
        scene_names_list = []

        for _, b in enumerate(batch):
            scene_names_list.append(b["scene_name"])
            xyz.append(torch.from_numpy(b["xyz"]))
            point_features.append(torch.from_numpy(b["point_features"]))
            gt_geometry_list.append(b["gt_geometry"])

        data['xyz'] = torch.cat(xyz, dim=0)
        data['point_features'] = torch.cat(point_features, dim=0)
        data['xyz_splits'] = torch.tensor([c.shape[0] for c in xyz])
        data['gt_geometry'] = gt_geometry_list

        data['scene_names'] = scene_names_list

        return data

    else:
        data = {}
        xyz = []
        all_xyz = []
        all_normals = []
        point_features = []
        scene_names_list = []
        for _, b in enumerate(batch):
            scene_names_list.append(b["scene_name"])
            xyz.append(torch.from_numpy(b["xyz"]))
            all_xyz.append(torch.from_numpy(b["all_xyz"]))
            all_normals.append(torch.from_numpy(b["all_normals"]))
            point_features.append(torch.from_numpy(b["point_features"]))

        data['all_xyz'] = torch.cat(all_xyz, dim=0)
        data['all_normals'] = torch.cat(all_normals, dim=0)
        data['xyz'] = torch.cat(xyz, dim=0)
        data['point_features'] = torch.cat(point_features, dim=0)

        data['scene_names'] = scene_names_list
        data['row_splits'] = [c.shape[0] for c in all_xyz]
        data['xyz_splits'] = torch.tensor([c.shape[0] for c in xyz])
        if "gt_onet_sample" in batch[0]:
            data['gt_onet_sample'] = [b["gt_onet_sample"] for b in batch]
        return data



