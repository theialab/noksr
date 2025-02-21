import os
from pathlib import Path
from torch.utils.data import Dataset

import numpy as np

from noksr.utils.transform import ComposedTransforms
from noksr.data.dataset.carla_gt_geometry import get_class
from noksr.data.dataset.general_dataset import DatasetSpec as DS
from noksr.data.dataset.general_dataset import RandomSafeDataset

from pycg import exp


class Carla(RandomSafeDataset):
    def __init__(self, cfg, split):
        super().__init__(cfg, split)

        self.skip_on_error = False
        self.custom_name = "carla"
        self.cfg = cfg

        self.split = split # use only train set for overfitting
        split = self.split
        self.intake_start = cfg.data.intake_start
        self.take = cfg.data.take
        self.input_splats = cfg.data.input_splats

        self.gt_type = cfg.data.supervision.gt_type

        self.transforms = ComposedTransforms(cfg.data.transforms)
        self.use_dummy_gt = False

        # If drives not specified, use all sub-folders
        base_path = Path(cfg.data.base_path)
        drives = cfg.data.drives
        if drives is None:
            drives = os.listdir(base_path)
            drives = [c for c in drives if (base_path / c).is_dir()]
        self.drives = drives
        self.input_path = cfg.data.input_path

        # Get all items
        self.all_items = []
        self.drive_base_paths = {}
        for c in drives:
            self.drive_base_paths[c] = base_path / c
            split_file = self.drive_base_paths[c] / (split + '.lst')
            with split_file.open('r') as f:
                models_c = f.read().split('\n')
            if '' in models_c:
                models_c.remove('')
            self.all_items += [{'drive': c, 'item': m} for m in models_c]

        if self.cfg.data.over_fitting:
            self.all_items = self.all_items[self.intake_start:self.take+self.intake_start]



    def __len__(self):
        return len(self.all_items)

    def get_name(self):
        return f"{self.custom_name}-cat{len(self.drives)}-{self.split}"

    def get_short_name(self):
        return self.custom_name

    def _get_item(self, data_id, rng):
        # self.num_input_points = 50000
        drive_name = self.all_items[data_id]['drive']
        item_name = self.all_items[data_id]['item']

        named_data = {}

        try:
            if self.input_path is None:
                input_data = np.load(self.drive_base_paths[drive_name] / item_name / 'pointcloud.npz')
            else:
                input_data = np.load(Path(self.input_path) / drive_name / item_name / 'pointcloud.npz')
        except FileNotFoundError:
            exp.logger.warning(f"File not found for AV dataset for {item_name}")
            raise ConnectionAbortedError

        named_data[DS.SHAPE_NAME] = "/".join([drive_name, item_name])
        named_data[DS.INPUT_PC]= input_data['points'].astype(np.float32)
        named_data[DS.TARGET_NORMAL] = input_data['normals'].astype(np.float32)

        if self.transforms is not None:
            named_data = self.transforms(named_data, rng)

        point_features = np.zeros(shape=(len(named_data[DS.INPUT_PC]), 0), dtype=np.float32)
        if self.cfg.model.network.use_normal:
            point_features = np.concatenate((point_features, named_data[DS.TARGET_NORMAL]), axis=1)
        if self.cfg.model.network.use_xyz:
            point_features = np.concatenate((point_features, named_data[DS.INPUT_PC]), axis=1)  # add xyz to point features

        xyz = named_data[DS.INPUT_PC]
        normals = named_data[DS.TARGET_NORMAL]


        geom_cls = get_class(self.gt_type)
        
        if (self.drive_base_paths[drive_name] / item_name / "groundtruth.bin").exists():
            named_data[DS.GT_GEOMETRY] = geom_cls.load(self.drive_base_paths[drive_name] / item_name / "groundtruth.bin")
            data = {
                "gt_geometry": named_data[DS.GT_GEOMETRY],
                "xyz": xyz,  # N, 3
                "normals": normals,  # N, 3
                "scene_name": named_data[DS.SHAPE_NAME],
                "point_features": point_features,  # N, K
            }
        else:
            data = {
                "all_xyz": input_data['ref_xyz'].astype(np.float32),
                "all_normals": input_data['ref_normals'].astype(np.float32),
                "xyz": xyz,  # N, 3
                "normals": normals,  # N, 3
                "scene_name": named_data[DS.SHAPE_NAME],
                "point_features": point_features,  # N, K
            }

        return data
