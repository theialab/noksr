import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
from plyfile import PlyData

class SceneNN(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.dataset_root_path = cfg.data.dataset_path
        self.voxel_size = cfg.data.voxel_size
        self.num_input_points = cfg.data.num_input_points
        self.std_dev = cfg.data.std_dev
        self.intake_start = cfg.data.intake_start
        self.take = cfg.data.take
        self.input_splats = cfg.data.input_splats

        self.in_memory = cfg.data.in_memory
        self.train_files = cfg.data.train_files
        self.test_files = cfg.data.test_files
        if self.split == 'test':
            scene_ids = self.test_files
        else:
            scene_ids = self.train_files + self.test_files
        self.filenames = sorted([os.path.join(self.dataset_root_path, scene_id, f) 
                               for scene_id in scene_ids
                               for f in os.listdir(os.path.join(self.dataset_root_path, scene_id))
                               if f.endswith('.ply')])
        if self.cfg.data.over_fitting:
            self.filenames = self.filenames[self.intake_start:self.take+self.intake_start]

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        """Get item."""
        # load the mesh
        scene_filename = self.filenames[idx]

        ply_data = PlyData.read(scene_filename)
        vertex = ply_data['vertex']
        pos = np.stack([vertex[t] for t in ('x', 'y', 'z')], axis=1)
        nls = np.stack([vertex[t] for t in ('nx', 'ny', 'nz')], axis=1) if 'nx' in vertex and 'ny' in vertex and 'nz' in vertex else np.zeros_like(pos)

        all_xyz = pos
        all_normals = nls
        scene_name = os.path.basename(scene_filename).replace('.ply', '')

        all_point_features = np.zeros(shape=(len(all_xyz), 0), dtype=np.float32)
        if self.cfg.model.network.use_normal:
            all_point_features = np.concatenate((all_point_features, all_normals), axis=1)
        if self.cfg.model.network.use_xyz:
            all_point_features = np.concatenate((all_point_features, all_xyz), axis=1)  # add xyz to point features

        # sample input points
        num_points = all_xyz.shape[0]
        if self.num_input_points == -1:
            xyz = all_xyz
            point_features = all_point_features
            normals = all_normals
        else:
            sample_indices = np.random.choice(num_points, self.num_input_points, replace=True)
            xyz = all_xyz[sample_indices]
            point_features = all_point_features[sample_indices]
            normals = all_normals[sample_indices]

        noise = np.random.normal(0, self.std_dev, xyz.shape)
        xyz += noise

        data = {
            "all_xyz": all_xyz,
            "all_normals": all_normals,
            "xyz": xyz,  # N, 3
            "normals": normals,  # N, 3
            "point_features": point_features,  # N, 3
            "scene_name": scene_name
        }

        return data
