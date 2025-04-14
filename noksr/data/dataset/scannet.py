import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from noksr.utils.serialization import encode
import open3d as o3d

class Scannet(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = 'val' if self.cfg.data.over_fitting else split # use only val set for overfitting
        if 'ScanNet' in cfg.data: # subset of mixture data
            self.dataset_root_path = cfg.data.ScanNet.dataset_path
            self.dataset_path = cfg.data.ScanNet.dataset_path
        else:
            self.dataset_root_path = cfg.data.dataset_path
            self.dataset_path = cfg.data.dataset_path

        self.metadata = cfg.data.metadata
        self.num_input_points = cfg.data.num_input_points
        self.take = cfg.data.take
        self.intake_start = cfg.data.intake_start
        self.uniform_sampling = cfg.data.uniform_sampling
        self.input_splats = cfg.data.input_splats
        self.std_dev = cfg.data.std_dev

        self.in_memory = cfg.data.in_memory
        self.dataset_split = "test" if split == "test" else "train" # train and val scenes and all under train set
        self.data_map = {
            "train": self.metadata.train_list,
            "val": self.metadata.val_list,
            "test": self.metadata.test_list
        }
        self._load_from_disk()    

    def _load_from_disk(self):
        with open(getattr(self.metadata, f"{self.split}_list")) as f:
            self.scene_names = [line.strip() for line in f]
        
        self.scenes = []
        if self.cfg.data.over_fitting:
            self.scene_names = self.scene_names[self.intake_start:self.take+self.intake_start]
            if len(self.scene_names) == 1: # if only one scene is taken, overfit on scene 0221_00
                self.scene_names = ['scene0221_00']
        for scene_name in tqdm(self.scene_names, desc=f"Loading {self.split} data from disk"):
            scene_path = os.path.join(self.dataset_path, self.split, f"{scene_name}.pth")
            scene = torch.load(scene_path)
            scene["xyz"] = scene["xyz"].astype(np.float32)
            scene["rgb"] = scene["rgb"].astype(np.float32)
            scene['scene_name'] = scene_name
            point_features = np.zeros(shape=(len(scene['xyz']), 0), dtype=np.float32)
            if self.cfg.model.network.use_color:
                point_features = np.concatenate((point_features, scene['rgb']), axis=1)
            if self.cfg.model.network.use_normal:
                point_features = np.concatenate((point_features, scene['normal']), axis=1)
            if self.cfg.model.network.use_xyz:
                point_features = np.concatenate((point_features, scene['xyz']), axis=1)  # add xyz to point features
            scene['point_features'] = point_features
            self.scenes.append(scene) 
    
    def __len__(self):
        return len(self.scenes)
    
    def __getitem__(self, idx):
        scene = self.scenes[idx]
        all_xyz = scene['xyz']
        all_normals = scene['normal']
        scene_name = scene['scene_name']

        # sample input points
        num_points = scene["xyz"].shape[0]
        num_input_points = self.num_input_points
        if num_input_points == -1:
            xyz = scene["xyz"]
            point_features = scene['point_features']
        else:
            if not self.uniform_sampling:
                # Number of blocks along each axis
                num_blocks = 2
                total_blocks = num_blocks ** 3
                self.common_difference = 200
                # Calculate block sizes
                block_sizes = (all_xyz.max(axis=0) - all_xyz.min(axis=0)) / num_blocks

                # Create the number_per_block array with an arithmetic sequence
                average_points_per_block = num_input_points // total_blocks
                number_per_block = np.array([
                    average_points_per_block + (i - total_blocks // 2) * self.common_difference
                    for i in range(total_blocks)
                ])
                
                # Adjust number_per_block to ensure the sum is num_input_points
                total_points = np.sum(number_per_block)
                difference = num_input_points - total_points
                number_per_block[-1] += difference

                # Sample points from each block
                sample_indices = []
                block_index = 0
                total_chosen_indices = 0
                remaining_points = 0  # Points to be added to the next block
                for i in range(num_blocks):
                    for j in range(num_blocks):
                        for k in range(num_blocks):
                            block_min = all_xyz.min(axis=0) + block_sizes * np.array([i, j, k])
                            block_max = block_min + block_sizes
                            block_mask = np.all((all_xyz >= block_min) & (all_xyz < block_max), axis=1)
                            block_indices = np.where(block_mask)[0]
                            num_samples = number_per_block[block_index] + remaining_points
                            remaining_points = 0  # Reset remaining points
                            block_index += 1
                            if len(block_indices) > 0:
                                chosen_indices = np.random.choice(block_indices, num_samples, replace=True)
                                sample_indices.extend(chosen_indices)
                                total_chosen_indices += len(chosen_indices)
                                # print(f"Block {block_index} - Desired: {num_samples}, Actual: {len(chosen_indices)}")
                                if len(chosen_indices) < num_samples:
                                    remaining_points += (num_samples - len(chosen_indices))
                            else:
                                # print(f"Block {block_index} - No points available. Adding {num_samples} points to the next block.")
                                remaining_points += num_samples
                
            else:
                if num_points < num_input_points:
                    print(f"Scene {scene_name} has less than {num_input_points} points. Sampling with replacement.")
                    sample_indices = np.random.choice(num_points, num_input_points, replace=True)
                else:
                    sample_indices = np.random.choice(num_points, num_input_points, replace=True)
    
        xyz = scene["xyz"][sample_indices]
        point_features = scene['point_features'][sample_indices]
        noise = np.random.normal(0, self.std_dev, xyz.shape)
        xyz += noise
        
        data = {
            "all_xyz": all_xyz,
            "all_normals": all_normals,
            "xyz": xyz,  # N, 3
            "point_features": point_features,  # N, 3
            "scene_name": scene['scene_name']
        }

        return data

