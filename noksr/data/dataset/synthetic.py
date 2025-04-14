import os
from tqdm import tqdm
from statistics import mode
import numpy as np

from torch.utils.data import Dataset
from plyfile import PlyData


class Synthetic(Dataset):
    def __init__(self, cfg, split):
        # Attributes
        if 'Synthetic' in cfg.data: # subset of mixture data
            categories = cfg.data.Synthetic.classes
            self.dataset_folder = cfg.data.Synthetic.path
            self.multi_files = cfg.data.Synthetic.multi_files
            self.file_name = cfg.data.Synthetic.pointcloud_file
        else: 
            categories = cfg.data.classes
            self.dataset_folder = cfg.data.path
            self.multi_files = cfg.data.multi_files
            self.file_name = cfg.data.pointcloud_file
        self.scale = 2.2 # Emperical scale to transfer back to physical scale
        self.cfg = cfg
        self.split = split
        self.std_dev = cfg.data.std_dev * self.scale
        self.num_input_points = cfg.data.num_input_points

        self.no_except = True

        self.intake_start = cfg.data.intake_start
        self.take = cfg.data.take
        # If categories is None, use all subfolders
        if categories is None:
            categories = os.listdir(self.dataset_folder)
            categories = [c for c in categories
                          if os.path.isdir(os.path.join(self.dataset_folder, c))]

        self.metadata = {
            c: {'id': c, 'name': 'n/a'} for c in categories
        } 
        
        # Set index
        for c_idx, c in enumerate(categories):
            self.metadata[c]['idx'] = c_idx

        # Get all models
        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(self.dataset_folder, c)
            if not os.path.isdir(subpath):
                print('Category %s does not exist in dataset.' % c)

            if self.split is None:
                self.models += [
                    {'category': c, 'model': m} for m in [d for d in os.listdir(subpath) if (os.path.isdir(os.path.join(subpath, d)) and d != '') ]
                ]

            else:
                split_file = os.path.join(subpath, self.split + '.lst')
                with open(split_file, 'r') as f:
                    models_c = f.read().split('\n')
                
                if '' in models_c:
                    models_c.remove('')

                self.models += [
                    {'category': c, 'model': m}
                    for m in models_c
                ]
        
        # overfit in one data
        if self.cfg.data.over_fitting:
            self.models = self.models[self.intake_start:self.take+self.intake_start]

            
    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.models)
    
    def load(self, model_path, idx, vol):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            vol (dict): precomputed volume info
        '''
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))
        
        item_path = os.path.join(model_path, 'item_dict.npz')
        item_dict = np.load(item_path, allow_pickle=True)
        points_dict = np.load(file_path, allow_pickle=True)
        points = points_dict['points'] * self.scale # roughly transfer back to physical scale
        normals = points_dict['normals']
        semantics = points_dict['semantics']
        # Break symmetry if given in float16:
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            normals = normals.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)
            normals += 1e-4 * np.random.randn(*normals.shape)

        min_values = np.min(points, axis=0)
        points -= min_values

        return points, normals, semantics

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        category = self.models[idx]['category']
        model = self.models[idx]['model']
        c_idx = self.metadata[category]['idx']

        model_path = os.path.join(self.dataset_folder, category, model)
        
        all_xyz, all_normals, all_semantics = self.load(model_path, idx, c_idx)
        scene_name = f"{category}/{model}/{idx}"

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
