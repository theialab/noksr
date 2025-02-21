# import ext

import torch
from nksr.svh import SparseFeatureHierarchy
from pytorch3d.ops import knn_points
from scipy.spatial import KDTree


import torch

class Sampler:
    def __init__(self, **kwargs):
        # Default values can be set with kwargs.get('key', default_value)
        self.voxel_size = kwargs.get('voxel_size')
        self.adaptive_policy = {
            'method': 'normal',
            'tau': 0.1,
            'depth': 2
        }
        self.cfg = kwargs.get('cfg')
        self.ref_xyz = kwargs.get('ref_xyz')
        self.ref_normal = kwargs.get('ref_normal')
        self.svh = self._build_gt_svh()
        self.kdtree = KDTree(self.ref_xyz.detach().cpu().numpy())

    def _build_gt_svh(self):
        gt_svh = SparseFeatureHierarchy(
            voxel_size=self.voxel_size,
            depth=self.cfg.svh_tree_depth,
            device=self.ref_xyz.device
        )
        if self.adaptive_policy['method'] == "normal":
            gt_svh.build_adaptive_normal_variation(
                self.ref_xyz, self.ref_normal,
                tau=self.adaptive_policy['tau'],
                adaptive_depth=self.adaptive_policy['depth']
            )
        return gt_svh
    
    def _get_svh_samples(self, svh, n_samples, expand=0, expand_top=0):
        """
        Get random samples, across all layers of the decoder hierarchy
        :param svh: SparseFeatureHierarchy, hierarchy of spatial features
        :param n_samples: int, number of total samples
        :param expand: int, size of expansion
        :param expand_top: int, size of expansion of the coarsest level.
        :return: (n_samples, 3) tensor of positions
        """
        base_coords, base_scales = [], []
        for d in range(svh.depth):
            if svh.grids[d] is None:
                continue
            ijk_coords = svh.grids[d].active_grid_coords()
            d_expand = expand if d != svh.depth - 1 else expand_top
            if d_expand >= 3:
                mc_offsets = torch.arange(-d_expand // 2 + 1, d_expand // 2 + 1, device=svh.device)
                mc_offsets = torch.stack(torch.meshgrid(mc_offsets, mc_offsets, mc_offsets, indexing='ij'), dim=3)
                mc_offsets = mc_offsets.view(-1, 3)
                ijk_coords = (ijk_coords.unsqueeze(dim=1).repeat(1, mc_offsets.size(0), 1) +
                              mc_offsets.unsqueeze(0)).view(-1, 3)
                ijk_coords = torch.unique(ijk_coords, dim=0)
            base_coords.append(svh.grids[d].grid_to_world(ijk_coords.float()))
            base_scales.append(torch.full((ijk_coords.size(0),), svh.grids[d].voxel_size, device=svh.device))
        base_coords, base_scales = torch.cat(base_coords), torch.cat(base_scales)
        local_ids = (torch.rand((n_samples,), device=svh.device) * base_coords.size(0)).long()
        local_coords = (torch.rand((n_samples, 3), device=svh.device) - 0.5) * base_scales[local_ids, None]
        query_pos = base_coords[local_ids] + local_coords
        return query_pos

    def _get_samples(self):
        all_samples = []
        for config in self.cfg.samplers:
            if config.type == "uniform":
                all_samples.append(
                    self._get_svh_samples(self.svh, config.n_samples, config.expand, config.expand_top)
                )
            elif config.type == "band":
                band_inds = (torch.rand((config.n_samples, ), device=self.ref_xyz.device) * self.ref_xyz.size(0)).long()
                eps = config.eps * self.voxel_size
                band_pos = self.ref_xyz[band_inds] + \
                    self.ref_normal[band_inds] * torch.randn((config.n_samples, 1), device=self.ref_xyz.device) * eps
                all_samples.append(band_pos)
            elif config.type == 'on_surface':
                n_subsample = config.subsample
                if 0 < n_subsample < self.ref_xyz.size(0):
                    ref_xyz_inds = (torch.rand((n_subsample,), device=self.ref_xyz.device) *
                                    self.ref_xyz.size(0)).long()
                else:
                    ref_xyz_inds = (torch.rand((n_subsample,), device=self.ref_xyz.device) *
                                    self.ref_xyz.size(0)).long()
                all_samples.append(self.ref_xyz[ref_xyz_inds])

        return torch.cat(all_samples, 0)

    def transform_field(self, field: torch.Tensor):
        sdf_config = self.cfg
        assert sdf_config.gt_type != "binary"
        truncation_size = sdf_config.gt_band * self.voxel_size
        if sdf_config.gt_soft:
            field = torch.tanh(field / truncation_size) * truncation_size
        else:
            field = torch.clone(field)
            field[field > truncation_size] = truncation_size
            field[field < -truncation_size] = -truncation_size
        return field

    def compute_gt_sdf_from_pts(self, query_pos: torch.Tensor):
        k = 8  
        stdv = 0.02
        normals = self.ref_normal
        device = query_pos.device
        knn_output = knn_points(query_pos.unsqueeze(0).to(device), self.ref_xyz.unsqueeze(0).to(device), K=k)
        indices = knn_output.idx.squeeze(0)
        closest_points = self.ref_xyz[indices]
        surface_to_queries_vec = query_pos.unsqueeze(1) - closest_points #N, K, 3

        dot_products = torch.einsum("ijk,ijk->ij", surface_to_queries_vec, normals[indices]) #N, K
        vec_lengths = torch.norm(surface_to_queries_vec[:, 0, :], dim=-1) 
        use_dot_product = vec_lengths < stdv
        sdf = torch.where(use_dot_product, torch.abs(dot_products[:, 0]), vec_lengths)

        # Adjust the sign of the sdf values based on the majority of dot products
        num_pos = torch.sum(dot_products > 0, dim=1)
        inside = num_pos <= (k / 2)
        sdf[inside] *= -1
        
        return -sdf

    
class BatchedSampler:
    def __init__(self, hparams):
        self.hparams = hparams

    def batch_sdf_sample(self, data_dict):
        if 'gt_geometry' in data_dict:
            gt_geometry = data_dict['gt_geometry']
        else:
            xyz, normal = data_dict['all_xyz'], data_dict['all_normals']
            row_splits = data_dict['row_splits']
        batch_size = len(data_dict['scene_names'])

        start = 0
        batch_samples_pos = []
        batch_gt_sdf = []

        for i in range(batch_size):
            if 'gt_geometry' in data_dict:
                ref_xyz, ref_normal, _ = gt_geometry[i].torch_attr()
            else:
                end = start + row_splits[i]
                ref_xyz = xyz[start:end]
                ref_normal = normal[start:end]
                start = end

            # Instantiate Sampler using kwargs
            sampler = Sampler(
                voxel_size=self.hparams.data.voxel_size,
                cfg=self.hparams.data.supervision.sdf,
                ref_xyz=ref_xyz,
                ref_normal=ref_normal
            )
            samples_pos = sampler._get_samples()
            # nksr_sdf = sampler.compute_gt_chi_from_pts(samples_pos)
            gt_sdf = sampler.compute_gt_sdf_from_pts(samples_pos)
            if self.hparams.data.supervision.sdf.truncate:
                gt_sdf = sampler.transform_field(gt_sdf)

            batch_samples_pos.append(samples_pos)
            batch_gt_sdf.append(gt_sdf)
            
        batch_samples_pos = torch.cat(batch_samples_pos, dim=0)
        batch_gt_sdf = torch.cat(batch_gt_sdf, dim=0)

        return batch_samples_pos, batch_gt_sdf

    def batch_udf_sample(self, data_dict):
        if 'gt_geometry' in data_dict:
            gt_geometry = data_dict['gt_geometry']
        else:
            xyz, normal = data_dict['all_xyz'], data_dict['all_normals']
            row_splits = data_dict['row_splits']
        batch_size = len(data_dict['scene_names'])

        start = 0
        batch_samples_pos = []
        batch_gt_udf = []

        for i in range(batch_size):
            if 'gt_geometry' in data_dict:
                ref_xyz, ref_normal, _ = gt_geometry[i].torch_attr()
            else:
                end = start + row_splits[i]
                ref_xyz = xyz[start:end]
                ref_normal = normal[start:end]
                start = end

            # Instantiate Sampler for UDF using kwargs
            sampler = Sampler(
                voxel_size=self.hparams.data.voxel_size,
                cfg=self.hparams.data.supervision.udf,
                ref_xyz=ref_xyz,
                ref_normal=ref_normal
            )
            samples_pos = sampler._get_samples()
            if self.hparams.data.supervision.udf.abs_sdf:
                gt_udf = torch.abs(sampler.compute_gt_sdf_from_pts(samples_pos))
            else:
                knn_output = knn_points(samples_pos.unsqueeze(0).to(torch.device("cuda")),
                        ref_xyz.unsqueeze(0).to(torch.device("cuda")),
                        K=1)
                gt_udf = knn_output.dists.squeeze(0).squeeze(-1)

            batch_samples_pos.append(samples_pos)
            batch_gt_udf.append(gt_udf)
            

        batch_samples_pos = torch.cat(batch_samples_pos, dim=0)
        batch_gt_udf = torch.cat(batch_gt_udf, dim=0)

        return batch_samples_pos, batch_gt_udf
    
    def batch_on_surface_sample(self, data_dict):
        if 'gt_geometry' in data_dict:
            gt_geometry = data_dict['gt_geometry']
        else:
            xyz, normal = data_dict['all_xyz'], data_dict['all_normals']
            row_splits = data_dict['row_splits']
        batch_size = len(data_dict['scene_names'])

        start = 0
        batch_samples_pos = []
        batch_samples_normal = []
        batch_gt_udf = []

        for i in range(batch_size):
            if 'gt_geometry' in data_dict:
                ref_xyz, ref_normal, _ = gt_geometry[i].torch_attr()
            else:
                end = start + row_splits[i]
                ref_xyz = xyz[start:end]
                ref_normal = normal[start:end]
                start = end

            n_subsample = self.hparams.data.supervision.on_surface.subsample
            if 0 < n_subsample < ref_xyz.size(0):
                ref_xyz_inds = (torch.rand((n_subsample,), device=ref_xyz.device) *
                                ref_xyz.size(0)).long()
            else:
                ref_xyz_inds = (torch.rand((n_subsample,), device=ref_xyz.device) *
                                ref_xyz.size(0)).long()
            batch_samples_pos.append(ref_xyz[ref_xyz_inds])
            batch_samples_normal.append(ref_normal[ref_xyz_inds])

        batch_samples_pos = torch.cat(batch_samples_pos, dim=0)
        batch_samples_normal = torch.cat(batch_samples_normal, dim=0)
        return batch_samples_pos, batch_samples_normal


