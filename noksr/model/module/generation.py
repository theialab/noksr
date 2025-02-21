from pyexpat import features
import time

import torch
from pycg import vis
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm import tqdm
from typing import Callable, Tuple
from sklearn.neighbors import NearestNeighbors
from pytorch3d.ops import knn_points
import open3d as o3d
import pytorch_lightning as pl
from noksr.utils.samples import BatchedSampler
from noksr.utils.serialization import encode


class MeshingResult:
    def __init__(self, v: torch.Tensor = None, f: torch.Tensor = None, c: torch.Tensor = None):
        self.v = v
        self.f = f
        self.c = c

class Generator(pl.LightningModule):
    def __init__(self, model, mask_model, voxel_size, k_neighbors, last_n_layers, reconstruction_cfg):
        super().__init__()
        self.model = model # the model should be the UDF Decoder
        self.mask_model = mask_model # the distance mask decoder
        self.rec_cfg = reconstruction_cfg
        self.voxel_size = voxel_size
        self.threshold = 0.4
        self.k_neighbors = k_neighbors
        self.last_n_layers = last_n_layers                                      
    

    def compute_gt_sdf_from_pts(self, gt_xyz, gt_normals, query_pos: torch.Tensor):
        k = 8  
        stdv = 0.02
        knn_output = knn_points(query_pos.unsqueeze(0).to(torch.device("cuda")), gt_xyz.unsqueeze(0).to(torch.device("cuda")), K=k)
        indices = knn_output.idx.squeeze(0)
        indices = torch.tensor(indices, device=query_pos.device)
        closest_points = gt_xyz[indices]
        surface_to_queries_vec = query_pos.unsqueeze(1) - closest_points #N, K, 3

        dot_products = torch.einsum("ijk,ijk->ij", surface_to_queries_vec, gt_normals[indices]) #N, K
        vec_lengths = torch.norm(surface_to_queries_vec[:, 0, :], dim=-1) 
        use_dot_product = vec_lengths < stdv
        sdf = torch.where(use_dot_product, torch.abs(dot_products[:, 0]), vec_lengths)

        # Adjust the sign of the sdf values based on the majority of dot products
        num_pos = torch.sum(dot_products > 0, dim=1)
        inside = num_pos <= (k / 2)
        sdf[inside] *= -1
        
        return -sdf
    
    def generate_dual_mc_mesh(self, data_dict, encoder_outputs, device):
        from nksr.svh import SparseFeatureHierarchy, SparseIndexGrid
        from nksr.ext import meshing
        from nksr.meshing import MarchingCubes
        from nksr import utils

        max_depth = 100
        grid_upsample = 1
        mise_iter = 0
        knn_time = 0
        dmc_time = 0 
        aggregation_time = 0
        decoder_time = 0
        mask_threshold = self.rec_cfg.mask_threshold

        pts = data_dict['xyz'].detach()
        self.last_n_layers = 4
        self.trim = self.rec_cfg.trim
        self.gt_mask = self.rec_cfg.gt_mask
        self.gt_sdf = self.rec_cfg.gt_sdf
        # Generate DMC grid structure
        nksr_svh = SparseFeatureHierarchy(
            voxel_size=self.voxel_size,
            depth=self.last_n_layers,
            device= pts.device
        )
        nksr_svh.build_point_splatting(pts)

        flattened_grids = []
        for d in range(min(nksr_svh.depth, max_depth + 1)):
            f_grid = meshing.build_flattened_grid(
                nksr_svh.grids[d]._grid,
                nksr_svh.grids[d - 1]._grid if d > 0 else None,
                d != nksr_svh.depth - 1
            )
            if grid_upsample > 1:
                f_grid = f_grid.subdivided_grid(grid_upsample)
            flattened_grids.append(f_grid)

        dual_grid = meshing.build_joint_dual_grid(flattened_grids)
        dmc_graph = meshing.dual_cube_graph(flattened_grids, dual_grid)
        dmc_vertices = torch.cat([
            f_grid.grid_to_world(f_grid.active_grid_coords().float())
            for f_grid in flattened_grids if f_grid.num_voxels() > 0
        ], dim=0)
        del flattened_grids, dual_grid
        """ create a mask to trim spurious geometry """

        decoder_time -= time.time()
        dmc_value, sdf_knn_time, sdf_aggregation_time = self.model(encoder_outputs, dmc_vertices)
        decoder_time += time.time()
        knn_time += sdf_knn_time
        aggregation_time += sdf_aggregation_time
        if self.gt_sdf:
            if 'gt_geometry' in data_dict:
                ref_xyz, ref_normal, _ = data_dict['gt_geometry'][0].torch_attr()
            else:
                ref_xyz, ref_normal = data_dict['all_xyz'], data_dict['all_normals']
            dmc_value = self.compute_gt_sdf_from_pts(ref_xyz, ref_normal, dmc_vertices)

        for _ in range(mise_iter):
            cube_sign = dmc_value[dmc_graph] > 0
            cube_mask = ~torch.logical_or(torch.all(cube_sign, dim=1), torch.all(~cube_sign, dim=1))
            dmc_graph = dmc_graph[cube_mask]
            unq, dmc_graph = torch.unique(dmc_graph.view(-1), return_inverse=True)
            dmc_graph = dmc_graph.view(-1, 8)
            dmc_vertices = dmc_vertices[unq]
            dmc_graph, dmc_vertices = utils.subdivide_cube_indices(dmc_graph, dmc_vertices)
            dmc_value = torch.clamp(self.model(encoder_outputs, dmc_vertices.to(device)), max=self.threshold)

        dmc_time -= time.time()
        dual_v, dual_f = MarchingCubes().apply(dmc_graph, dmc_vertices, dmc_value)
        dmc_time += time.time()

        vert_mask = None
        if self.trim:
            if self.gt_mask: 
                nn = NearestNeighbors(n_neighbors=1)
                nn.fit(data_dict['all_xyz'].cpu().numpy())  # coords is an (N, 3) array
                dist, indx = nn.kneighbors(dual_v.detach().cpu().numpy())  # xyz is an (M, 3) array
                dist = torch.from_numpy(dist).to(dual_v.device).squeeze(-1)
                vert_mask = dist < mask_threshold
            else: 
                decoder_time -= time.time()
                dist, mask_knn_time, mask_aggregation_time = self.mask_model(encoder_outputs, dual_v.to(device))
                decoder_time += time.time()
                vert_mask = dist < mask_threshold
                knn_time += mask_knn_time
                aggregation_time += mask_aggregation_time
            dmc_time -= time.time()
            dual_v, dual_f = utils.apply_vertex_mask(dual_v, dual_f, vert_mask)
            dmc_time += time.time()

        dmc_time -= time.time()
        mesh_res =  MeshingResult(dual_v, dual_f, None)
        # del dual_v, dual_f
        mesh = vis.mesh(mesh_res.v, mesh_res.f)
        dmc_time += time.time()

        time_dict = {
            'neighboring_time': knn_time,
            'dmc_time': dmc_time,
            'aggregation_time': aggregation_time,
            'decoder_time': decoder_time,
        }
        return mesh, time_dict
    
    def generate_dual_mc_mesh_by_segment(self, data_dict, encoder_outputs, encoding_codes, depth, device):
        """ 
        This function generates a dual marching cube mesh by computing the sdf values for each segment individually.
        """
        from nksr.svh import SparseFeatureHierarchy, SparseIndexGrid
        from nksr.ext import meshing
        from nksr.meshing import MarchingCubes
        from nksr import utils

        max_depth = 100
        grid_upsample = 1
        mise_iter = 0
        knn_time = 0
        dmc_time = 0
        aggregation_time = 0
        decoder_time = 0
        mask_threshold = self.rec_cfg.mask_threshold

        pts = data_dict['xyz'].detach()
        self.last_n_layers = 4
        self.trim = self.rec_cfg.trim
        self.gt_mask = self.rec_cfg.gt_mask
        self.gt_sdf = self.rec_cfg.gt_sdf

        # Generate DMC grid structure
        nksr_svh = SparseFeatureHierarchy(
            voxel_size=self.voxel_size,
            depth=self.last_n_layers,
            device=pts.device
        )
        nksr_svh.build_point_splatting(pts)

        flattened_grids = []
        for d in range(min(nksr_svh.depth, max_depth + 1)):
            f_grid = meshing.build_flattened_grid(
                nksr_svh.grids[d]._grid,
                nksr_svh.grids[d - 1]._grid if d > 0 else None,
                d != nksr_svh.depth - 1
            )
            if grid_upsample > 1:
                f_grid = f_grid.subdivided_grid(grid_upsample)
            flattened_grids.append(f_grid)

        dual_grid = meshing.build_joint_dual_grid(flattened_grids)
        dmc_graph = meshing.dual_cube_graph(flattened_grids, dual_grid)
        dmc_vertices = torch.cat([
            f_grid.grid_to_world(f_grid.active_grid_coords().float())
            for f_grid in flattened_grids if f_grid.num_voxels() > 0
        ], dim=0)
        del flattened_grids, dual_grid

        # Encode and segment `dmc_vertices`
        in_quant_coords = torch.floor(dmc_vertices / 0.01).to(torch.int)
        dmc_quant_codes = encode(
            in_quant_coords,
            torch.zeros(in_quant_coords.shape[0], dtype=torch.int64, device=in_quant_coords.device),
            depth,
            order='z'
        )
        sorted_codes, sorted_indices = torch.sort(dmc_quant_codes)

        dmc_value_list = []
        for idx in range(len(encoding_codes)):
            if idx == 0:
                segment_mask = (sorted_codes < encoding_codes[idx+1])
            elif idx == len(encoding_codes) - 1:
                segment_mask = (sorted_codes >= encoding_codes[idx])
            else:
                segment_mask = (sorted_codes >= encoding_codes[idx]) & (sorted_codes < encoding_codes[idx+1])
            segment_indices = sorted_indices[segment_mask]
            segment_vertices = dmc_vertices[segment_indices]
            segment_encoder_output = encoder_outputs[idx]
            segment_dmc_value, sdf_knn_time, sdf_aggregation_time = self.model(segment_encoder_output, segment_vertices)
            dmc_value_list.append(segment_dmc_value)

            knn_time += sdf_knn_time
            aggregation_time += sdf_aggregation_time

        dmc_values = torch.zeros_like(sorted_codes, dtype=torch.float32, device=device)
        dmc_values[sorted_indices] = torch.cat(dmc_value_list)

        dmc_time -= time.time()
        dual_v, dual_f = MarchingCubes().apply(dmc_graph, dmc_vertices, dmc_values)
        dmc_time += time.time()

        """ create a mask to trim spurious geometry """
        in_quant_coords = torch.floor(dual_v / 0.01).to(torch.int)
        dual_quant_codes = encode(
            in_quant_coords,
            torch.zeros(in_quant_coords.shape[0], dtype=torch.int64, device=in_quant_coords.device),
            depth,
            order='z'
        )
        sorted_dual_codes, sorted_dual_indices = torch.sort(dual_quant_codes)

        dist_list = []
        for idx in range(len(encoding_codes)):
            if idx == 0:
                segment_mask = (sorted_dual_codes < encoding_codes[idx+1])
            elif idx == len(encoding_codes) - 1:
                segment_mask = (sorted_dual_codes >= encoding_codes[idx])
            else:
                segment_mask = (sorted_dual_codes >= encoding_codes[idx]) & (sorted_dual_codes < encoding_codes[idx+1])
            segment_indices = sorted_dual_indices[segment_mask]
            segment_dual_v = dual_v[segment_indices]

            if self.gt_mask:
                nn = NearestNeighbors(n_neighbors=1)
                nn.fit(data_dict['all_xyz'].cpu().numpy())  # Reference points (N, 3)
                segment_dist, _ = nn.kneighbors(segment_dual_v.detach().cpu().numpy())  # Query points (M, 3)
                segment_dist = torch.from_numpy(segment_dist).to(dual_v.device).squeeze(-1)
            else:
                decoder_time -= time.time()
                segment_dist, mask_knn_time, mask_aggregation_time = self.mask_model(encoder_outputs[idx], segment_dual_v.to(device))
                decoder_time += time.time()

                knn_time += mask_knn_time
                aggregation_time += mask_aggregation_time

            dist_list.append(segment_dist)

        dist = torch.zeros_like(sorted_dual_codes, dtype=torch.float32, device=device)
        dist[sorted_dual_indices] = torch.cat(dist_list)

        vert_mask = dist < mask_threshold
        dual_v, dual_f = utils.apply_vertex_mask(dual_v, dual_f, vert_mask)

        dmc_time -= time.time()
        mesh_res = MeshingResult(dual_v, dual_f, None)
        mesh = vis.mesh(mesh_res.v, mesh_res.f)
        dmc_time += time.time()

        time_dict = {
            'neighboring_time': knn_time,
            'dmc_time': dmc_time,
            'aggregation_time': aggregation_time,
            'decoder_time': decoder_time,
        }

        return mesh, time_dict
