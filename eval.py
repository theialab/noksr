import os
import hydra
import torch
from tqdm import tqdm
import numpy as np
import open3d as o3d
from collections import defaultdict
from torch.utils.data import Dataset
from pathlib import Path
from importlib import import_module
import pytorch_lightning as pl
from noksr.data.data_module import DataModule
from noksr.utils.evaluation import UnitMeshEvaluator, MeshEvaluator
from noksr.model.module import Generator
from nksr.svh import SparseFeatureHierarchy
from noksr.utils.segmentation import segment_and_generate_encoder_outputs
                        
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    # fix the seed
    pl.seed_everything(cfg.global_test_seed, workers=True)

    output_path = os.path.join(cfg.exp_output_root_path, "reconstruction", "visualizations")
    os.makedirs(output_path, exist_ok=True)
    # output_path = cfg.exp_output_root_path

    print("==> initializing data ...")
    data_module = DataModule(cfg)
    data_module.setup("test")
    val_loader = data_module.val_dataloader()
    
    print("=> initializing model...")
    model = getattr(import_module("noksr.model"), cfg.model.network.module)(cfg)
    # Load checkpoint
    if os.path.isfile(cfg.model.ckpt_path):
        print(f"=> loading model checkpoint '{cfg.model.ckpt_path}'")
        checkpoint = torch.load(cfg.model.ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint successfully.")
    else:
        raise FileNotFoundError(f"No checkpoint found at '{cfg.model.ckpt_path}'. Please ensure the path is correct.")

    model.to(device)
    model.eval()

    if cfg.data.reconstruction.trim:
        dense_generator = Generator(
            model.sdf_decoder,
            model.mask_decoder,
            cfg.data.voxel_size,
            cfg.model.network.sdf_decoder.k_neighbors,
            cfg.model.network.sdf_decoder.last_n_layers,
            cfg.data.reconstruction
        )
    else:
        dense_generator = Generator(
            model.sdf_decoder,
            None,
            cfg.data.voxel_size,
            cfg.model.network.sdf_decoder.k_neighbors,
            cfg.model.network.sdf_decoder.last_n_layers,
            cfg.data.reconstruction
        )

    # Initialize a dictionary to keep track of sums and count
    eval_sums = defaultdict(float)
    batch_count = 0
    import time
    print("=> start inference...")
    start_time = time.time()
    total_reconstruction_duration = 0.0
    total_neighboring_time = 0.0
    total_dmc_time = 0.0
    total_aggregation_time = 0.0
    total_forward_duration = 0.0
    total_decoder_time = 0.0
    results_dict = []
    for batch in tqdm(val_loader, desc="Inference", unit="batch"):
        batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
        if 'gt_geometry' in batch:
            gt_geometry = batch['gt_geometry']
            batch['all_xyz'], batch['all_normal'], _ = gt_geometry[0].torch_attr()
        process_start = time.time()
        forward_start = time.time()
        if cfg.data.reconstruction.by_segment:
            encoder_outputs, encoding_codes, depth = segment_and_generate_encoder_outputs(
                batch=batch,
                model=model,
                device=device,
                segment_num=cfg.data.reconstruction.segment_num,  # Example: 10 segments
                grid_size=0.01,
                serial_order='z'
            )

            forward_end = time.time()
            torch.set_grad_enabled(False)
            dmc_mesh, time_dict = dense_generator.generate_dual_mc_mesh_by_segment(
                data_dict=batch,
                encoder_outputs=encoder_outputs,
                encoding_codes=encoding_codes,
                depth=depth,
                device=device
            )
        else:
            if cfg.model.network.backbone == 'PointTransformerV3':
                pt_data = {}
                pt_data['feat'] = batch['point_features']
                pt_data['offset'] = batch['xyz_splits']
                pt_data['grid_size'] = 0.01
                pt_data['coord'] = batch['xyz']
                encoder_outputs = model.point_transformer(pt_data)

            forward_end = time.time()
            torch.set_grad_enabled(False)

            dmc_mesh, time_dict  = dense_generator.generate_dual_mc_mesh(batch, encoder_outputs, device)
        total_neighboring_time += time_dict['neighboring_time']
        total_dmc_time += time_dict['dmc_time']
        total_aggregation_time += time_dict['aggregation_time']
        total_decoder_time += time_dict['decoder_time']
        # Calculate time taken for these three steps
        process_end = time.time()
        total_forward_duration += forward_end - forward_start
        process_duration = process_end - process_start
        total_reconstruction_duration += process_duration  # Accumulate the duration
        print("\nTotal Reconstruction Time: {:.2f} seconds".format(total_reconstruction_duration))
        print("├── Total PointTransformerV3 Time: {:.2f} seconds".format(total_forward_duration))
        print("├── Total Decoder Time: {:.2f} seconds".format(total_decoder_time))
        print("│   ├── Total Neighboring Time: {:.2f} seconds".format(total_neighboring_time))
        print("│   └── Total Aggregation Time: {:.2f} seconds"
            .format(total_aggregation_time))
        print("├── Total Dual Marching Cube Time: {:.2f} seconds".format(total_dmc_time))

        # Evaluate the reconstructed mesh
        if cfg.data.evaluation.evaluator == "UnitMeshEvaluator":
            evaluator = UnitMeshEvaluator(n_points=100000, metric_names=UnitMeshEvaluator.ESSENTIAL_METRICS)
        elif cfg.data.evaluation.evaluator == "MeshEvaluator":
            evaluator = MeshEvaluator(n_points=int(5e6), metric_names=MeshEvaluator.ESSENTIAL_METRICS)
        if "gt_onet_sample" in batch:
            eval_dict = evaluator.eval_mesh(dmc_mesh, batch['all_xyz'], None, onet_samples=batch['gt_onet_sample'][0])
        else:
            eval_dict = evaluator.eval_mesh(dmc_mesh, batch['all_xyz'], None)

        # eval_dict['voxel_num'] = batch['voxel_nums'][0]
        for k, v in eval_dict.items():
            eval_sums[k] += v
        scene_name = batch['scene_names']
        eval_dict["data_id"] = batch_count
        eval_dict["scene_name"] = scene_name
        results_dict.append(eval_dict)
        print(f"Scene Name: {scene_name}")
        print(f"completeness: {eval_dict['completeness']:.4f}")
        print(f"accuracy: {eval_dict['accuracy']:.4f}")
        print(f"Chamfer-L2: {eval_dict['chamfer-L2']:.4f}")
        print(f"Chamfer-L1: {eval_dict['chamfer-L1']:.4f}")
        print(f"F-Score (1.0%): {eval_dict['f-score-10']:.4f}")
        print(f"F-Score (1.5%): {eval_dict['f-score-15']:.4f}")
        print(f"F-Score (2.0%): {eval_dict['f-score-20']:.4f}")
        if "o3d-iou" in eval_dict:
            print(f"O3D-IoU: {eval_dict['o3d-iou']:.4f}")
        print()  # For better readability

        # Save the mesh
        if cfg.data.visualization.save:
            scene_name = batch['scene_names'][0].replace('/', '-')
            if cfg.data.visualization.Mesh:
                mesh_file = f"{output_path}/noksr-{cfg.data.dataset}-data_id_{cfg.data.intake_start}-{scene_name}_CD_{eval_dict['chamfer-L1']:.4f}_mesh.obj"  # or "output_mesh.obj" for OBJ format
                o3d.io.write_triangle_mesh(mesh_file, dmc_mesh)
            if cfg.data.visualization.Input_points:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(batch['xyz'].cpu().numpy())
                point_cloud_file = f"{output_path}/noksr-{cfg.data.dataset}-data_id_{cfg.data.intake_start}-{scene_name}_input_pcd.ply"  
                o3d.io.write_point_cloud(point_cloud_file, pcd)
            if cfg.data.visualization.Dense_points:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(batch['all_xyz'].cpu().numpy())
                point_cloud_file = f"{output_path}/noksr-{cfg.data.dataset}-data_id_{cfg.data.intake_start}-{scene_name}_dense_pcd.ply"
                o3d.io.write_point_cloud(point_cloud_file, pcd)

        batch_count += 1
        torch.cuda.empty_cache()

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n---Total reconstruction time without data loading: {total_reconstruction_duration:.2f} seconds")
    print(f"---Total reconstruction time including data loading for all scenes: {total_time:.2f} seconds")
    if batch_count > 0:
        print("\n--- Evaluation Metrics Averages ---")
        for k in eval_sums:
            average = eval_sums[k] / batch_count
            print(f"{k}: {average:.5f}")
    else:
        print("No batches were processed.")


if __name__ == "__main__":
    main()
