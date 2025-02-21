import os
import numpy as np
import torch
from pathlib import Path
import shutil
from noksr.utils.serialization import encode
from tqdm import tqdm

def _serial_scene(scene, segment_length, segment_num, grid_size=0.01, serial_order='z'):
    # Process ground truth points
    in_xyz = torch.from_numpy(scene['in_xyz']).to(torch.float32)
    in_quant_coords = torch.floor(in_xyz / grid_size).to(torch.int)
    gt_xyz = torch.from_numpy(scene['gt_xyz']).to(torch.float32)
    gt_quant_coords = torch.floor(gt_xyz / grid_size).to(torch.int)

    depth = int(max(gt_quant_coords.max(), in_quant_coords.max())).bit_length()
    gt_quant_codes = encode(gt_quant_coords, torch.zeros(gt_quant_coords.shape[0], dtype=torch.int64, device=gt_quant_coords.device), depth, order=serial_order)
    gt_sorted_quant_codes, gt_sorted_indices = torch.sort(gt_quant_codes)

    total_gt_points = len(gt_sorted_quant_codes)

    in_quant_codes = encode(in_quant_coords, torch.zeros(in_quant_coords.shape[0], dtype=torch.int64, device=in_quant_coords.device), depth, order=serial_order)
    in_sorted_quant_codes, in_sorted_indices = torch.sort(in_quant_codes)

    # Segment ground truth points and find corresponding input points
    segments = []
    for i in range(segment_num):
        # Ground truth segmentation
        gt_start_idx = i * segment_length
        gt_end_idx = gt_start_idx + segment_length
        gt_end_idx = min(gt_end_idx, total_gt_points)
        gt_segment_indices = gt_sorted_indices[gt_start_idx:gt_end_idx]

        # Input segmentation within the ground truth range
        in_segment_indices = in_sorted_indices[(in_sorted_quant_codes >= gt_sorted_quant_codes[gt_start_idx].item()) & 
                                               (in_sorted_quant_codes < gt_sorted_quant_codes[min(gt_end_idx, total_gt_points - 1)].item())]
        
        segments.append({
            "gt_indices": gt_segment_indices.numpy(),
            "in_indices": in_segment_indices.numpy()
        })

    return segments

def update_lst_files(input_drive_path, output_drive_path, segmented_scenes):
    """
    Update .lst files in the output directory by expanding entries for segmented scenes.

    Args:
        input_drive_path (Path): Path to the input drive directory.
        output_drive_path (Path): Path to the output drive directory.
        segmented_scenes (dict): Dictionary of original scenes to their segment counts (e.g., {"rgn0000": 12}).
    """
    lst_files = ['test.lst', 'testall.lst', 'train.lst', 'val.lst']

    for lst_file in lst_files:
        input_lst_file = input_drive_path / lst_file
        output_lst_file = output_drive_path / lst_file

        if not input_lst_file.exists():
            continue  # Skip if the .lst file doesn't exist in the input

        with input_lst_file.open('r') as infile, output_lst_file.open('w') as outfile:
            for line in infile:
                scene_name = line.strip()  # Original scene name without extension
                if scene_name in segmented_scenes:
                    # Expand scene into its segments
                    num_segments = segmented_scenes[scene_name]
                    for i in range(num_segments):
                        outfile.write(f"{scene_name.split('-')[0]}-crop{(i):04d}\n")
                else:
                    # Write the scene as-is if it's not segmented
                    outfile.write(line)
                    
def process_dataset(input_path, output_path, fixed_segment_length):
    input_base = Path(input_path)
    output_base = Path(output_path)
    output_base.mkdir(parents=True, exist_ok=True)
    target_drives = ['Town01-0', 'Town01-1', 'Town01-2',
                     'Town02-0', 'Town02-1', 'Town02-2',
                     'Town10-0', 'Town10-1', 'Town10-2', 'Town10-3', 'Town10-4']

    # Copy list files (e.g., test.lst, val.lst)
    for file in input_base.glob('*.lst'):
        shutil.copy(file, output_base / file.name)

    # Count total number of scenes for progress bar initialization
    total_scenes = sum(
        1
        for drive in input_base.iterdir()
        if drive.is_dir() and drive.name in target_drives
        for item in drive.iterdir()
        if item.is_dir()
    )

    # Process each drive folder with progress bar
    with tqdm(total=total_scenes, desc="Processing Scenes", unit="scene") as pbar:
        for drive in input_base.iterdir():
            if drive.is_dir() and drive.name in target_drives:
                drive_output = output_base / drive.name
                drive_output.mkdir(parents=True, exist_ok=True)
                segmented_scenes = {} 

                for item in drive.iterdir():
                    if item.is_dir():
                        scene_name = item.name
                        output_scene_base = drive_output

                        # Load data
                        data_file = item / 'pointcloud.npz'
                        gt_file = item / 'groundtruth.bin'
                        gt_data = np.load(gt_file, allow_pickle=True)
                        data = np.load(data_file)
                        scene = {
                            'in_xyz': data['points'],
                            'in_normal': data['normals'],
                            'gt_xyz': gt_data['xyz'],
                            'gt_normal': gt_data['normal']
                        }

                        total_gt_points = len(scene['gt_xyz'])

                        # Step 1: Compute segment number
                        segment_num = max(1, total_gt_points // fixed_segment_length)

                        # Step 2: Recompute segment length for even distribution
                        segment_length = total_gt_points // segment_num

                        # Serialize and segment scene
                        segments = _serial_scene(scene, segment_length, segment_num)
                        segmented_scenes[scene_name] = len(segments)  # Record segment count for this scene
                        for i, segment in enumerate(segments):
                            segment_scene_name = f"{scene_name.split('-')[0]}-crop{(i):04d}"
                            segment_output = output_scene_base / segment_scene_name
                            segment_output.mkdir(parents=True, exist_ok=True)

                            # Save segmented data
                            segment_data = {
                                'points': scene['in_xyz'][segment["in_indices"]],
                                'normals': scene['in_normal'][segment["in_indices"]],
                                'ref_xyz': scene['gt_xyz'][segment["gt_indices"]],
                                'ref_normals': scene['gt_normal'][segment["gt_indices"]]
                            }
                            np.savez(segment_output / 'pointcloud.npz', **segment_data)

                        # Update progress bar
                        pbar.update(1)

                update_lst_files(drive, drive_output, segmented_scenes)

if __name__ == "__main__":
    """ 
    This script is used to regenerate the training segments by 1-d serialization, original Carla patches are uniformly sampled.
    """
    input_path = "./data/carla-lidar/dataset-no-patch"  # Replace with your input dataset path
    output_path = "./data/carla-lidar/dataset-seg-patch"  # Replace with your desired output path
    fixed_segment_length = 300000  # Fixed segment length to start computation

    process_dataset(input_path, output_path, fixed_segment_length)