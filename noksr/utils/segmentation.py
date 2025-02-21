import torch
from noksr.utils.serialization import encode

def segment_and_generate_encoder_outputs(batch, model, device, segment_num, grid_size=0.01, serial_order='z'):
    """
    Segment input `xyz` of the batch, generate encoder outputs for each segment, and return necessary metadata.

    Args:
        batch (dict): Batch data containing `xyz` and other input features.
        model (torch.nn.Module): The neural network model for processing data.
        device (torch.device): The device to run inference on.
        segment_num (int): Fixed number of segments.
        grid_size (float): Grid size for serialization.
        serial_order (str): Order for serialization ('z' by default).

    Returns:
        encoder_outputs (list): List of encoder outputs for each segment.
        encoding_codes (list): List of (start_code, end_code) for each segment.
        depth (int): Depth used for serialization.
    """
    # Extract necessary data from the batch
    xyz = batch['xyz'].to(torch.float32)  # Convert to float32 for processing
    total_points = xyz.shape[0]

    # Compute segment length based on fixed number of segments
    segment_length = max(1, total_points // segment_num)

    in_quant_coords = torch.floor(xyz / grid_size).to(torch.int)
    depth = int(torch.abs(in_quant_coords).max()).bit_length()  # Calculate serialization depth
    in_quant_codes = encode(
        in_quant_coords,
        torch.zeros(in_quant_coords.shape[0], dtype=torch.int64, device=in_quant_coords.device),
        depth,
        order=serial_order
    )
    in_sorted_quant_codes, in_sorted_indices = torch.sort(in_quant_codes)

    segments = []
    encoding_codes = []
    for i in range(segment_num):
        start_idx = i * segment_length
        end_idx = min(start_idx + segment_length, total_points)
        segment_indices = in_sorted_indices[start_idx:end_idx]
        segments.append(segment_indices)

        # Store the start and end encoding codes for the segment
        start_code = in_sorted_quant_codes[start_idx].item()
        end_code = in_sorted_quant_codes[end_idx - 1].item()
        encoding_codes.append(start_code)

    # Generate encoder outputs for each segment
    encoder_outputs = []
    for segment_indices in segments:
        # Create a new batch for the current segment
        segment_batch = {
            "xyz": batch['xyz'][segment_indices],
            "point_features": batch['point_features'][segment_indices],
            "scene_names": batch['scene_names'],
            "xyz_splits": torch.tensor([len(segment_indices)], device=batch['xyz'].device)
        }

        pt_data = {
            'feat': segment_batch['point_features'],
            'offset': segment_batch['xyz_splits'],  # Offset for the segment
            'grid_size': grid_size,
            'coord': segment_batch['xyz']
        }
        segment_encoder_output = model.point_transformer(pt_data)
        encoder_outputs.append(segment_encoder_output)

    return encoder_outputs, encoding_codes, depth