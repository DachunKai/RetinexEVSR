import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
import time
from basicsr.archs.retinexevsr_arch import RetinexEVSR


def test_retinexevsr():
    # Define model configuration
    net = RetinexEVSR(
        dim=32,
        num_blocks=5
    ).cuda()
    net.eval()  # Set the model to evaluation mode

    # Prepare input data
    t = 10  # Number of frames in the input sequence
    lqs = torch.rand(1, t, 3, 256, 320).cuda()  # Low-quality frames
    voxels = torch.rand(1, t, 5, 256, 320).cuda()  # Corresponding event voxels

    # Use thop to calculate MACs and number of parameters
    total_macs, _ = profile(model=net, inputs=(lqs, voxels), verbose=False)
    params = sum(p.numel() for p in net.parameters())

    # Calculate MACs per frame
    macs_per_frame = total_macs / t

    # Measure inference time
    repeat_time = 10  # Total number of repetitions
    warm_up = 5  # Number of warm-up iterations
    infer_time = 0  # Total inference time

    with torch.no_grad():
        for i in range(repeat_time):
            start_time = time.time()
            output = net(lqs, voxels)
            if i >= warm_up:  # Start accumulating inference time after warm-up iterations
                infer_time += (time.time() - start_time)

    # Calculate average inference time per frame
    avg_infer_time_per_frame = infer_time / (repeat_time - warm_up + 1) / t

    # Print results
    print(f'Output Shape: {output.shape}')
    print(f'GMACs per Frame: {macs_per_frame / 10**9:.2f} G')
    print(f'Params: {params / 10**6:.2f} M')
    print(f'Average Inference Time per Frame: {avg_infer_time_per_frame:.4f} s')


if __name__ == '__main__':
    test_retinexevsr()
