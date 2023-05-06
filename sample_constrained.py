import os

import torch
import tqdm

from src.diffusion import WhitenedDiffuser_ConstrainedChain
from src.model import SE3Denoiser
from src.utils import save_fused_pdbs


if __name__ == '__main__':
    num_sample = 10
    device = "cpu"
    device = "cuda:3"

    model_dir = os.path.join("models", "20230429_constrained_1")
    sample_dir = os.path.join(model_dir, "samples")
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)
    checkpoint = torch.load(os.path.join(model_dir, "checkpoint_3.pt"))
    model = SE3Denoiser()
    diffuser = WhitenedDiffuser_ConstrainedChain(model)
    diffuser.load_state_dict(checkpoint["diffuser_state_dict"])
    diffuser.to(device)

    trajectory, one_shots = diffuser.sample(60, steps=500, return_trajectory=True)
    xyzs = [sample.view([-1, 3]).tolist() for sample in trajectory]
    save_fused_pdbs(xyzs, pdb_out=os.path.join("constrained_trajectory_2.pdb"))
