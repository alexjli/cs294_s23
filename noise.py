import os

import torch
from torch.utils.data import DataLoader
import tqdm
import numpy as np

from src.data import CATHDataset
from src.diffusion import WhitenedDiffuser_IdealChain, WhitenedDiffuser_ConstrainedChain
from src.model import SE3Denoiser
from src.utils import save_fused_pdbs


if __name__ == '__main__':
    num_epoch = 10
    num_sample = 2
    device = "cpu"
    device = "cuda:0"

    print("Loading CATH dataset")
    cath = CATHDataset(path="data/chain_set.jsonl",
                       splits_path="data/chain_set_splits.json")
    train_dataloader = DataLoader(cath.train, batch_size=1, collate_fn=lambda x: x)
    val_dataloader = DataLoader(cath.val, batch_size=1, collate_fn=lambda x: x)
    test_dataloader = DataLoader(cath.test, batch_size=1, collate_fn=lambda x: x)

    model_dir = os.path.join("models", "20230429_0")
    checkpoint = torch.load(os.path.join(model_dir, "checkpoint_5.pt"))
    model = SE3Denoiser()
    diffuser = WhitenedDiffuser_ConstrainedChain(model,
                                                 a=1,
                                                 r=1)
    # diffuser = WhitenedDiffuser_IdealChain(model, r_gamma=1)
    diffuser.load_state_dict(checkpoint["diffuser_state_dict"])
    diffuser.to(device)

    for data in train_dataloader:
        data = data[0]
        coords = torch.tensor(data["coords"]).to(device)  # N x 4 x 3
        if coords.shape[0] < 60:
            continue

        # center at 0
        is_nan_node = coords.isnan().any(dim=-1).any(dim=-1)  # N
        if is_nan_node.any():
            continue
        coords = coords - coords[~is_nan_node].view([-1, 3]).mean(dim=0)
        print(data["name"])

        steps = 50
        ts = (torch.arange(steps) / steps).to(device).unsqueeze(-1)
        print(coords.shape)
        print(ts.shape)

        noised_samples, ts = diffuser.forward_diffusion(coords.to(device), num_samples=steps, ts=ts)
        xyzs = [sample.view([-1, 3]).tolist() for sample in torch.unbind(noised_samples, dim=0)]
        save_fused_pdbs(xyzs, pdb_out=os.path.join(f"noising_constrained.pdb"))

        break
