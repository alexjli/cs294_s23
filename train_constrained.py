import os

import torch
from torch.utils.data import DataLoader
import tqdm
import numpy as np

from src.data import CATHDataset
from src.diffusion import WhitenedDiffuser_ConstrainedChain
from src.model import SE3Denoiser


def loop(diffuser, dataloader, optimizer=None, test=False):
    grad_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(not test)
    if test:
        diffuser.eval()
    else:
        diffuser.train()
    loss_list = []
    for data in (pbar := tqdm.tqdm(dataloader)):
        data = data[0]
        coords = torch.tensor(data["coords"]).to(device)  # N x 4 x 3

        # center at 0
        is_nan_node = coords.isnan().any(dim=-1).any(dim=-1)  # N
        coords = coords - coords[~is_nan_node].view([-1, 3]).mean(dim=0)

        if coords.shape[0] < 60:
            continue
        loss, ts = diffuser.denoising_loss(coords, num_sample)
        if not test:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_list.append(loss.item())
        pbar.set_description(f"Avg Loss {np.mean(loss_list):.4g}, Point loss {loss.item():.4g}, ts {ts.tolist()}")

    torch.set_grad_enabled(grad_enabled)
    return loss_list


if __name__ == '__main__':
    num_epoch = 10
    num_sample = 1
    device = "cpu"
    device = "cuda:2"
    device = "cuda:0"

    model_dir = os.path.join("models","20230429_constrained_1")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("Loading CATH dataset")
    cath = CATHDataset(path="data/chain_set.jsonl",
                       splits_path="data/chain_set_splits.json")
    train_dataloader = DataLoader(cath.train, batch_size=1, collate_fn=lambda x: x)
    val_dataloader = DataLoader(cath.val, batch_size=1, collate_fn=lambda x: x)
    test_dataloader = DataLoader(cath.test, batch_size=1, collate_fn=lambda x: x)

    model = SE3Denoiser()
    diffuser = WhitenedDiffuser_ConstrainedChain(model, a=1, r=1)
    diffuser.to(device)

    optimizer = torch.optim.Adam(diffuser.parameters())

    for n in range(num_epoch):
        print("Epoch", n)
        train_loss_list = loop(diffuser, train_dataloader, optimizer)
        print(f"Train loss: {np.mean(train_loss_list)}")
        val_loss_list = loop(diffuser, val_dataloader, optimizer, test=True)
        print(f"Val loss: {np.mean(val_loss_list)}")

        checkpoint = {
            "diffuser_state_dict": diffuser.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": np.mean(train_loss_list),
            "val_loss": np.mean(val_loss_list)
        }
        torch.save(checkpoint, os.path.join(model_dir, f"checkpoint_{n}.pt"))
