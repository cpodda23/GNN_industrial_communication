# ===============================================================
# training.py — Training per IndustrialMAC_HeteroGNN
# ===============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from nets import IndustrialMAC_HeteroGNN
from wirelessNetwork import build_hetero_graph


# ===============================================================
# Dataset Loader
# ===============================================================
class IndustrialDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir="data"):
        self.files = sorted([f"{data_dir}/{f}" for f in os.listdir(data_dir) if f.endswith(".pt")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = torch.load(self.files[idx])

        return {
            "device_pos": sample["nodes_pos"],       # [N_dev,2]
            "ap_pos": sample["ap_pos"],              # [N_ap,2]
            "csi": sample["csi"],                    # [N_dev,N_ap,F,T]
            "schedule": sample["schedule"],          # [N_dev,T]
            "ap_assign": sample["ap_assign"]         # [N_dev]
        }


# ===============================================================
# Training Loop
# ===============================================================
def train_model(
        data_dir="data",
        lr=1e-3,
        epochs=20,
        batch_size=1,
        device="cuda" if torch.cuda.is_available() else "cpu"):

    dataset = IndustrialDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = IndustrialMAC_HeteroGNN().to(device)

    # Loss functions
    sched_loss_fn = nn.BCELoss()
    ap_loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Training on {device}...")

    for epoch in range(epochs):
        total_loss = 0.0

        for batch in loader:

            # Caricamento tensori
            device_pos = batch["device_pos"].squeeze(0).to(device)   # [N_dev,2]
            ap_pos = batch["ap_pos"].squeeze(0).to(device)           # [N_ap,2]
            csi = batch["csi"].squeeze(0).to(device)                 # [N_dev,N_ap,F,T]

            target_sched = batch["schedule"].squeeze(0).to(device)   # [N_dev,T]
            target_ap = batch["ap_assign"].squeeze(0).to(device)     # [N_dev]

            # Costruzione grafo eterogeneo
            g = build_hetero_graph(device_pos.cpu().numpy(), ap_pos.cpu().numpy())
            g = g.to(device)

            # Forward
            pred_sched, pred_ap = model(g, device_pos, ap_pos, csi)

            # Loss
            loss_sched = sched_loss_fn(pred_sched, target_sched)
            loss_ap = ap_loss_fn(pred_ap, target_ap)
            loss = loss_sched + loss_ap

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}/{epochs}] Loss = {total_loss:.4f}")

        # Checkpoint
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
            print(f"Salvato modello: model_epoch_{epoch+1}.pth")

    print("Training completato.")
    return model


# ===============================================================
# Main
# ===============================================================
if __name__ == "__main__":
    train_model()
