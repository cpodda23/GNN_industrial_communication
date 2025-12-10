import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from nets import IndustrialMAC_HeteroGNN
from wirelessNetwork import build_hetero_graph


EPOCHES = 30
BATCH_SIZE = 1
LEARNING_RATE = 1e-3

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
            "csi": sample["csi"],                    # [N_dev,N_ap,F,T,2]
            "schedule": sample["schedule"],          # [N_dev,T]
            "ap_assign": sample["ap_assign"]         # [N_dev,T]
        }


# ===============================================================
# Training Loop
# ===============================================================

def train_model(
        data_dir="data",
        lr=LEARNING_RATE,
        epochs=EPOCHES,
        batch_size=BATCH_SIZE,
        device="cuda" if torch.cuda.is_available() else "cpu"):

    dataset = IndustrialDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = IndustrialMAC_HeteroGNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Training on {device}...\n")

    for epoch in range(epochs):
        total_loss = 0.0

        for batch in loader:

            # ------------------------------
            # Loading tensors
            # ------------------------------
            device_pos  = batch["device_pos"].squeeze(0).to(device)   # [N,2]
            ap_pos      = batch["ap_pos"].squeeze(0).to(device)       # [A,2]
            csi         = batch["csi"].squeeze(0).to(device)          # [N,A,F,T,2]

            target_sched = batch["schedule"].squeeze(0).to(device)    # [N,T]
            target_ap    = batch["ap_assign"].squeeze(0).to(device)   # [N,T]

            # ------------------------------
            # Graph construction
            # ------------------------------
            g = build_hetero_graph(device_pos.cpu().numpy(),
                                   ap_pos.cpu().numpy())
            g = g.to(device)

            # ------------------------------
            # Forward
            # ------------------------------
            pred_sched_hard, pred_ap_logits = model(g, device_pos, ap_pos, csi)
            # pred_ap_logits : [N, T, A]  (differentiable logits)

            N, T, A = pred_ap_logits.shape

            # ------------------------------
            # PREPARATION FOR CROSS-ENTROPY
            # ------------------------------
            logits = pred_ap_logits.reshape(N*T, A)
            targets = target_ap.reshape(N*T)

            # Consider only slots where the node actually transmits
            mask = target_sched.reshape(N*T) > 0.5

            logits = logits[mask]        # [K, A]
            targets = targets[mask]      # [K]

            if logits.numel() == 0:
                loss = torch.tensor(0.0, device=device)
            else:
                loss = F.cross_entropy(logits, targets.long())

            # ------------------------------
            # Backprop
            # ------------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}/{epochs}] Loss = {total_loss:.4f}")

        # ------------------------------
        # Checkpoint every 5 epochs
        # ------------------------------
        if (epoch+1) % 5 == 0:
            path = f"model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), path)
            print(f"Saved model: {path}")

    print("\nTraining completed.\n")
    return model


if __name__ == "__main__":
    train_model()
