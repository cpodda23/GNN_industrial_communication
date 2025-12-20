import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from nets import IndustrialMAC_HeteroGNN
from wirelessNetwork import build_hetero_graph

EPOCHES = 50
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
            "ap_assign": sample["ap_assign"],        # [N_dev,T]
            "node_packets": sample["node_packets"]   # [N_dev]
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
        total_comp_loss = 0.0
        total_ce_loss = 0.0

        for batch in loader:

            # ------------------------------
            # Loading tensors
            # ------------------------------
            device_pos  = batch["device_pos"].squeeze(0).to(device)   # [N,2]
            ap_pos      = batch["ap_pos"].squeeze(0).to(device)       # [A,2]
            csi         = batch["csi"].squeeze(0).to(device)          # [N,A,F,T,2]

            target_sched = batch["schedule"].squeeze(0).to(device)    # [N,T]
            target_ap    = batch["ap_assign"].squeeze(0).to(device)   # [N,T]
            node_packets = batch["node_packets"].squeeze(0).to(device)   # [N]

            # ------------------------------
            # Graph construction
            # ------------------------------
            g = build_hetero_graph(device_pos.cpu().numpy(),
                                   ap_pos.cpu().numpy())
            g = g.to(device)

            # ------------------------------
            # Forward
            # ------------------------------
            pred_sched_hard, pred_ap_logits, pred_sched_soft = model(g, device_pos, ap_pos, csi, node_packets)

            N, T, A = pred_ap_logits.shape

            # ------------------------------
            # LOSS 1: Cross Entropy (which AP is assigned)
            # ------------------------------
            logits = pred_ap_logits.reshape(N*T, A)
            targets = target_ap.reshape(N*T)

            # Consider only slots where the node actually transmits
            mask = target_sched.reshape(N*T) > 0.5
            
            loss_ce = torch.tensor(0.0, device=device)
            if mask.sum() > 0:
                loss_ce = F.cross_entropy(logits[mask], targets[mask].long())

            # ------------------------------
            # LOSS 2: Completion Loss (how many slots allocated vs packets to send)
            # ------------------------------
            loss_comp = calculate_completion_loss(pred_sched_soft, node_packets)
            
            
            # calculate loss with weighted sum of both losses
            loss = loss_ce + 0.25 * loss_comp
            
            # ------------------------------
            # Backprop
            # ------------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate total losses by summing over every batch
            total_loss += loss.item()
            total_ce_loss += loss_ce.item()
            total_comp_loss += loss_comp.item()
            
        print("\n====================  Total losses (sum on all batches) ====================")
        print(f"[Epoch {epoch+1}/{epochs}] Loss = {total_loss:.4f} | CE: {total_ce_loss:.4f} | Comp: {total_comp_loss:.4f}")

        # ------------------------------
        # Checkpoint every 5 epochs
        # ------------------------------
        if (epoch+1) % 5 == 0:
            path = f"model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), path)
            print(f"Saved model: {path}")

        # for the last epoch print its losses for 1 sample only
        if epoch == epochs - 1:
                print("\n====================  Losses on single sample ====================")
                print(f"[Epoch {epoch+1}/{epochs}] Loss = {total_loss/len(loader):.4f} | CE: {total_ce_loss/len(loader):.4f} | Comp: {total_comp_loss/len(loader):.4f}")

    print("\nTraining completed.\n")
    return model

# ===============================================================
# Transmission Completion Loss Calculation
# ===============================================================

def calculate_completion_loss(pred_sched_soft, node_packets):

    # Calculate the capacity allocated by summing probabilities over Time slots
    allocated_capacity = torch.sum(pred_sched_soft, dim=1) # [N]
    diff = node_packets.float() - allocated_capacity

    # Penalize missing packets, if positive diff: weight = 1.0
    # Penalize extra packets if negative diff: weight = 0.1
    loss = torch.where(diff > 0, diff, diff * -0.1)
    return torch.mean(torch.abs(loss))

if __name__ == "__main__":
    train_model()
