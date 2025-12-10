import torch
from torch.utils.data import DataLoader
import numpy as np

from nets import IndustrialMAC_HeteroGNN
from training import IndustrialDataset
from wirelessNetwork import build_hetero_graph
from resource_grid import plot_node_time_scheduling, plot_ap_time_assignment, plot_ofdm_grid, plot_ofdm_time_frequency_window, plot_all_doppler_windows
from data_generation import NUM_AP, FREQ_SUBCARRIERS, DOPPLER_HZ
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===============================================================
# Metrics for hard model
# ===============================================================

def scheduling_accuracy(pred_sched, true_sched):
    """
    pred_sched: [N,T] binary (0/1) — obtained from ap_onehot_final
    true_sched: [N,T] binary
    """
    correct = (pred_sched == true_sched).float().mean().item()
    return correct


def ap_accuracy(pred_ap_onehot, true_ap, true_sched):
    """
    pred_ap_onehot: [N, T, A] one-hot (hard)
    true_ap:         [N, T]     AP assigned in the dataset
    true_sched:      [N, T]     active slots in the dataset
    """

    # pred AP = index of the one-hot
    pred_labels = pred_ap_onehot.argmax(dim=-1)   # [N, T]

    # Consider only slots where the node actually transmits
    mask = true_sched > 0.5

    pred_filtered = pred_labels[mask]   # [K]
    true_filtered = true_ap[mask]       # [K]

    if pred_filtered.numel() == 0:
        return 0.0

    return (pred_filtered == true_filtered).float().mean().item()


def collision_score(pred_sched, num_ap=NUM_AP):
    """
    In a MAC system with multiple APs (NUM_AP),
    the ideal scheduling has NUM_AP transmissions per slot.

    Collision score = average deviation from the ideal value.
    """

    # pred_sched: [N,T]
    per_slot = pred_sched.sum(dim=0)   # [T]
    # Difference from ideal value (num_ap)
    diff = torch.abs(per_slot - num_ap)
    return diff.mean().item()
    

# ===============================================================
# Testing loop
# ===============================================================

def test_model(
        model_path="model_epoch_30.pth",
        data_dir="data",
        num_samples=20):

    print(f"Loading model from: {model_path}")

    model = IndustrialMAC_HeteroGNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    dataset = IndustrialDataset(data_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    sched_acc_list = []
    ap_acc_list = []
    coll_list = []

    with torch.no_grad():
        for i, batch in enumerate(loader):

            if i >= num_samples:
                break

            device_pos = batch["device_pos"].squeeze(0).to(DEVICE)
            ap_pos     = batch["ap_pos"].squeeze(0).to(DEVICE)
            csi        = batch["csi"].squeeze(0).to(DEVICE)

            true_sched = batch["schedule"].squeeze(0).to(DEVICE)
            true_ap    = batch["ap_assign"].squeeze(0).to(DEVICE)

            # Graph construction
            g = build_hetero_graph(device_pos.cpu().numpy(), ap_pos.cpu().numpy())
            g = g.to(DEVICE)

            # Forward
            pred_sched, pred_ap_onehot = model(g, device_pos, ap_pos, csi)

            # Metrics
            sa = scheduling_accuracy(pred_sched, true_sched)
            aa = ap_accuracy(pred_ap_onehot, true_ap, true_sched)
            cs = collision_score(pred_sched, num_ap=pred_ap_onehot.shape[-1])

            sched_acc_list.append(sa)
            ap_acc_list.append(aa)
            coll_list.append(cs)

            print(f"[Sample {i}]")
            print(f"  Scheduling accuracy: {sa:.4f}")
            print(f"  AP accuracy:         {aa:.4f}")
            print(f"  Collision score:     {cs:.4f}")
            
            # Visualizations for the first sample
            if i<4:
                plot_all_doppler_windows(pred_sched, pred_ap_onehot, FREQ_SUBCARRIERS, DOPPLER_HZ)

    # Final report
    print("\n==================== FINAL RESULTS ====================")
    print(f"Scheduling accuracy average: {np.mean(sched_acc_list):.4f}")
    print(f"AP accuracy average:         {np.mean(ap_acc_list):.4f}")
    print(f"Collision score average:     {np.mean(coll_list):.4f}")
    print("==========================================================")

    


    return {
        "sched_acc": np.mean(sched_acc_list),
        "ap_acc":   np.mean(ap_acc_list),
        "collision": np.mean(coll_list)
    }


if __name__ == "__main__":
    test_model()
