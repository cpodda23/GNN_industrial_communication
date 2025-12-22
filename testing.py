import torch
from torch.utils.data import DataLoader
import numpy as np

from nets import IndustrialMAC_HeteroGNN
from training import IndustrialDataset
from wirelessNetwork import build_hetero_graph
from data_generation import NUM_AP, FREQ_SUBCARRIERS, TIME_SLOTS
from resource_grid import visualize_ofdm_grid
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===============================================================
# Metrics for hard model
# ===============================================================

def scheduling_accuracy(pred_sched, true_sched):
    """
    Calculates the scheduling accuracy as the fraction of correctly predicted scheduling decisions.
    
    pred_sched: [N,T] binary (0/1) — obtained from ap_onehot_final
    true_sched: [N,T] binary
    """
    correct = (pred_sched == true_sched).float().mean().item()
    return correct


def ap_accuracy(pred_ap_onehot, true_ap, true_sched):
    """
    Calculates the accuracy of AP assignment only on the slots where the node is scheduled to transmit.
    
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
    In a MAC system with multiple APs (NUM_AP), the ideal scheduling has NUM_AP transmissions per slot.

    Collision score = average deviation from the ideal value, calculated as:
        |(number of nodes transmitting at t) - NUM_AP|
    """

    # pred_sched: [N,T]
    per_slot = pred_sched.sum(dim=0)   # [T]
    # Difference from ideal value (num_ap)
    diff = torch.abs(per_slot - num_ap)
    return diff.mean().item()


def transmission_completion_accuracy(pred_sched, node_packets):
    """
    Checks if all packets have been transmitted by each node within the available time slots.
    A node must have completed all its packet transmissions.
    """
    completion = np.zeros_like(node_packets, dtype=np.float32)
    for n in range(len(node_packets)):
        # Check if the node has completed all its packets
        transmitted_packets = np.sum(pred_sched[n, :] > 0.5)  # Count slots where node is transmitting
        if transmitted_packets >= node_packets[n]:
            completion[n] = 1.0  # Mark as successful completion
    # Calculate the fraction of nodes that completed their transmissions
    return np.mean(completion)
    

# ===============================================================
# Testing loop
# ===============================================================

def test_model(
        model_path="model_epoch_50.pth",
        data_dir="data",
        num_samples=50):

    print(f"Loading model from: {model_path}")

    model = IndustrialMAC_HeteroGNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    dataset = IndustrialDataset(data_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    sched_acc_list = []
    ap_acc_list = []
    coll_list = []
    trans_comp_acc_list = []

    with torch.no_grad():
        for i, batch in enumerate(loader):

            if i >= num_samples:
                break

            device_pos = batch["device_pos"].squeeze(0).to(DEVICE)
            ap_pos     = batch["ap_pos"].squeeze(0).to(DEVICE)
            csi        = batch["csi"].squeeze(0).to(DEVICE)

            true_sched = batch["schedule"].squeeze(0).to(DEVICE)
            true_ap    = batch["ap_assign"].squeeze(0).to(DEVICE)
            node_packets_tensor = batch["node_packets"].squeeze(0).to(DEVICE)
            node_packets_numpy = batch["node_packets"].squeeze(0).cpu().numpy()

            # Graph construction
            g = build_hetero_graph(device_pos.cpu().numpy(), ap_pos.cpu().numpy())
            g = g.to(DEVICE)

            # Forward
            pred_sched, pred_ap_onehot, _ = model(g, device_pos, ap_pos, csi, node_packets_tensor)

            # Metrics
            sa = scheduling_accuracy(pred_sched, true_sched)
            aa = ap_accuracy(pred_ap_onehot, true_ap, true_sched)
            cs = collision_score(pred_sched, num_ap=pred_ap_onehot.shape[-1])
            trans_comp_acc = transmission_completion_accuracy(pred_sched.cpu().numpy(), node_packets_numpy)

            sched_acc_list.append(sa)
            ap_acc_list.append(aa)
            coll_list.append(cs)
            trans_comp_acc_list.append(trans_comp_acc)


            print(f"[Sample {i}]")
            print(f"  Scheduling accuracy: {sa:.4f}")
            print(f"  AP accuracy:         {aa:.4f}")
            print(f"  Collision score:     {cs:.4f}")
            print(f"  Transmission completion accuracy: {trans_comp_acc:.4f}")

            
            # Visualization for the first sample
            if i == 0: 
                print("Visualize OFDM grid of Sample 0...")
                
                # Rimuovi la dimensione del batch se presente (es. [1, 10, 40] -> [10, 40])
                sched_to_plot = pred_sched.squeeze(0) 
                logits_to_plot = pred_ap_onehot.squeeze(0)
                
                visualize_ofdm_grid(
                    sched_to_plot, 
                    logits_to_plot, 
                    num_subcarriers=FREQ_SUBCARRIERS,
                    time_slots=TIME_SLOTS
    )

    # Final report
    print("\n==================== FINAL RESULTS ====================")
    print(f"Scheduling accuracy average: {np.mean(sched_acc_list):.4f}")
    print(f"AP accuracy average:         {np.mean(ap_acc_list):.4f}")
    print(f"Collision score average:     {np.mean(coll_list):.4f}")
    print(f"Transmission completion accuracy average: {np.mean(trans_comp_acc_list):.4f}")

    print("==========================================================")

    


    return {
        "sched_acc": np.mean(sched_acc_list),
        "ap_acc":   np.mean(ap_acc_list),
        "collision": np.mean(coll_list),
        "trans_comp_acc": np.mean(trans_comp_acc_list)

    }


if __name__ == "__main__":
    test_model()
