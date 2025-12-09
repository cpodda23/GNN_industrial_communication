import torch
from torch.utils.data import DataLoader
import numpy as np

from nets import IndustrialMAC_HeteroGNN
from training import IndustrialDataset
from wirelessNetwork import build_hetero_graph
from resource_grid import plot_node_time_scheduling, plot_ap_time_assignment, plot_ofdm_grid, plot_ofdm_time_frequency

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===============================================================
# Metriche di valutazione per modello hard
# ===============================================================

def scheduling_accuracy(pred_sched, true_sched):
    """
    pred_sched: [N,T] binario (0/1) — ottenuto da ap_onehot_final
    true_sched: [N,T] binario
    """
    correct = (pred_sched == true_sched).float().mean().item()
    return correct


def ap_accuracy(pred_ap_onehot, true_ap, true_sched):
    """
    pred_ap_onehot: [N, T, A] one-hot (hard)
    true_ap:         [N, T]     AP assegnato nel dataset
    true_sched:      [N, T]     slot attivi nel dataset
    """

    # pred AP = indice della one-hot
    pred_labels = pred_ap_onehot.argmax(dim=-1)   # [N, T]

    # Considera solo slot in cui il nodo trasmette realmente
    mask = true_sched > 0.5

    pred_filtered = pred_labels[mask]   # [K]
    true_filtered = true_ap[mask]       # [K]

    if pred_filtered.numel() == 0:
        return 0.0

    return (pred_filtered == true_filtered).float().mean().item()


def collision_score(pred_sched, num_ap=3):
    """
    In un sistema MAC con AP multipli (NUM_AP),
    lo scheduling ideale ha NUM_AP trasmissioni per slot.

    Collision score = deviazione media dal valore ideale.
    """

    # pred_sched: [N,T]
    per_slot = pred_sched.sum(dim=0)   # [T]
    # Differenza da valore ideale (num_ap)
    diff = torch.abs(per_slot - num_ap)
    return diff.mean().item()
    

# ===============================================================
# Testing loop
# ===============================================================

def test_model(
        model_path="model_epoch_30.pth",
        data_dir="data",
        num_samples=20):

    print(f"Caricamento modello da: {model_path}")

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

            # Grafo
            g = build_hetero_graph(device_pos.cpu().numpy(), ap_pos.cpu().numpy())
            g = g.to(DEVICE)

            # Forward
            pred_sched, pred_ap_onehot = model(g, device_pos, ap_pos, csi)

            # Metriche
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
            
            # Visualizzazioni per il primo sample
            # if i < 5:
            #     plot_node_time_scheduling(pred_sched, pred_ap_onehot)
            #     plot_ap_time_assignment(pred_ap_onehot)
            #     plot_ofdm_grid(pred_ap_onehot)
            plot_ofdm_time_frequency(pred_sched, pred_ap_onehot)

    # Report finale
    print("\n==================== RISULTATI FINALI ====================")
    print(f"Scheduling accuracy media: {np.mean(sched_acc_list):.4f}")
    print(f"AP accuracy media:         {np.mean(ap_acc_list):.4f}")
    print(f"Collision score medio:     {np.mean(coll_list):.4f}")
    print("==========================================================")

    


    return {
        "sched_acc": np.mean(sched_acc_list),
        "ap_acc":   np.mean(ap_acc_list),
        "collision": np.mean(coll_list)
    }


# ===============================================================
# Main
# ===============================================================
if __name__ == "__main__":
    test_model()
