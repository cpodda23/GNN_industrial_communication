# ===============================================================
# testing.py — Testing per IndustrialMAC_HeteroGNN
# ===============================================================

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from nets import IndustrialMAC_HeteroGNN
from training import IndustrialDataset
from wirelessNetwork import build_hetero_graph


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===============================================================
# Metriche di valutazione
# ===============================================================

def scheduling_accuracy(pred_sched, true_sched, threshold=0.5):
    pred_bin = (pred_sched > threshold).float()
    correct = (pred_bin == true_sched).float().mean().item()
    return correct


def ap_accuracy(pred_ap, true_ap):
    pred_labels = pred_ap.argmax(dim=1)
    correct = (pred_labels == true_ap).float().mean().item()
    return correct


def collision_score(pred_sched):
    """
    Penalizza slot con più device attivi contemporaneamente.
    score = media del numero di device attivi per slot.
    Ideale = 1.
    """
    pred_bin = (pred_sched > 0.5).float()
    per_slot = pred_bin.sum(dim=0)  # [T]

    return per_slot.mean().item()


# ===============================================================
# Testing loop
# ===============================================================

def test_model(
        model_path="model_epoch_20.pth",
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
            ap_pos = batch["ap_pos"].squeeze(0).to(DEVICE)
            csi = batch["csi"].squeeze(0).to(DEVICE)

            true_sched = batch["schedule"].squeeze(0).to(DEVICE)
            true_ap = batch["ap_assign"].squeeze(0).to(DEVICE)

            # Costruzione grafo eterogeneo
            g = build_hetero_graph(
                device_pos.cpu().numpy(),
                ap_pos.cpu().numpy()
            )
            g = g.to(DEVICE)

            # Forward
            pred_sched, pred_ap = model(g, device_pos, ap_pos, csi)

            # Metriche
            sa = scheduling_accuracy(pred_sched, true_sched)
            aa = ap_accuracy(pred_ap, true_ap)
            cs = collision_score(pred_sched)

            sched_acc_list.append(sa)
            ap_acc_list.append(aa)
            coll_list.append(cs)

            print(f"[Sample {i}]")
            print(f"  Scheduling accuracy: {sa:.4f}")
            print(f"  AP accuracy:         {aa:.4f}")
            print(f"  Collision score:     {cs:.4f}")

    # =========================
    # Report finale
    # =========================
    print("\n==================== RISULTATI FINALI ====================")
    print(f"Scheduling accuracy media: {np.mean(sched_acc_list):.4f}")
    print(f"AP accuracy media:         {np.mean(ap_acc_list):.4f}")
    print(f"Collision score medio:     {np.mean(coll_list):.4f}")
    print("==========================================================")

    return {
        "sched_acc": np.mean(sched_acc_list),
        "ap_acc": np.mean(ap_acc_list),
        "collision": np.mean(coll_list)
    }



# ===============================================================
# Main
# ===============================================================
if __name__ == "__main__":
    test_model()
