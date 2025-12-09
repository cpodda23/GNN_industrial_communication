import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np

def plot_node_time_scheduling(pred_sched, pred_ap_onehot, true_sched=None):
    """
    pred_sched      : [N,T] valori 0/1
    pred_ap_onehot  : [N,T,A] valori one-hot
    true_sched      : opzionale, stessa shape di pred_sched per confronto
    """

    N, T = pred_sched.shape
    A = pred_ap_onehot.shape[-1]

    # colore = AP scelto (0..A-1), -1 = inattivo
    ap_assign = pred_ap_onehot.argmax(dim=-1).cpu().numpy()     # [N,T]
    sched = pred_sched.cpu().numpy()

    # matrice finale con -1 per inattivi
    mat = np.where(sched == 1, ap_assign, -1)

    plt.figure(figsize=(12, 6))
    cmap = plt.get_cmap("tab10", A+1)

    plt.imshow(mat, aspect="auto", cmap=cmap, vmin=-1, vmax=A-1)
    plt.colorbar(label="Assigned AP (-1 = inactive)")
    plt.xlabel("Timeslot")
    plt.ylabel("Node")
    plt.title("NodeScheduling × Time (color = AP)")
    plt.show()

def plot_ap_time_assignment(pred_ap_onehot):
    """
    pred_ap_onehot : [N,T,A]
    """
    pred_ap = pred_ap_onehot.argmax(dim=0).cpu().numpy()   # shape [T,A]

    plt.figure(figsize=(12, 4))
    plt.imshow(pred_ap.T, aspect="auto", cmap="tab10")
    plt.colorbar(label="Served Node")
    plt.xlabel("Timeslot")
    plt.ylabel("AP")
    plt.title("AP Assignment × Time (color = Node)")
    plt.show()


def plot_ofdm_grid(pred_ap_onehot, num_subcarriers=32):
    """
    Visualizza una griglia OFDM (frequenza × tempo).
    Ogni slot assume il colore dell'AP assegnato.
    """
    N, T, A = pred_ap_onehot.shape
    ap_assign = pred_ap_onehot.argmax(dim=-1).cpu().numpy()  # [N,T]

    # prendiamo la decisione AP più attiva (o media, o max)
    # qui usiamo l'AP più frequentato
    ap_per_ts = np.argmax(ap_assign.sum(axis=0))

    # costruiamo grid F×T
    grid = np.tile(ap_assign[np.newaxis, :, :], (num_subcarriers, 1, 1))
    # grid shape = [F,N,T], scegliamo asse N
    grid = np.max(grid, axis=1)   # [F,T]

    plt.figure(figsize=(14, 6))
    plt.imshow(grid, aspect="auto", cmap="tab10")
    plt.colorbar(label="Dominant AP")
    plt.xlabel("Timeslot")
    plt.ylabel("Frequency (subcarrier)")
    plt.title("Resource Grid OFDM (Dominant AP in each slot)")
    plt.show()


def plot_ofdm_time_frequency(pred_sched, pred_ap_onehot, num_subcarriers=32):
    """
    Visualizza una griglia OFDM (Frequency × Time).
    - pred_sched: [N, T] (0/1)
    - pred_ap_onehot: [N, T, A]
    - num_subcarriers: numero di subcarrier (asse frequenza)
    """

    # ---------------------------------------------------------------
    # 1. Estrai nodi attivi e AP assegnati per ciascuno slot
    # ---------------------------------------------------------------
    N, T = pred_sched.shape
    _, _, A = pred_ap_onehot.shape

    # nodo servito per slot per ogni AP: shape [T, A]
    node_for_ap = pred_ap_onehot.argmax(dim=0).cpu().numpy()  # [T, A]

    # ---------------------------------------------------------------
    # 2. Costruzione matrice OFDM Time × Frequency
    #    Ogni slot avrà fino a A nodi attivi, uno per AP
    # ---------------------------------------------------------------
    grid = np.zeros((num_subcarriers, T), dtype=int)
    ap_grid = np.zeros((num_subcarriers, T), dtype=int)

    # distribuzione subcarrier: ogni AP occupa un blocco
    sc_per_ap = num_subcarriers // A

    for t in range(T):
        for ap in range(A):

            node = node_for_ap[t, ap]

            start = ap * sc_per_ap
            end = (ap + 1) * sc_per_ap
            grid[start:end, t] = node
            ap_grid[start:end, t] = ap

    # ---------------------------------------------------------------
    # 3. Plot della griglia
    # ---------------------------------------------------------------
    plt.figure(figsize=(15, 6))
    cmap_ap = ListedColormap([
        "white",      # 0 → inattivo
        "#1f77b4",    # 1 → AP0 (blu)
        "#d62728",    # 2 → AP1 (rosso)
        "#2ca02c"     # 3 → AP2 (verde)
    ])    
    plot_grid = ap_grid.copy()
    plot_grid = plot_grid + 1   # ora: -1→0, 0→1, 1→2, 2→3
    plt.imshow(plot_grid, aspect="auto", cmap=cmap_ap, vmin=0, vmax=3)
    cbar = plt.colorbar(ticks=[1,2,3])
    cbar.ax.set_yticklabels(["AP0", "AP1", "AP2"])
    cbar.set_label("Access Point (AP)")
    

    plt.xlabel("Timeslot")
    plt.ylabel("Frequency (subcarrier)")
    plt.title("OFDM Time × Frequency – Nodo e AP for every slot")

    # ---------------------------------------------------------------
    # 4. Disegna i numeri dei nodi come etichette visive nella griglia
    # ---------------------------------------------------------------
    for t in range(T):
        for ap in range(A):
            node = node_for_ap[t, ap]
            start = ap * sc_per_ap
            mid = start + sc_per_ap // 2
            plt.text(t, mid, str(node),
                     ha='center', va='center', color='white', fontsize=10)

    plt.show()
