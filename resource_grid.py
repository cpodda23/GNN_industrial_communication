import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from data_generation import NUM_AP, FREQ_SUBCARRIERS

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


def plot_ofdm_grid(pred_ap_onehot, num_subcarriers=FREQ_SUBCARRIERS):
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


def plot_ofdm_time_frequency_window(pred_sched, pred_ap_onehot,
                                    num_subcarriers=FREQ_SUBCARRIERS,
                                    t_start=0,
                                    window_size=None):
    """
    Visualizza una griglia OFDM (Frequency × Time) su una finestra temporale.

    pred_sched      : [N, T]
    pred_ap_onehot  : [N, T, A]
    num_subcarriers : numero di subcarrier (asse frequenza)
    t_start         : indice di slot iniziale della finestra
    window_size     : numero di slot (W). Se None → usa tutti da t_start in poi.
    """

    N, T = pred_sched.shape
    _, _, A = pred_ap_onehot.shape

    if window_size is None:
        window_size = T - t_start

    t_end = min(t_start + window_size, T)   # esclusivo
    W = t_end - t_start                     # W effettivo

    # 1. nodo servito per AP e timeslot: [T, A]
    node_for_ap_full = pred_ap_onehot.argmax(dim=0).cpu().numpy()  # [T, A]

    # 2. seleziona solo la finestra temporale [t_start : t_end]
    node_for_ap = node_for_ap_full[t_start:t_end, :]               # [W, A]

    # 3. costruiamo griglia F × W
    grid = np.zeros((num_subcarriers, W), dtype=int)
    ap_grid = np.zeros((num_subcarriers, W), dtype=int)

    sc_per_ap = num_subcarriers // A

    for local_t, global_t in enumerate(range(t_start, t_end)):
        for ap in range(A):
            node = node_for_ap[local_t, ap]
            start = ap * sc_per_ap
            end   = (ap + 1) * sc_per_ap
            grid[start:end, local_t] = node
            ap_grid[start:end, local_t] = ap

    # 4. colormap: bianco per inattivo (se lo usi), 3 colori per AP
    cmap_ap = ListedColormap([
        "white",     # -1 → inattivo"
        "#1f77b4",    # 1 → AP0
        "#d62728",    # 2 → AP1
        "#2ca02c"     # 3 → AP2
    ])

    # qui ap_grid ∈ {0,1,2}, lo mappiamo a {1,2,3}
    plot_grid = ap_grid + 1

    plt.figure(figsize=(12, 5))
    plt.imshow(plot_grid, aspect='auto', cmap=cmap_ap, vmin=0, vmax=3)

    cbar = plt.colorbar(ticks=[1,2,3])
    cbar.ax.set_yticklabels(["AP0", "AP1", "AP2"])
    cbar.set_label("Access Point (AP)")
    
    plt.xticks(
    ticks=range(W),
    labels=[str(t) for t in range(t_start, t_end)]
)


    plt.xlabel(f"Timeslot (window from {t_start} to {t_end-1})")
    plt.ylabel("Frequency (subcarrier)")
    plt.title("OFDM Time × Frequency — Doppler window")

    # opzionale: scrivi i numeri dei nodi al centro del blocco di ogni AP
    for local_t in range(W):
        for ap in range(A):
            node = node_for_ap[local_t, ap]
            start = ap * sc_per_ap
            mid   = start + sc_per_ap // 2
            plt.text(local_t, mid, str(node),
                     ha='center', va='center', color='white', fontsize=8)

    plt.tight_layout()
    plt.show()
    
def plot_all_doppler_windows(pred_sched, pred_ap_onehot,
                             num_subcarriers=FREQ_SUBCARRIERS,
                             window_size=5):
    """
    Genera tutte le "fotografie Doppler" consecutive dell'intervallo totale.
    """

    N, T = pred_sched.shape

    t_start = 0
    while t_start < T:
        plot_ofdm_time_frequency_window(
            pred_sched,
            pred_ap_onehot,
            num_subcarriers=num_subcarriers,
            t_start=t_start,
            window_size=window_size
        )
        t_start += window_size


