# ===========================================================
# data_generation.py — Versione INDUSTRIALE avanzata
# Compatibile con GNN eterogenea device + AP
# ===========================================================

import torch
import numpy as np
import os

# ===========================================================
# Seeds (riproducibilità)
# ===========================================================
np.random.seed(0)
torch.manual_seed(0)

# ===========================================================
# Parametri del sistema industriale
# ===========================================================
NUM_NODES = 10              # nodi industriali
NUM_AP = 3                  # Access Points
FREQ_SUBCARRIERS = 32       # subcarrier OFDM
TIME_SLOTS = 20             # simboli OFDM
AREA_SIZE = 50              # metri, fabbrica
DOPPLER_HZ = 5              # doppler indoor
SHADOWING_STD = 2.0         # deviazione standard shadowing log-normal

OUTPUT_DIR = "data"


# ===========================================================
# Funzione 1: genera topologia industriale
# ===========================================================
def generate_topology():
    nodes_pos = np.random.uniform(0, AREA_SIZE, size=(NUM_NODES, 2))

    ap_pos = np.array([
        [0, 0],
        [AREA_SIZE, 0],
        [AREA_SIZE / 2, AREA_SIZE]
    ])

    return nodes_pos, ap_pos


# ===========================================================
# Funzione 2: modello di attenuazione
# ===========================================================
def pathloss(d):
    PL0 = -30     # dB reference
    n = 2.2       # indoor industrial
    shadow = np.random.normal(0, SHADOWING_STD)  # shadowing log-normal
    return PL0 + 10 * n * np.log10(d + 1e-6) + shadow


# ===========================================================
# Funzione 3: genera CSI OFDM COMPLETO (mag + phase)
# ===========================================================
def generate_csi(nodes_pos, ap_pos):
    """
    Ritorna CSI shape:
    [NUM_NODES, NUM_AP, FREQ_SUBCARRIERS, TIME_SLOTS, 2]
    Dove l'ultima dimensione è:
      [:,:,:, :, 0] = magnitude
      [:,:,:, :, 1] = phase
    """

    CSI = np.zeros((NUM_NODES, NUM_AP, FREQ_SUBCARRIERS, TIME_SLOTS, 2), dtype=np.float32)

    for n in range(NUM_NODES):
        for ap in range(NUM_AP):

            # distanza nodo-AP
            d = np.linalg.norm(nodes_pos[n] - ap_pos[ap])
            pl = pathloss(d)

            # fading complesso variabile nel tempo
            H = np.zeros((FREQ_SUBCARRIERS, TIME_SLOTS), dtype=np.complex64)

            for t in range(TIME_SLOTS):

                doppler_phase = np.exp(1j * 2 * np.pi * DOPPLER_HZ * t * 0.001)

                fading = (np.random.randn(FREQ_SUBCARRIERS) +
                          1j * np.random.randn(FREQ_SUBCARRIERS)) / np.sqrt(2)

                H[:, t] = fading * doppler_phase * 10 ** (-pl / 20)

            # convert to mag + phase
            CSI[n, ap, :, :, 0] = np.abs(H)
            CSI[n, ap, :, :, 1] = np.angle(H)

    return CSI


# ===========================================================
# Funzione 4: scheduling deterministico + AP selection
# ===========================================================
def generate_scheduling(nodes_pos, ap_pos):
    schedule = np.zeros((NUM_NODES, TIME_SLOTS))

    # TDMA deterministico
    slot = 0
    for n in range(NUM_NODES):
        schedule[n, slot] = 1
        slot = (slot + 1) % TIME_SLOTS

    # AP assignment basato sul pathloss minimo
    ap_assign = []
    for n in range(NUM_NODES):
        distances = [np.linalg.norm(nodes_pos[n] - ap_pos[a]) for a in range(NUM_AP)]
        pl_list = [pathloss(d) for d in distances]
        best_ap = np.argmin(pl_list)
        ap_assign.append(best_ap)

    return schedule, np.array(ap_assign)


# ===========================================================
# Funzione principale: generazione dataset
# ===========================================================
def generate_dataset(num_samples=200):

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for i in range(num_samples):

        nodes_pos, ap_pos = generate_topology()
        csi = generate_csi(nodes_pos, ap_pos)
        schedule, ap_assign = generate_scheduling(nodes_pos, ap_pos)

        sample = {
            "nodes_pos": torch.tensor(nodes_pos, dtype=torch.float),
            "ap_pos": torch.tensor(ap_pos, dtype=torch.float),
            "csi": torch.tensor(csi, dtype=torch.float),      
            "schedule": torch.tensor(schedule, dtype=torch.float),
            "ap_assign": torch.tensor(ap_assign, dtype=torch.long)
        }

        torch.save(sample, f"{OUTPUT_DIR}/sample_{i:04d}.pt")
        print(f"[OK] Salvato sample {i}")


# ===========================================================
# MAIN
# ===========================================================
if __name__ == "__main__":
    generate_dataset(num_samples=50)
