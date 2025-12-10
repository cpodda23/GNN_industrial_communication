import torch
import numpy as np
import os

# Random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Parameters of the industrial system
NUM_NODES = 10              # industrial nodes
NUM_AP = 3                  # Access Points
FREQ_SUBCARRIERS = 32       # OFDM subcarriers
TIME_SLOTS = 40             # OFDM symbols
AREA_SIZE = 50              # meters, factory area
DOPPLER_HZ = 5              # indoor doppler
SHADOWING_STD = 2.0         # log-normal shadowing standard deviation

OUTPUT_DIR = "data"


# ===========================================================
# Function 1: generate industrial topology
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
# Function 2: pathloss model
# ===========================================================
def pathloss(d):
    PL0 = -30     # dB reference
    n = 2.2       # indoor industrial
    shadow = np.random.normal(0, SHADOWING_STD)  # shadowing log-normal
    return PL0 + 10 * n * np.log10(d + 1e-6) + shadow


# ===========================================================
# Function 3: generate COMPLETE OFDM CSI (mag + phase)
# ===========================================================
def generate_csi(nodes_pos, ap_pos):
    """
    Returns CSI shape:
    [NUM_NODES, NUM_AP, FREQ_SUBCARRIERS, TIME_SLOTS, 2]
    Where the last dimension is:
      [:,:,:, :, 0] = magnitude
      [:,:,:, :, 1] = phase
    """

    CSI = np.zeros((NUM_NODES, NUM_AP, FREQ_SUBCARRIERS, TIME_SLOTS, 2), dtype=np.float32)

    for n in range(NUM_NODES):
        for ap in range(NUM_AP):

            # distance node-AP
            d = np.linalg.norm(nodes_pos[n] - ap_pos[ap])
            pl = pathloss(d)

            # complex fading variable over time
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
# Function 4: deterministic scheduling + AP selection
# ===========================================================
NOISE_POWER = 1e-9  # Thermal noise (can be adjusted)
EPS = 1e-12

def generate_scheduling_ofdm(csi, num_nodes, num_ap, num_subcarriers, time_slots):
    """
    Scheduling based on OFDM:
    - each AP selects 1 node per timeslot
    - selection depends on the real OFDM CSI
    - multiple nodes can be active in the same slot (multi-AP)
    """

    # Output
    schedule = np.zeros((num_nodes, time_slots), dtype=np.float32)
    ap_assign = -1 * np.ones((num_nodes, time_slots), dtype=np.int32)

    # LOOP over timeslots
    for t in range(time_slots):

        # Each AP assigns the slot to the node with the best throughput
        for a in range(num_ap):

            # R[n] = estimated OFDM throughput for node n with AP a at slot t
            R = np.zeros(num_nodes)

            for n in range(num_nodes):

                # Complex magnitude for all subcarriers
                mag = csi[n, a, :, t, 0]      # shape [F]
                # We can interpret the channel as H and estimate an SNR
                H2 = mag ** 2
                SNR = H2 / (NOISE_POWER + EPS)

                # Throughput over all subcarriers
                R[n] = np.sum(np.log2(1 + SNR))

            # Select best node for AP at slot t
            best_node = np.argmax(R)

            # Update schedule
            schedule[best_node, t] = 1.0
            ap_assign[best_node, t] = a

    return schedule, ap_assign



# ===========================================================
# Main function: dataset generation
# ===========================================================
def generate_dataset(num_samples=200):

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for i in range(num_samples):

        nodes_pos, ap_pos = generate_topology()
        csi = generate_csi(nodes_pos, ap_pos)
        schedule, ap_assign = generate_scheduling_ofdm(csi, NUM_NODES, NUM_AP, FREQ_SUBCARRIERS, TIME_SLOTS)

        sample = {
            "nodes_pos": torch.tensor(nodes_pos, dtype=torch.float),
            "ap_pos": torch.tensor(ap_pos, dtype=torch.float),
            "csi": torch.tensor(csi, dtype=torch.float),      
            "schedule": torch.tensor(schedule, dtype=torch.float),
            "ap_assign": torch.tensor(ap_assign, dtype=torch.long)
        }

        torch.save(sample, f"{OUTPUT_DIR}/sample_{i:04d}.pt")
        print(f"[OK] Sample saved {i}")


if __name__ == "__main__":
    generate_dataset(num_samples=50)
