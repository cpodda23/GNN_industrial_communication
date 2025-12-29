import torch
import numpy as np
import os

# Random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Parameters of the industrial system
NUM_NODES = 10              # industrial nodes
NUM_AP = 3                  # Access Points
FREQ_SUBCARRIERS = 64       # OFDM subcarriers
TIME_SLOTS = 40             # OFDM symbols
AREA_SIZE = 50              # meters, factory area
DOPPLER_HZ = 5              # indoor doppler
SHADOWING_STD = 2.0         # log-normal shadowing standard deviation
NUM_SAMPLES = 1000            # dataset samples

OUTPUT_DIR = "data"


# ===========================================================
# Generate industrial topology
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
# Pathloss model
# ===========================================================
def pathloss(d):
    PL0 = -30     # dB reference
    n = 2.2       # indoor industrial
    shadow = np.random.normal(0, SHADOWING_STD)  # shadowing log-normal
    return PL0 + 10 * n * np.log10(d + 1e-6) + shadow


# ===========================================================
# Generate COMPLETE OFDM CSI (mag + phase)
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
    
    base_subcarriers_per_ap = FREQ_SUBCARRIERS // NUM_AP
    remainder = FREQ_SUBCARRIERS % NUM_AP

    for n in range(NUM_NODES):
        for ap in range(NUM_AP):

            # distance node-AP
            d = np.linalg.norm(nodes_pos[n] - ap_pos[ap])
            pl = pathloss(d)

            subcarrier_start = ap * base_subcarriers_per_ap + min(ap, remainder)
            subcarrier_end = subcarrier_start + base_subcarriers_per_ap + (1 if ap < remainder else 0)

            num_sc_ap = subcarrier_end - subcarrier_start

            # complex fading variable over time
            H = np.zeros((num_sc_ap, TIME_SLOTS), dtype=np.complex64)

            for t in range(TIME_SLOTS):

                doppler_phase = np.exp(1j * 2 * np.pi * DOPPLER_HZ * t * 0.001)

                fading = (np.random.randn(num_sc_ap) +
                          1j * np.random.randn(num_sc_ap)) / np.sqrt(2)

                H[:, t] = fading * doppler_phase * 10 ** (-pl / 20)

            # convert to mag + phase
            CSI[n, ap, subcarrier_start:subcarrier_end, :, 0] = np.abs(H)
            CSI[n, ap, subcarrier_start:subcarrier_end, :, 1] = np.angle(H)

    return CSI


# ===========================================================
# Deterministic scheduling + AP selection
# ===========================================================
NOISE_POWER = 1e-9  # Thermal noise (can be adjusted)
EPS = 1e-12

def generate_scheduling_ofdm(csi, num_nodes, num_ap, num_subcarriers, time_slots):
    """
    Scheduling based on OFDM:
    - each AP selects 1 node per timeslot
    - selection depends on the real OFDM CSI (max throughput)
    - multiple nodes can be active in the same slot (multi-AP)
    - each node has a random number of packets to send completely
    """

    # Output
    schedule = np.zeros((num_nodes, time_slots), dtype=np.float32)
    ap_assign = -1 * np.ones((num_nodes, time_slots), dtype=np.int32)
    
    # Number of packets for each node
    node_packets = np.random.randint(1, 11, size=num_nodes)  # 1-10 packets per node
    print("Node packets:", node_packets)

    # Keep track of how many packets are left for each node
    packets_left = node_packets.copy()

    # LOOP over timeslots
    for t in range(time_slots):
        
        # Track nodes that are already selected for this timeslot
        available_nodes = set(range(num_nodes))

        # Each AP assigns the slot to the node with the best throughput
        for a in range(num_ap):
            
            # Initialize selection variables
            best_node = -1
            max_throughput = -1.0

            # iteerate over available nodes to find one with packets left
            for n in list(available_nodes):
                
                # Skip nodes with no packets left
                if packets_left[n] <= 0:
                    continue
                
                # Complex magnitude for all subcarriers
                mag = csi[n, a, :, t, 0]      # shape [F]
                # We can interpret the channel as H and estimate an SNR
                H2 = mag ** 2
                SNR = H2 / (NOISE_POWER + EPS)
                # Throughput over all subcarriers
                current_R = np.sum(np.log2(1 + SNR))

                # Logic Max-Throughput
                if current_R > max_throughput:
                    max_throughput = current_R
                    best_node = n
                    
            # No available node found (every node has sent all packets)
            if best_node == -1:
                continue


            # Update schedule
            schedule[best_node, t] = 1.0
            ap_assign[best_node, t] = a

            # Decrement the number of packets left for the selected node
            packets_left[best_node] -= 1
            
            # Remove the selected node from available nodes for this timeslot (to avoid double assignment)
            available_nodes.remove(best_node)

    return schedule, ap_assign, node_packets



# ===========================================================
# Main function: dataset generation
# ===========================================================
def generate_dataset(num_samples):

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for i in range(num_samples):

        nodes_pos, ap_pos = generate_topology()
        csi = generate_csi(nodes_pos, ap_pos)
        schedule, ap_assign, node_packets = generate_scheduling_ofdm(csi, NUM_NODES, NUM_AP, FREQ_SUBCARRIERS, TIME_SLOTS)

        sample = {
            "nodes_pos": torch.tensor(nodes_pos, dtype=torch.float),
            "ap_pos": torch.tensor(ap_pos, dtype=torch.float),
            "csi": torch.tensor(csi, dtype=torch.float),      
            "schedule": torch.tensor(schedule, dtype=torch.float),
            "ap_assign": torch.tensor(ap_assign, dtype=torch.long),
            "node_packets": torch.tensor(node_packets, dtype=torch.long)

        }

        torch.save(sample, f"{OUTPUT_DIR}/sample_{i:04d}.pt")
        print(f"[OK] Sample saved {i}")


if __name__ == "__main__":
    generate_dataset(num_samples=NUM_SAMPLES)
