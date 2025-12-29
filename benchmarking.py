import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from nets import IndustrialMAC_HeteroGNN
from training import IndustrialDataset
from wirelessNetwork import build_hetero_graph
from data_generation import NUM_AP, FREQ_SUBCARRIERS, TIME_SLOTS, NOISE_POWER, EPS
from resource_grid import visualize_grid

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "model_epoch_50.pth"
DATA_DIR = "data" 
NUM_SAMPLES = 100 


def decisions_to_fake_logits(ap_assign, num_ap):
    """
    Convert a matrix of AP assignments [N, T] (integers)
    into a matrix of 'fake logits' [N, T, Num_AP] for the visualizer.
    """
    N, T = ap_assign.shape
    # Initialize everything to a very low value (probability ~0)
    logits = np.ones((N, T, num_ap)) * -100.0
    
    for n in range(N):
        for t in range(T):
            chosen = int(ap_assign[n, t])
            if chosen >= 0 and chosen < num_ap:
                # high value on the chosen AP (probability ~1)
                logits[n, t, chosen] = 100.0
                
    # Return as tensor because visualize_ofdm_grid sometimes expects it
    return torch.tensor(logits)

def calculate_throughput(csi, schedule, ap_assign, num_ap, num_subcarriers):
    """Calculate total bits transmitted (Sum-Rate)"""
    total_rate = 0.0
    num_nodes, _, _, time_slots, _ = csi.shape
    base_sc = num_subcarriers // num_ap
    remainder = num_subcarriers % num_ap

    for t in range(time_slots):
        active_nodes = np.where(schedule[:, t] > 0.5)[0]
        
        # Check collisions per AP
        ap_usage = {}
        for n in active_nodes:
            ap = int(ap_assign[n, t])
            if ap not in ap_usage: ap_usage[ap] = []
            ap_usage[ap].append(n)
            
        for ap, nodes in ap_usage.items():
            start = ap * base_sc + min(ap, remainder)
            end = start + base_sc + (1 if ap < remainder else 0)

            # IF COLLISION (more nodes on same AP): Rate = 0 (max penalty)
            if len(nodes) > 1:
                pass 
            else:
                # IF OK: Calculate Shannon Capacity
                node = nodes[0]
                mag = csi[node, ap, start:end, t, 0]
                snr = (mag ** 2) / (NOISE_POWER + EPS)
                rate = np.sum(np.log2(1 + snr))
                total_rate += rate
    return total_rate

# --- 1. TDMA (Time Division) ---
def strategy_tdma(num_nodes, time_slots, csi):
    schedule = np.zeros((num_nodes, time_slots))
    ap_assign = np.zeros((num_nodes, time_slots))
    
    for t in range(time_slots):
        # Round Robin: for each time slot, only one node transmits
        active_node = t % num_nodes
        schedule[active_node, t] = 1
        
        # In TDMA, the active node chooses the best AP for it in that moment
        # (TDMA "Smart", otherwise it would be too poor)
        best_ap = -1
        best_snr = -1
        for ap in range(NUM_AP):
            mag = csi[active_node, ap, :, t, 0]
            snr = np.mean(mag**2)
            if snr > best_snr:
                best_snr = snr
                best_ap = ap
        ap_assign[active_node, t] = best_ap
        
    return schedule, ap_assign

# --- 2. FDMA (Frequency Division) ---
def strategy_fdma(num_nodes, time_slots, num_ap):
    schedule = np.zeros((num_nodes, time_slots))
    ap_assign = np.zeros((num_nodes, time_slots))
    
    # Static assignment: divide nodes among APs
    # Example: 10 nodes, 3 APs.
    # Nodes 0,1,2,3 -> AP 0
    # Nodes 4,5,6   -> AP 1
    # Nodes 7,8,9   -> AP 2
    
    nodes_per_ap = num_nodes // num_ap
    
    for n in range(num_nodes):
        # Assign fixed AP based on node index
        my_fixed_ap = min(n // (nodes_per_ap + 1), num_ap - 1)
        # Or a simpler distribution:
        my_fixed_ap = n % num_ap
        
        # on pure FDMA, transmit always
        schedule[n, :] = 1 
        ap_assign[n, :] = my_fixed_ap
        
    return schedule, ap_assign

# --- MAIN BENCHMARK ---
def run_benchmark():
    model = IndustrialMAC_HeteroGNN().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Model {MODEL_PATH} loaded.")
    except:
        print("ERROR: Model not found. GNN will fail.")

    dataset = IndustrialDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    res = {"Optimal": [], "OFDM GNN": [], "TDMA": [], "FDMA": []}

    print(f"Start comparison on {NUM_SAMPLES} samples...")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= NUM_SAMPLES: break
            
            d_pos = batch["device_pos"].squeeze(0).to(DEVICE)
            a_pos = batch["ap_pos"].squeeze(0).to(DEVICE)
            csi_tensor = batch["csi"].squeeze(0).to(DEVICE)
            node_pkts = batch["node_packets"].squeeze(0).to(DEVICE)
            
            # Ground Truth (Optimal)
            true_sched = batch["schedule"].squeeze(0).numpy()
            true_ap = batch["ap_assign"].squeeze(0).numpy()
            csi_np = batch["csi"].squeeze(0).numpy()
            num_nodes = d_pos.shape[0]

            # 1. Optimal
            r_opt = calculate_throughput(csi_np, true_sched, true_ap, NUM_AP, FREQ_SUBCARRIERS)
            res["Optimal"].append(r_opt)

            # 2. OFDM GNN
            g = build_hetero_graph(d_pos.cpu().numpy(), a_pos.cpu().numpy()).to(DEVICE)
            pred_sched, pred_ap_logits, _ = model(g, d_pos, a_pos, csi_tensor, node_pkts)
            sched_gnn = pred_sched.cpu().numpy()
            ap_gnn = pred_ap_logits.argmax(dim=-1).cpu().numpy()
            r_gnn = calculate_throughput(csi_np, sched_gnn, ap_gnn, NUM_AP, FREQ_SUBCARRIERS)
            res["OFDM GNN"].append(r_gnn)

            # 3. TDMA
            s_tdma, a_tdma = strategy_tdma(num_nodes, TIME_SLOTS, csi_np)
            r_tdma = calculate_throughput(csi_np, s_tdma, a_tdma, NUM_AP, FREQ_SUBCARRIERS)
            res["TDMA"].append(r_tdma)

            # 4. FDMA
            s_fdma, a_fdma = strategy_fdma(num_nodes, TIME_SLOTS, NUM_AP)
            r_fdma = calculate_throughput(csi_np, s_fdma, a_fdma, NUM_AP, FREQ_SUBCARRIERS)
            res["FDMA"].append(r_fdma)

    # --- PLOT RESULTS ---
    means = {k: np.mean(v) for k, v in res.items()}
    
    print("\n=== FINAL RESULTS (Bits per Frame) ===")
    for k, v in means.items():
        print(f"{k}: {v:.2f}")

    plt.figure(figsize=(10, 6))
    bars = plt.bar(means.keys(), means.values(), color=['green', 'blue', 'orange', 'red'])
    plt.title("Resource Allocation Performance Comparison")
    plt.ylabel("System Throughput (Bits/Frame)")
    plt.grid(axis='y', alpha=0.3)
    
    # Aggiungi valori sulle barre
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.0f}", ha='center', va='bottom')
        
    plt.show()

def run_visual_benchmark():
    print("\n=== VISUALIZATION OF RESOURCE GRIDS FOR DIFFERENT STRATEGIES ===")
    dataset = IndustrialDataset(DATA_DIR)
    # Important: batch_size=1
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # take FIRST sample
    batch = next(iter(loader))
    
    # extract the data removing the batch dimension [1, ...] -> [...]
    d_pos = batch["device_pos"].squeeze(0)         # [N, 2]
    a_pos = batch["ap_pos"].squeeze(0)             # [A, 2]
    csi_np = batch["csi"].squeeze(0).numpy()       # [N, A, F, T, 2]
    node_pkts = batch["node_packets"].squeeze(0)   # [N]
    
    # Ground Truth (OPTIMAL)
    sched_opt = batch["schedule"].squeeze(0).numpy()
    ap_opt = batch["ap_assign"].squeeze(0).numpy()
    
    num_nodes, time_slots = sched_opt.shape
    
    # 1. Calculate TDMA
    sched_tdma, ap_tdma = strategy_tdma(num_nodes, time_slots, csi_np)
    
    # 2. Calculate FDMA
    sched_fdma, ap_fdma = strategy_fdma(num_nodes, time_slots, NUM_AP)
    
    # 3. Calculate GNN OFDM
    model = IndustrialMAC_HeteroGNN().to(DEVICE)
    sched_gnn = None
    logits_gnn = None
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        
        # Build graph (move to GPU)
        g = build_hetero_graph(d_pos.numpy(), a_pos.numpy()).to(DEVICE)
        
        # GNN Inference
        pred_sched, pred_ap_logits, _ = model(
            g, 
            d_pos.to(DEVICE), 
            a_pos.to(DEVICE), 
            batch["csi"].squeeze(0).to(DEVICE), 
            node_pkts.to(DEVICE)
        )
        
        sched_gnn = pred_sched.cpu().detach().numpy()
        logits_gnn = pred_ap_logits.cpu().detach().numpy()
        
    except Exception as e:
        print(f"Error during GNN inference: {e}")
        import traceback
        traceback.print_exc()
        return

    # === PLOT 1: OPTIMAL ===
    logits_opt = decisions_to_fake_logits(ap_opt, NUM_AP)
    visualize_grid(sched_opt, logits_opt, FREQ_SUBCARRIERS, TIME_SLOTS, title="Optimal")
    
    # === PLOT 2: TDMA ===
    logits_tdma = decisions_to_fake_logits(ap_tdma, NUM_AP)
    visualize_grid(sched_tdma, logits_tdma, FREQ_SUBCARRIERS, TIME_SLOTS, title="TDMA")
    
    # === PLOT 3: FDMA ===
    logits_fdma = decisions_to_fake_logits(ap_fdma, NUM_AP)
    visualize_grid(sched_fdma, logits_fdma, FREQ_SUBCARRIERS, TIME_SLOTS, title="FDMA")

    # === PLOT 4: GNN ===
    if sched_gnn is not None:
        visualize_grid(sched_gnn, logits_gnn, FREQ_SUBCARRIERS, TIME_SLOTS, title="OFDM GNN")
    
if __name__ == "__main__":
    run_benchmark()
    run_visual_benchmark()