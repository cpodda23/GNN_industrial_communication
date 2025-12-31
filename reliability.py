import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

THRESHOLD_DB = 10.0
NOISE_POWER = 1e-6

def calculate_success_probability(sinr_linear, threshold_db, steepness=1.0):
    # Sigmoid function to map SINR to success probability
    sinr_db = 10 * np.log10(sinr_linear + 1e-9)
    # probabability of success as a sigmoid function
    p_success = 1 / (1 + np.exp(-steepness * (sinr_db - threshold_db)))
    return p_success

def simulate_frame_with_errors(csi, schedule, ap_assign, num_ap, num_subcarriers, noise_power):
    num_nodes, time_slots = schedule.shape
    # 1. Initialize status grid (0=not sent, 1=success, 2=failure, 3=collision)
    status_grid = np.zeros((num_nodes, time_slots), dtype=int)
    base_sc = num_subcarriers // num_ap
    remainder = num_subcarriers % num_ap

    # 2. Simulate transmissions
    for t in range(time_slots):
        # Find active nodes in this time slot
        active_nodes = np.where(schedule[:, t] > 0.5)[0]
        if len(active_nodes) == 0: continue
            
        ap_usage = {}
        for n in active_nodes:
            ap = int(ap_assign[n, t])
            if ap not in ap_usage: ap_usage[ap] = []
            ap_usage[ap].append(n)
            
        for ap, nodes in ap_usage.items():
            if len(nodes) > 1:
                # Collision occurred if more than one node transmits on the same AP
                for n in nodes: status_grid[n, t] = 3
                continue
            
            node = nodes[0]
            start = ap * base_sc + min(ap, remainder)
            end = start + base_sc + (1 if ap < remainder else 0)
            mag = csi[node, ap, start:end, t, 0]
            avg_power = np.mean(mag ** 2)
            sinr = avg_power / noise_power
            p_succ = calculate_success_probability(sinr, threshold_db=THRESHOLD_DB) 
            
            # Simulate success/failure based on probability
            if np.random.rand() < p_succ: status_grid[node, t] = 1 
            else: status_grid[node, t] = 2 
                
    return status_grid

def apply_retransmissions(schedule, ap_assign, status_grid, packet_id_grid, mode='spatial'):
    """
    Two retransmission strategies:
    - mode='time': retrasmission on SAME AP (Time Division Retransmission)
    - mode='spatial': retrasmission on DIFFERENT AP (Spatial Diversity)
    """
    num_nodes, time_slots = schedule.shape
    num_ap = np.max(ap_assign) + 1 
    if num_ap < 0: num_ap = 3 # Default to 3 APs if none assigned
    
    new_schedule = schedule.copy()
    new_ap_assign = ap_assign.copy()
    new_packet_id_grid = packet_id_grid.copy()
    retrans_mask = np.zeros_like(schedule)
    
    # Queue: (node_idx, original_ap, PACKET_ID)
    packet_queue = []

    for t in range(time_slots):
        # 1. Identify failed packets at time t
        failed_nodes = np.where((status_grid[:, t] == 2) | (status_grid[:, t] == 3))[0]
        for n in failed_nodes:
            ap_used = ap_assign[n, t]
            pid = packet_id_grid[n, t]
            packet_queue.append((n, ap_used, pid))
        
        if not packet_queue: continue
            
        # 2. Try to place packets in future slots
        remaining_queue = []
        for (node, original_ap, pid) in packet_queue:
            placed = False
            
            # Scan future time slots
            for t_future in range(t + 1, time_slots):
                if placed: break
                
                # Skip if already scheduled
                if new_schedule[node, t_future] == 1: continue
                
                # Determine candidate APs based on mode
                if mode == 'time':
                    # TIME DIVISION: Use the same AP as original
                    candidate_aps = [original_ap]
                elif mode == 'spatial':
                    # SPATIAL DIVERSITY: Use different APs
                    candidate_aps = [ap for ap in range(num_ap) if ap != original_ap]
                else:
                    raise ValueError("Mode must be 'time' or 'spatial'")

                for candidate_ap in candidate_aps:
                    # Check if AP is free at t_future
                    users_on_ap = [u for u in range(num_nodes) 
                                   if new_schedule[u, t_future] == 1 
                                   and new_ap_assign[u, t_future] == candidate_ap]
                    
                    if len(users_on_ap) == 0:
                        # Place the retransmission
                        new_schedule[node, t_future] = 1
                        new_ap_assign[node, t_future] = candidate_ap
                        new_packet_id_grid[node, t_future] = pid
                        retrans_mask[node, t_future] = 1
                        placed = True
                        break 
            
            if not placed:
                remaining_queue.append((node, original_ap, pid))
        
        packet_queue = remaining_queue

    return new_schedule, new_ap_assign, retrans_mask, new_packet_id_grid

def visualize_retransmission_mode(csi, schedule, ap_assign, num_ap, num_subcarriers, time_slots, mode='spatial'):
    
    SLOT_DURATION_MS = 1.0  # Assumed slot duration in milliseconds (1 slot = 1 ms)
    
    # 1. Setup ID
    packet_id_grid = np.zeros_like(schedule, dtype=int)
    node_counters = np.zeros(schedule.shape[0], dtype=int)
    for t in range(time_slots):
        active_nodes = np.where(schedule[:, t] > 0.5)[0]
        for n in active_nodes:
            node_counters[n] += 1
            packet_id_grid[n, t] = node_counters[n]

    # 2. Simulate Errors
    status_grid = simulate_frame_with_errors(csi, schedule, ap_assign, num_ap, num_subcarriers, noise_power=NOISE_POWER)
    
    # 3. Apply Retransmissions with specific MODE
    final_sched, final_ap_assign, retrans_mask, final_packet_id_grid = apply_retransmissions(
        schedule, ap_assign, status_grid, packet_id_grid, mode=mode
    )

    # --- CALCULATE METRICS ---
    pkts_sent_total = np.sum(schedule)
    pkts_error = np.sum((status_grid == 2) | (status_grid == 3))
    pkts_recovered = np.sum(retrans_mask)
    
    # Reliability & PER
    rel_init = 1.0 - (pkts_error / pkts_sent_total) if pkts_sent_total > 0 else 0
    rel_final = 1.0 - ((pkts_error - pkts_recovered) / pkts_sent_total) if pkts_sent_total > 0 else 0
    per_final = 1.0 - rel_final  # Packet Error Rate final
    
    # Latency (Calculate in Slot -> ms)
    total_delay_slots = 0
    recovered_count = 0
    retx_indices = np.where(retrans_mask == 1)
    
    for i in range(len(retx_indices[0])):
        node = retx_indices[0][i]
        t_new = retx_indices[1][i]
        pid = final_packet_id_grid[node, t_new]
        
        # Find t_old (original transmission time)
        t_old = -1
        for t_scan in range(t_new):
            if packet_id_grid[node, t_scan] == pid and ((status_grid[node, t_scan] == 2) or (status_grid[node, t_scan] == 3)):
                t_old = t_scan
                break
        # if original found, calculate delay
        if t_old != -1:
            delay = t_new - t_old
            total_delay_slots += delay
            recovered_count += 1
            
    avg_delay_slots = total_delay_slots / recovered_count if recovered_count > 0 else 0.0
    avg_delay_ms = avg_delay_slots * SLOT_DURATION_MS # Convert to ms

    # Resource Cost
    slots_original = pkts_sent_total
    slots_extra = pkts_recovered
    efficiency_loss = slots_extra / slots_original if slots_original > 0 else 0

    print(f"\nREPORT: {mode.upper()} RETRANSMISSION STRATEGY")
    
    print(f"\n1. RELIABILITY & PER")
    print(f"   - Success Rate:      {rel_final*100:.2f} %")
    print(f"   - Packet Error Rate: {per_final:.2e} (Scientific Notation)")
    print(f"   - Packets Recovered: {int(pkts_recovered)} / {int(pkts_error)} failures")

    print(f"2. LATENCY")
    print(f"   - Avg Delay (Slots): {avg_delay_slots:.2f} slots")
    print(f"   - Avg Delay (Time):  {avg_delay_ms:.2f} ms")
    
    print(f"3. EFFICIENCY")
    print(f"   - Spectral Overhead: {efficiency_loss*100:.1f} % extra resources used")
    print(f"=======================================================")

    # --- PLOT ---
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0, time_slots)
    ax.set_ylim(0, num_subcarriers)
    title_str = "Time Division Retransmission (Same AP)" if mode == 'time' else "Spatial Diversity Retransmission (Different AP)"
    ax.set_title(f"{title_str}\nGain: {rel_init*100:.1f}% -> {rel_final*100:.1f}%")
    ax.set_xlabel("Time Slots")
    ax.set_ylabel("Frequency (Subcarriers)")
    
    base_sc = num_subcarriers // num_ap
    rem_sc = num_subcarriers % num_ap

    for t in range(time_slots):
        active_nodes = np.where(final_sched[:, t] > 0.5)[0]
        for node_idx in active_nodes:
            ap = int(final_ap_assign[node_idx, t])
            pkt_num = final_packet_id_grid[node_idx, t]
            is_retrans = retrans_mask[node_idx, t] == 1
            original_status = status_grid[node_idx, t] if not is_retrans else 1
            
            label_text = ""
            font_w = 'normal'
            text_col = 'white'
            
            if is_retrans:
                face_c = 'cyan'
                edge_c = 'blue'
                label_text = f"N{node_idx}\nRETX\nP{pkt_num}"
                font_w = 'bold'
                text_col = 'black'
            elif original_status == 3:
                face_c = 'red'
                edge_c = 'white'
                label_text = f"N{node_idx}\nCOLL\nP{pkt_num}"
                font_w = 'bold'
            elif original_status == 2:
                face_c = 'orange'
                edge_c = 'white'
                label_text = f"N{node_idx}\nERR\nP{pkt_num}"
                font_w = 'bold'
            else:
                face_c = 'green'
                edge_c = 'white'
                label_text = f"N{node_idx}\nP{pkt_num}"
            
            start_f = ap * base_sc + min(ap, rem_sc)
            height_f = base_sc + (1 if ap < rem_sc else 0)
            nodes_on_ap = [n for n in active_nodes if final_ap_assign[n,t] == ap]
            slot_h = height_f / len(nodes_on_ap)
            sub_idx = nodes_on_ap.index(node_idx)
            y_pos = start_f + (sub_idx * slot_h)
            
            rect = patches.Rectangle((t, y_pos), 1, slot_h, linewidth=1, edgecolor=edge_c, facecolor=face_c, alpha=0.9)
            ax.add_patch(rect)
            ax.text(t+0.5, y_pos+slot_h/2, label_text, ha='center', va='center', fontsize=6, color=text_col, weight=font_w)

    for ap in range(num_ap):
        end = (ap + 1) * base_sc + min(ap + 1, rem_sc)
        ax.axhline(y=end, color='gray', linestyle='--')
        ax.text(-0.5, end - base_sc/2, f"AP {ap}", va='center', ha='right', fontweight='bold')

    plt.tight_layout()
    plt.show()