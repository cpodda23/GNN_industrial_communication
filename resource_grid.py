import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

def get_ap_range(ap_idx, total_subcarriers, num_aps):
    """
    Calculate the frequency subcarrier range for a given AP index.
    (Reply of data_generation.py)
    """
    base = total_subcarriers // num_aps
    remainder = total_subcarriers % num_aps
    
    start = ap_idx * base + min(ap_idx, remainder)
    end = start + base + (1 if ap_idx < remainder else 0)
    
    return start, end

def visualize_grid(pred_sched, pred_ap_logits, num_subcarriers, time_slots, title):
    # Convert in numpy if necessary
    if isinstance(pred_sched, torch.Tensor):
        pred_sched = pred_sched.cpu().detach().numpy()
    if isinstance(pred_ap_logits, torch.Tensor):
        pred_ap_logits = pred_ap_logits.cpu().detach().numpy()
        
    num_nodes, _ = pred_sched.shape
    # numer of AP from shape of logits [N, T, A]
    num_aps = pred_ap_logits.shape[2] 
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Set limits and labels
    ax.set_xlim(0, time_slots)
    ax.set_ylim(0, num_subcarriers)
    
    ax.set_xlabel("Time Slots (t)")
    ax.set_ylabel("Frequency Subcarriers (f)")
    ax.set_title(f"{title} Resource Grid")
    
    # Horizontal lines that separate APs
    for ap in range(num_aps):
        start, end = get_ap_range(ap, num_subcarriers, num_aps)
        ax.axhline(y=end, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        # AP Label
        ax.text(-1, start + (end-start)/2, f"AP {ap}", 
                va='center', ha='right', fontsize=10, fontweight='bold', color='gray')

    colors = plt.cm.get_cmap('tab20', num_nodes)
    packet_counters = np.zeros(num_nodes, dtype=int)
    
    # Iterate over time slots
    for t in range(time_slots):
        
        active_nodes = np.where(pred_sched[:, t] > 0.5)[0]
        
        if len(active_nodes) == 0:
            continue
            
        # Collision detection and grouping (more nodes on same AP)
        # create a disctionary {ap_idx: [node1, node2, ...]}
        nodes_per_ap = {}
        
        for node_idx in active_nodes:
            ap_chosen = np.argmax(pred_ap_logits[node_idx, t])
            if ap_chosen not in nodes_per_ap:
                nodes_per_ap[ap_chosen] = []
            nodes_per_ap[ap_chosen].append(node_idx)
            
        # Draw rectangles for each node in their assigned AP
        for ap_chosen, nodes_in_ap in nodes_per_ap.items():
            
            # Get AP frequency range
            y_start_base, y_end_base = get_ap_range(ap_chosen, num_subcarriers, num_aps)
            total_height = y_end_base - y_start_base
            
            slot_height = total_height / len(nodes_in_ap)
            
            # Draw each node's rectangle
            for i, node_idx in enumerate(nodes_in_ap):
                packet_counters[node_idx] += 1
                pkt_num = packet_counters[node_idx]
                
                # Calculate rectangle position y
                y_rect = y_start_base + (i * slot_height)
                
                # Collision Check string
                is_collision = len(nodes_in_ap) > 1
                
                # Design node rectangle
                rect = patches.Rectangle(
                    (t, y_rect), 
                    1, 
                    slot_height,
                    linewidth=1,
                    edgecolor='white',
                    facecolor=colors(node_idx),
                    alpha=0.8
                )
                ax.add_patch(rect)
                
                # Add text inside rectangle
                text_str = f"N{node_idx}\nP{pkt_num}"
                font_weight = 'normal'
                text_col = 'white'
                
                # Text highlight in case of collision
                if is_collision:
                    text_str += "\n!!"
                    font_weight = 'bold'
                    text_col = 'yellow'
                
                ax.text(
                    t + 0.5, y_rect + slot_height/2, 
                    text_str,
                    ha='center', va='center', 
                    fontsize=7, color=text_col, weight=font_weight,
                    clip_on=True
                )

    ax.grid(True, which='both', linestyle=':', alpha=0.3)
    
    # Legend for nodes
    handles = [patches.Patch(color=colors(i), label=f'Node {i}') for i in range(num_nodes)]
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.12, 1))
    
    plt.tight_layout()
    plt.show()