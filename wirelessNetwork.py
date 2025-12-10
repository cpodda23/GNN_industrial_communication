import numpy as np
import torch
import dgl
from data_generation import NUM_NODES, NUM_AP

# Parameters
AREA_SIZE = 50        # meters, factory area
DIST_THRESHOLD = 30   # distance threshold for device-device edges

# =============================================================
# Topology generation (device + AP)
# =============================================================
def generate_topology():
    nodes_pos = np.random.uniform(0, AREA_SIZE, size=(NUM_NODES, 2))

    ap_pos = np.array([
        [0, 0],
        [AREA_SIZE, 0],
        [AREA_SIZE/2, AREA_SIZE]
    ])

    return nodes_pos, ap_pos


# =============================================================
# Model industrial pathloss in indoor environment
# =============================================================
def pathloss(d):
    PL0 = -30
    n = 2.2
    return PL0 + 10 * n * np.log10(d + 1e-6)


# =============================================================
# Construction of a HETEROGENEOUS graph for DGL
# =============================================================
def build_hetero_graph(nodes_pos, ap_pos):
    N = nodes_pos.shape[0]
    A = ap_pos.shape[0]

    # Edge lists for each relation
    dd_src, dd_dst = [], []     # device → device
    da_src, da_dst = [], []     # device → ap
    ad_src, ad_dst = [], []     # ap → device (optional but very useful)

    # ----------- EDGE DEVICE → DEVICE -------------
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = np.linalg.norm(nodes_pos[i] - nodes_pos[j])
            if d < DIST_THRESHOLD:
                dd_src.append(i)
                dd_dst.append(j)

    # ----------- EDGE DEVICE → AP ---------------
    for i in range(N):
        for ap in range(A):
            da_src.append(i)
            da_dst.append(ap)

            ad_src.append(ap)
            ad_dst.append(i)  # reverse edge

    # CONSTRUCTION OF HETEROGENEOUS GRAPH
    graph_data = {
        ('device', 'dd', 'device'): (torch.tensor(dd_src), torch.tensor(dd_dst)),
        ('device', 'da', 'ap'):     (torch.tensor(da_src), torch.tensor(da_dst)),
        ('ap', 'ad', 'device'):     (torch.tensor(ad_src), torch.tensor(ad_dst)),
    }

    g = dgl.heterograph(graph_data)

    # ----------- CALCULATION OF EDGE FEATURES (PATHLOSS) ------------

    # DEVICE → DEVICE
    dd_pl = []
    for s, t in zip(dd_src, dd_dst):
        d = np.linalg.norm(nodes_pos[s] - nodes_pos[t])
        dd_pl.append([pathloss(d)])
    g.edges['dd'].data['pl'] = torch.tensor(dd_pl, dtype=torch.float)

    # DEVICE → AP
    da_pl = []
    for s, t in zip(da_src, da_dst):
        d = np.linalg.norm(nodes_pos[s] - ap_pos[t])
        da_pl.append([pathloss(d)])
    g.edges['da'].data['pl'] = torch.tensor(da_pl, dtype=torch.float)

    # AP → DEVICE
    ad_pl = []
    for s, t in zip(ad_src, ad_dst):
        d = np.linalg.norm(ap_pos[s] - nodes_pos[t])
        ad_pl.append([pathloss(d)])
    g.edges['ad'].data['pl'] = torch.tensor(ad_pl, dtype=torch.float)

    return g


# =============================================================
# Main function to create the wireless environment
# =============================================================
def create_wireless_environment():
    nodes_pos, ap_pos = generate_topology()
    g = build_hetero_graph(nodes_pos, ap_pos)

    return {
        "nodes_pos": nodes_pos,
        "ap_pos": ap_pos,
        "graph": g
    }

if __name__ == "__main__":
    env = create_wireless_environment()
    print("Heterogeneous graph:", env["graph"])
    print("Nodes types:", env["graph"].ntypes)
    print("Edges types:", env["graph"].etypes)
