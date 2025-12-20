import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import HeteroGraphConv, GraphConv

from data_generation import NUM_AP, TIME_SLOTS, FREQ_SUBCARRIERS

"""
Step 1: Raw data (CSI) -> CNN (CSIEncoder) -> Vector Feature 
Step 2: Vector Feature + Position + Packets -> GNN (HeteroGraphConv) -> Final Decision
"""

class CSIEncoder(nn.Module):
    """
    Convert CSI 3D (FREQ x TIME x 2) in embedding 1D
    """
    def __init__(self, freq=FREQ_SUBCARRIERS, time=TIME_SLOTS, channels=2, hidden=64):
        super().__init__()

        # Simple CNN with 2 convolutional layers (conv) and 1 Fully Connected layer (fc)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * freq * time, hidden),
            nn.ReLU()
        )

    def forward(self, csi):
        """
        csi shape: [N, AP, F, T, 2]
        """
        N, AP, F, T, C = csi.shape

        embeds = []
        for ap in range(AP):

            # extract CSI for specific AP → [N, F, T, 2]
            x = csi[:, ap]

            # permute for CNN → [N, 2, F, T]
            x = x.permute(0, 3, 1, 2)

            x = self.conv(x)
            x = torch.flatten(x, start_dim=1)
            x = self.fc(x)
            embeds.append(x)

        # final output: [N, AP, hidden]
        return torch.stack(embeds, dim=1)


class DeviceEncoder(nn.Module):
    """
    Convert device position + num_packets + CSI embedding into device node features as embedding
    Use it to encode device nodes in the graph
    """
    def __init__(self, ap_count, hidden=64):
        super().__init__()

        # MLP for position + num_packets
        self.pos_fc = nn.Sequential(
            nn.Linear(3, 16), # 3 for pos(x,y) + num_packets
            nn.ReLU()
        )
        # MLP for CSI embedding
        self.csi_fc = nn.Sequential(
            nn.Linear(ap_count * hidden, 64),
            nn.ReLU()
        )
        # MLP to merge both
        self.merge = nn.Sequential(
            nn.Linear(16 + 64, 64),
            nn.ReLU()
        )
    # Forward method to process inputs
    def forward(self, pos, csi_embed, node_packets):
        packets_in = node_packets.view(-1, 1).float()
        inputs = torch.cat([pos, packets_in], dim=1)
        pos_f = self.pos_fc(inputs)
        flatten_csi = csi_embed.flatten(1)
        csi_f = self.csi_fc(flatten_csi)

        return self.merge(torch.cat([pos_f, csi_f], dim=1))


class APEncoder(nn.Module):
    """_summary_
    Convert AP position (x,y) into AP node features as embedding
    Use it to encode AP nodes in the graph
    """
    def __init__(self):
        super().__init__()
        # MLP for AP position
        self.ap_fc = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU()
        )

    def forward(self, ap_pos):
        return self.ap_fc(ap_pos)


class IndustrialMAC_HeteroGNN(nn.Module):
    """
    Heterogeneous GNN for Industrial IoT MAC Scheduling
    1. Encode node features (device and AP)
    2. Apply heterogeneous graph convolutions
    3. Output scheduling and AP selection logits
    4. Use Gumbel-Softmax for differentiable AP selection
    5. Return hard scheduling decisions and logits for training
    6. Designed for multiple APs and time slots
    7. Training mode: stochastic Gumbel-Softmax
    8. Evaluation mode: deterministic argmax
    """
    def __init__(self, num_ap=NUM_AP, time_slots=TIME_SLOTS):
        super().__init__()

        self.csi_encoder = CSIEncoder()
        self.device_encoder = DeviceEncoder(num_ap)
        self.ap_encoder = APEncoder()

        # GNN layers using HeteroGraphConv from DGL
        self.conv1 = HeteroGraphConv({
        ('device','dd','device'): GraphConv(64, 64),
        ('device','da','ap'):     GraphConv(64, 64),
        ('ap','ad','device'):     GraphConv(64, 64),
    }, aggregate='sum')

        self.conv2 = HeteroGraphConv({
        ('device','dd','device'): GraphConv(64, 64),
        ('device','da','ap'):     GraphConv(64, 64),
        ('ap','ad','device'):     GraphConv(64, 64),
    }, aggregate='sum')

        # Scheduling head (only device nodes)
        self.sched_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, time_slots),
        )

        # AP selection head
        self.ap_head = nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, time_slots * num_ap) # Output logits per time slot and AP
    )
        self.time_slots = time_slots
        self.num_ap = num_ap


    def forward(self, g, device_pos, ap_pos, csi, node_packets):
        # encode node features
        # csi shape: [N, AP, F, T, 2]
        csi_embed = self.csi_encoder(csi)
        h_device = self.device_encoder(device_pos, csi_embed, node_packets)
        h_ap = self.ap_encoder(ap_pos)

        h = {"device": h_device, "ap": h_ap}

        h = self.conv1(g, h)
        h = {k: F.relu(v) for k, v in h.items()}

        h = self.conv2(g, h)
        h = {k: F.relu(v) for k, v in h.items()}

        # logits shape: [N, T]
        ap_logits = self.ap_head(h["device"])  # [N, T*A]
        ap_logits = ap_logits.view(-1, self.time_slots, self.num_ap)  # [N, T, A]


        # Step 1 — Think from AP perspective: we want EACH AP to choose a node.
        # Transpose to [A, T, N]
        logits_AP = ap_logits.permute(2, 1, 0)

        # Step 2 — Straight Through Gumbel Softmax but deterministic in eval        
        if self.training:
            # --- TRAINING MODE (Stochastic) ---
            # Use Gumbel-Softmax with noise for exploration and gradients.
            # hard=False gives soft vectors for backprop.
            ap_soft_AP = F.gumbel_softmax(logits_AP, tau=0.2, hard=False, dim=-1)
            
            # for hard decision, take ARGMAX
            index = ap_soft_AP.argmax(dim=-1, keepdim=True)
            ap_hard_AP = torch.zeros_like(ap_soft_AP).scatter_(-1, index, 1.0)
            ap_onehot_AP = ap_hard_AP - ap_soft_AP.detach() + ap_soft_AP

        else:
            # --- EVAL MODE (Deterministic) ---
            # No Gumbel noise. Take ARGMAX of logits.
            # This ensures: Same Input -> Same Output.
            
            # Compute clean Softmax (no noise) for completion loss (optional in test)
            ap_soft_AP = F.softmax(logits_AP, dim=-1)

            # Deterministic hard decision
            index = logits_AP.argmax(dim=-1, keepdim=True)
            ap_onehot_AP = torch.zeros_like(logits_AP).scatter_(-1, index, 1.0)

        # Step 3 — Return to format [N, T, A]
        ap_onehot_final = ap_onehot_AP.permute(2, 1, 0)
        
        # [A, T, N] -> sum(0) -> [T, N] -> permute -> [N, T]
        sched_soft = ap_soft_AP.sum(dim=0).permute(1, 0)

        # scheduling hard decisions: [N, T]
        # A node is active if at least one AP has chosen it
        sched_hard = (ap_onehot_final.sum(dim=-1) > 0).float()  # [N, T]

        # ==============================
        # OUTPUT
        # ==============================
        # - sched_hard is binary (only for metrics)
        # - ap_logits used in training for CE loss (differentiable)
        # - sched_soft used in training for completion loss (differentiable)
        return sched_hard, ap_logits, sched_soft
