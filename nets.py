# ===========================================================
# nets.py — Heterogeneous GNN con device + AP
# ===========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import HeteroGraphConv, GraphConv


# ===========================================================
# CSI Encoder (uguale alla versione precedente)
# ===========================================================

class CSIEncoder(nn.Module):
    """
    Converte CSI 3D (FREQ x TIME x 2) in embedding 1D.
    Ora gestisce CSI completo: magnitude + phase.
    """
    def __init__(self, freq=32, time=20, channels=2, hidden=64):
        super().__init__()

        # Adesso abbiamo 2 canali in ingresso (mag + fase)
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

            # estrai CSI per AP specifico → [N, F, T, 2]
            x = csi[:, ap]

            # permuta per CNN → [N, 2, F, T]
            x = x.permute(0, 3, 1, 2)

            x = self.conv(x)
            x = torch.flatten(x, start_dim=1)
            x = self.fc(x)
            embeds.append(x)

        # output finale: [N, AP, hidden]
        return torch.stack(embeds, dim=1)


# ===========================================================
# Node Encoder
# ===========================================================

class DeviceEncoder(nn.Module):
    def __init__(self, ap_count, hidden=64):
        super().__init__()

        self.pos_fc = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU()
        )

        # CSI embedding produce [N, AP, hidden]
        # Flatten + FC
        self.csi_fc = nn.Sequential(
            nn.Linear(ap_count * hidden, 64),
            nn.ReLU()
        )

        self.merge = nn.Sequential(
            nn.Linear(16 + 64, 64),
            nn.ReLU()
        )

    def forward(self, pos, csi_embed):
        pos_f = self.pos_fc(pos)
        flatten_csi = csi_embed.flatten(1)
        csi_f = self.csi_fc(flatten_csi)

        return self.merge(torch.cat([pos_f, csi_f], dim=1))


class APEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap_fc = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU()
        )

    def forward(self, ap_pos):
        return self.ap_fc(ap_pos)


# ===========================================================
# Heterogeneous GNN
# ===========================================================

class IndustrialMAC_HeteroGNN(nn.Module):

    def __init__(self, num_ap=3, time_slots=20):
        super().__init__()

        self.csi_encoder = CSIEncoder()
        self.device_encoder = DeviceEncoder(num_ap)
        self.ap_encoder = APEncoder()

        # Heterogeneous convolution layers
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
            nn.Sigmoid()
        )

        # AP selection head
        self.ap_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_ap)
        )

    def forward(self, g, device_pos, ap_pos, csi):
        """
        g           : grafo eterogeneo DGL
        device_pos  : [N_dev, 2]
        ap_pos      : [N_ap, 2]
        csi         : [N_dev, N_ap, F, T]
        """

        # Encode nodi device
        csi_embed = self.csi_encoder(csi)
        h_device = self.device_encoder(device_pos, csi_embed)

        # Encode nodi AP
        h_ap = self.ap_encoder(ap_pos)

        # Put in dict for HeteroGraphConv
        h = {
            'device': h_device,
            'ap': h_ap
        }

        # Heterogeneous message passing
        h = self.conv1(g, h)
        h = {k: F.relu(v) for k, v in h.items()}

        h = self.conv2(g, h)
        h = {k: F.relu(v) for k, v in h.items()}

        # Readout SOLO per i device
        sched = self.sched_head(h['device'])
        ap_logits = self.ap_head(h['device'])
        ap_probs = F.softmax(ap_logits, dim=1)

        return sched, ap_probs
