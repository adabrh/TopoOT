import numpy as np
import torch
import torch.nn.functional as F
from OT.sinkhorn import sinkhorn_transport_embeddings, cosine_cost

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    def forward(self, output1, output2, label):
        d = F.pairwise_distance(output1, output2)
        return torch.mean((1 - label) * d.pow(2) + label * torch.clamp(self.margin - d, min=0.0).pow(2))

def build_binary_prototypes(emb: torch.Tensor, labels01: torch.Tensor):
    bg = emb[labels01 == 0]; fg = emb[labels01 == 1]
    if bg.numel() == 0: bg = emb
    if fg.numel() == 0: fg = emb
    proto_bg = F.normalize(bg.mean(dim=0, keepdim=True), dim=1).squeeze(0)
    proto_fg = F.normalize(fg.mean(dim=0, keepdim=True), dim=1).squeeze(0)
    return proto_bg, proto_fg

def ot_consistency_loss(emb: torch.Tensor, anomaly_map_hw: torch.Tensor, reg: float = 0.05, max_iter: int = 200) -> torch.Tensor:
    device = emb.device
    H, W = anomaly_map_hw.shape
    P = H * W
    emb = F.normalize(emb, dim=1)
    A = anomaly_map_hw.reshape(-1, 1).to(device).float()  # (P,1)
    M_cls = torch.cat([1.0 - A, A], dim=1)               # (P,2)
    pseudo = (A.squeeze(1) > 0.5).float()
    proto_bg, proto_fg = build_binary_prototypes(emb, pseudo)
    protos = torch.stack([proto_bg, proto_fg], dim=0)    # (2,D)
    sim = emb @ F.normalize(protos, dim=1).t()           # (P,2)
    M_prot = (sim.clamp(-1, 1) + 1.0) * 0.5             # (P,2)
    C = cosine_cost(emb, protos)                          # (P,2)
    T = sinkhorn_transport_embeddings(C, reg=reg, max_iter=max_iter)  # (P,2)
    gt = T / (T.sum(dim=1, keepdim=True) + 1e-9)         # (P,2)
    H_ent = -(gt * (gt.clamp_min(1e-9)).log()).sum(dim=1, keepdim=True)
    const = 1.0 / np.log(3.0)
    w = (1.0 - const * H_ent).clamp_min(0.0) * gt        # (P,2)
    loss = (w * torch.abs(M_prot - M_cls)).sum() / (P * 2.0 + 1e-9)
    return loss
