import numpy as np
import torch
import torch.nn.functional as F
import ot  # POT

def sinkhorn_plan_pd(PD1: np.ndarray, PD2: np.ndarray, reg: float = 1e-2, numItermax: int = 10000):
    if PD1.shape[0] == 0 or PD2.shape[0] == 0:
        return None, None, None
    C = ot.dist(PD1, PD2, metric='euclidean')
    C = (C / (C.max() + 1e-12))**2
    n1, n2 = PD1.shape[0], PD2.shape[0]
    w1 = np.ones(n1) / max(1, n1)
    w2 = np.ones(n2) / max(1, n2)
    G = ot.sinkhorn(w1, w2, C, reg, numItermax=numItermax, stopThr=1e-9)
    S = G / (1.0 + np.sqrt(np.maximum(C, 0.0)))
    return G, C, S

def cosine_cost(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, dim=1); y = F.normalize(y, dim=1)
    sim = x @ y.t()
    return (1.0 - sim).clamp_min(0.0)

def sinkhorn_transport_embeddings(cost: torch.Tensor, reg: float = 0.05, max_iter: int = 200, eps: float = 1e-9) -> torch.Tensor:
    M, N = cost.shape
    K = torch.exp(-cost / reg).clamp_min(eps)
    u = torch.full((M, 1), 1.0 / M, device=cost.device)
    v = torch.full((N, 1), 1.0 / N, device=cost.device)
    for _ in range(max_iter):
        u = (1.0 / M) / (K @ v + eps)
        v = (1.0 / N) / (K.t() @ u + eps)
    return (u * K) * v.t()
