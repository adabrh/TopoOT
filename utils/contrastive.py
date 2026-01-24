import torch
from torch.utils.data import TensorDataset

def build_contrastive_pairs(feats: torch.Tensor, labs: torch.Tensor, rng: torch.Generator | None = None):
    if rng is None:
        rng = torch.Generator()
        rng.manual_seed(1234)
    idx_bg = torch.where(labs == 0)[0]
    idx_fg = torch.where(labs == 1)[0]

    pair1, pair2, pair_lab = [], [], []
    N = feats.shape[0]
    for i in range(N):
        li = int(labs[i].item())
        if li == 0:
            j = idx_bg[torch.randint(len(idx_bg), (1,), generator=rng)].item()
            k = idx_fg[torch.randint(len(idx_fg), (1,), generator=rng)].item() if len(idx_fg)>0 else j
        else:
            j = idx_fg[torch.randint(len(idx_fg), (1,), generator=rng)].item() if len(idx_fg)>0 else i
            k = idx_bg[torch.randint(len(idx_bg), (1,), generator=rng)].item() if len(idx_bg)>0 else i
        pair1.append(feats[i]); pair2.append(feats[j]); pair_lab.append(0.0)
        pair1.append(feats[i]); pair2.append(feats[k]); pair_lab.append(1.0)

    ds_pairs = TensorDataset(torch.stack(pair1), torch.stack(pair2),
                             torch.tensor(pair_lab, dtype=torch.float))
    return ds_pairs
