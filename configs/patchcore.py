from dataclasses import dataclass
from typing import List

@dataclass
class PatchCoreConfig:
    anomaly_npy_dir: str = "" # path to anomaly score dir
    dataset_path: str = "" # path to Dataset
    filtration_steps_list: List[int] = (1000,) 
    sinkhorn_reg: float = 3e-1
    sinkhorn_iter: int = 15000
    tiny: float = 1e-12
    intra_rank_mode: str = "and_product"
    cross_rank_mode: str = "and_product"
    wS: float = 1.0
    wP: float = 0.5
    sub_h1_delta: float = 0.35
    sup_h0_delta: float = 0.17
    saved_features_path: str = "" # Path to saved features
    mlp_epochs: int = 5
    mlp_lr: float = 1e-3
    mlp_hidden: int = 256
    mlp_embed: int = 128
    ot_reg_embed: float = 0.05
    ot_iter_embed: int = 200
    ctr_margin: float = 0.4

def get_config() -> PatchCoreConfig:
    return PatchCoreConfig()
