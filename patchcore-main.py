import os, argparse, pickle
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.mvtec import MVTecDataset

from configs.patchcore import get_config

from TDA.cubical_complexes import normalize01, cubical_complex

from OT.helpers import pd_list
from OT.intra   import intra_components
from OT.inter   import inter_component

from utils.mask import mask


from TTT.mlp import MLPEncoder
from TTT.losses import ContrastiveLoss, ot_consistency_loss


from utils.metrics import compute_metrics_binary
from utils.contrastive import build_contrastive_pairs


def parse_args():
    p = argparse.ArgumentParser(
        description="PatchCore TopoPT arguments."
    )
    p.add_argument('class_name', type=str, help='MVTec-AD class name')

    return p.parse_args()

def main():
    args = parse_args()
    cfg = get_config()

    CLASS_NAME = args.class_name.lower()
    ANOMALY_NPY_DIR = cfg.anomaly_npy_dir
    DATASET_PATH =  cfg.dataset_path
    print(f"[INFO] Using dataset path: {DATASET_PATH}")

    FILT_STEPS = list(cfg.filtration_steps_list)

    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

    # --- Load Features for MLP ---
    features_pkl = os.path.join(cfg.saved_features_path, f"test_{CLASS_NAME}.pkl")
    if not os.path.exists(features_pkl):
        raise FileNotFoundError(f"Test features not found for class {CLASS_NAME}: {features_pkl}")
    with open(features_pkl, 'rb') as f:
        test_data = pickle.load(f)
    embedding_vectors = torch.tensor(test_data['features'], dtype=torch.float32) 

    # Dataset/Loader
    test_dataset = MVTecDataset(dataset_path=DATASET_PATH, class_name=CLASS_NAME, is_train=False)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # METRICS (only for refined mask)
    per_image_metrics = {'precision': [], 'recall': [], 'f1': []}

    print(f"[INFO] Starting pipeline for class '{CLASS_NAME}'")

    for idx, (image, anomaly_label, gt_mask, defect_type) in tqdm(
            enumerate(test_loader), total=len(test_loader), desc="Processing samples"):
        if int(anomaly_label.item()) == 0:
            continue

        npy_filename = f"{CLASS_NAME}{idx}.npy"
        anomaly_map_path = os.path.join(ANOMALY_NPY_DIR, CLASS_NAME, npy_filename)
        if not os.path.exists(anomaly_map_path):
            print(f"[WARN] Missing anomaly map: {anomaly_map_path} — skipping")
            continue

        anomaly_map = normalize01(np.load(anomaly_map_path))

        # Match GT to anomaly_map size
        gt_np = gt_mask.squeeze().cpu().numpy()
        if gt_np.shape != anomaly_map.shape:
            H, W = anomaly_map.shape
            gt_img = Image.fromarray((gt_np * 255).astype(np.uint8))
            gt_img = gt_img.resize((W, H), resample=Image.NEAREST)
            gt_np = (np.array(gt_img) > 127).astype(np.uint8)

        # PDs for all filters
        pd_dicts_steps = [cubical_complex(anomaly_map, k_levels=k) for k in (FILT_STEPS + [None])]

        sub_pd_list = pd_list(pd_dicts_steps, 'sub')
        sup_pd_list = pd_list(pd_dicts_steps, 'sup')

        # INTRA top-1 per level
        sub_best = intra_components(
            sub_pd_list, reg=cfg.sinkhorn_reg, iters=cfg.sinkhorn_iter,
            mode=cfg.intra_rank_mode, wS=cfg.wS, wP=cfg.wP, tiny=cfg.tiny
        )
        sup_best = intra_components(
            sup_pd_list, reg=cfg.sinkhorn_reg, iters=cfg.sinkhorn_iter,
            mode=cfg.intra_rank_mode, wS=cfg.wS, wP=cfg.wP, tiny=cfg.tiny
        )

        # INTER: single best component overall
        best_item = inter_component(
            sub_best, sup_best, reg=cfg.sinkhorn_reg, iters=cfg.sinkhorn_iter,
            mode=cfg.cross_rank_mode, wS=cfg.wS, wP=cfg.wP
        )

        # Final mask (in postprocess module)
        final_mask = mask(
            anomaly_map, best_item,
            sub_h1_delta=cfg.sub_h1_delta,
            sup_h0_delta=cfg.sup_h0_delta
        ).astype(np.uint8)

        # ====== MLP refinement ======
        feat_tensor = embedding_vectors[idx] 
        Ht, Wt = final_mask.shape
        if feat_tensor.shape[1:] != (Ht, Wt):
            feats_resized = F.interpolate(
                feat_tensor.unsqueeze(0), size=(Ht, Wt),
                mode='bilinear', align_corners=False
            ).squeeze(0)
        else:
            feats_resized = feat_tensor

        feats_np = np.transpose(feats_resized.numpy(), (1, 2, 0)).reshape(-1, feats_resized.shape[0])
        labs_np  = final_mask.reshape(-1).astype(np.float32)

        if labs_np.max() == labs_np.min():
            refined_mask = final_mask.copy()
        else:
            feats = torch.from_numpy(feats_np).float()   
            labs  = torch.from_numpy(labs_np).float()   

            from TTT.mlp import MLPEncoder
            from TTT.losses import ContrastiveLoss, ot_consistency_loss
            from utils.contrastive import build_contrastive_pairs

            ds_pairs = build_contrastive_pairs(feats, labs)
            loader = DataLoader(ds_pairs, batch_size=512, shuffle=True, pin_memory=USE_CUDA)

            mlp = MLPEncoder(in_features=feats.shape[1],
                             out_features=cfg.mlp_embed).to(DEVICE)
            opt = torch.optim.Adam(mlp.parameters(), lr=cfg.mlp_lr)
            contrastive_criterion = ContrastiveLoss(margin=cfg.ctr_margin).to(DEVICE)
            alpha, beta = 0.5, 0.5

            mlp.train()
            for _ in range(cfg.mlp_epochs):
                opt.zero_grad()
                feats_dev = feats.to(DEVICE, non_blocking=True)
                emb_full  = mlp(feats_dev)
                loss_ot   = ot_consistency_loss(
                    emb_full, labs.reshape(Ht, Wt),
                    reg=cfg.ot_reg_embed, max_iter=cfg.ot_iter_embed
                )
                (alpha * loss_ot).backward()
                num_batches = max(1, len(loader))
                for x1, x2, y in loader:
                    x1 = x1.to(DEVICE, non_blocking=True)
                    x2 = x2.to(DEVICE, non_blocking=True)
                    y  = y.to(DEVICE, non_blocking=True)
                    out1 = mlp(x1); out2 = mlp(x2)
                    loss_ctr = contrastive_criterion(out1, out2, y)
                    (beta / num_batches * loss_ctr).backward()
                opt.step()

            # Inference
            mlp.eval()
            with torch.no_grad():
                emb_all = mlp(feats.to(DEVICE, non_blocking=True))  # (P,D)
                labs_dev = labs.to(DEVICE)
                bg = emb_all[labs_dev == 0]; fg = emb_all[labs_dev == 1]
                if bg.numel() == 0: bg = emb_all
                if fg.numel() == 0: fg = emb_all
                proto_bg = F.normalize(bg.mean(dim=0, keepdim=True), dim=1).squeeze(0)
                proto_fg = F.normalize(fg.mean(dim=0, keepdim=True), dim=1).squeeze(0)
                d_bg = torch.norm(emb_all - proto_bg[None, :], dim=1)
                d_fg = torch.norm(emb_all - proto_fg[None, :], dim=1)
                pred_flat = (d_fg < d_bg).detach().cpu().numpy().astype(np.uint8)
            refined_mask = pred_flat.reshape(Ht, Wt).astype(np.uint8)

        # ====== METRICS (only for refined mask) ======
        m = compute_metrics_binary(gt_np, refined_mask)
        per_image_metrics['precision'].append(m['precision'])
        per_image_metrics['recall'].append(m['recall'])
        per_image_metrics['f1'].append(m['f1'])

    def mean_or_nan(xs):
        xs = [x for x in xs if x == x]
        return float(np.mean(xs)) if xs else float('nan')

    print("\n=========================== RESULTS (PatchCore) ===============================")
    print(f"Class: {CLASS_NAME}")
    P_mean = mean_or_nan(per_image_metrics['precision'])
    R_mean = mean_or_nan(per_image_metrics['recall'])
    F_mean = mean_or_nan(per_image_metrics['f1'])
    print(f"[METRICS] Precision={P_mean:.4f} | Recall={R_mean:.4f} | F1={F_mean:.4f}")
    print("==================================================================================")

if __name__ == "__main__":
    main()
