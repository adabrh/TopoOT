import numpy as np
from .sinkhorn import sinkhorn_plan_pd
from .helpers import score_and_sort

def intra_components(
    pd_list,
    reg=1e-2,
    iters=10000,
    mode='and_product',
    wS=1.0,
    wP=0.5,
    tiny=1e-12,
):
    if not pd_list:
        return []

    s_best = [np.zeros(item['points'].shape[0], dtype=float) for item in pd_list]
    c_best = [np.full(item['points'].shape[0], np.inf, dtype=float) for item in pd_list]

    for k in range(len(pd_list)):
        for l in range(k + 1, len(pd_list)):
            PDk = pd_list[k]['points']; PDl = pd_list[l]['points']
            if PDk.size == 0 or PDl.size == 0:
                continue
            _, C, S = sinkhorn_plan_pd(PDk, PDl, reg=reg, numItermax=iters)
            if S is None:
                continue

            # update k side
            j_best_k = np.argmax(S, axis=1)
            Sk = S[np.arange(PDk.shape[0]), j_best_k]
            Ck = C[np.arange(PDk.shape[0]), j_best_k]
            improve_k = Sk > s_best[k] + tiny
            tie_k     = np.abs(Sk - s_best[k]) <= tiny
            c_better  = Ck < c_best[k]
            s_best[k][improve_k] = Sk[improve_k]
            c_best[k][improve_k] = Ck[improve_k]
            c_best[k][tie_k & c_better] = Ck[tie_k & c_better]

            # update l side
            i_best_l = np.argmax(S, axis=0)
            Sl = S[i_best_l, np.arange(PDl.shape[0])]
            Cl = C[i_best_l, np.arange(PDl.shape[0])]
            improve_l = Sl > s_best[l] + tiny
            tie_l     = np.abs(Sl - s_best[l]) <= tiny
            c_better2 = Cl < c_best[l]
            s_best[l][improve_l] = Sl[improve_l]
            c_best[l][improve_l] = Cl[improve_l]
            c_best[l][tie_l & c_better2] = Cl[tie_l & c_better2]

    # flatten + score
    items = []
    for k, item in enumerate(pd_list):
        P = item['points']; L = item['labels']
        for i in range(P.shape[0]):
            b, d = float(P[i, 0]), float(P[i, 1])
            if d <= b:
                continue
            lbl = L[i] if i < len(L) else 'H0'
            pers = d - b
            items.append({
                'step_idx': k, 'comp_idx': i, 'label': lbl,
                'birth': b, 'death': d, 'pers': pers,
                'Sbest': float(s_best[k][i]), 'Cbest': float(c_best[k][i]),
            })

    items = score_and_sort(items, mode=mode, wS=wS, wP=wP, tiny=tiny)
    return items[:10]
