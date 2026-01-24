import numpy as np
from .sinkhorn import sinkhorn_plan_pd
from .helpers import score_and_sort

def inter_component(sub_items, sup_items, reg, iters, mode, wS=1.0, wP=0.5, final_k=1, tiny=1e-12):
    """
    Cross-level selection between SUB(H1) list and SUPER(H0) list.
    Returns a LIST of up to final_k dicts, each augmented with:
      filt ('sublevel'/'superlevel'), Sbest, Cbest, score
    """
    # Normalize inputs to lists
    if sub_items is None: sub_items = []
    if sup_items is None: sup_items = []
    if isinstance(sub_items, dict): sub_items = [sub_items]
    if isinstance(sup_items, dict): sup_items = [sup_items]

    # Nothing to select
    if len(sub_items) == 0 and len(sup_items) == 0:
        return []

    # If one side empty → take from the other directly
    if len(sub_items) == 0:
        out = [dict(m, filt='superlevel', Sbest=0.0, Cbest=float('inf')) for m in sup_items]
        out = score_and_sort(out, mode=mode, wS=wS, wP=wP, tiny=tiny)
        return out[:max(1, int(final_k))]

    if len(sup_items) == 0:
        out = [dict(m, filt='sublevel', Sbest=0.0, Cbest=float('inf')) for m in sub_items]
        out = score_and_sort(out, mode=mode, wS=wS, wP=wP, tiny=tiny)
        return out[:max(1, int(final_k))]

    # Build PD arrays
    P_sub = np.array([[m['birth'], m['death']] for m in sub_items], dtype=float)
    P_sup = np.array([[m['birth'], m['death']] for m in sup_items], dtype=float)

    _, C, S = sinkhorn_plan_pd(P_sub, P_sup, reg=reg, numItermax=iters)

    scored = []
    # Score SUB items by best match to any SUPER
    for i, m in enumerate(sub_items):
        if S is not None and S.size:
            j = int(np.argmax(S[i, :]))
            Sbest = float(S[i, j]); Cbest = float(C[i, j])
        else:
            Sbest, Cbest = 0.0, float('inf')
        mm = dict(m); mm['filt'] = 'sublevel'; mm['Sbest'] = Sbest; mm['Cbest'] = Cbest
        scored.append(mm)

    # Score SUPER items by best match to any SUB
    for j, m in enumerate(sup_items):
        if S is not None and S.size:
            i = int(np.argmax(S[:, j]))
            Sbest = float(S[i, j]); Cbest = float(C[i, j])
        else:
            Sbest, Cbest = 0.0, float('inf')
        mm = dict(m); mm['filt'] = 'superlevel'; mm['Sbest'] = Sbest; mm['Cbest'] = Cbest
        scored.append(mm)

    scored = score_and_sort(scored, mode=mode, wS=wS, wP=wP, tiny=tiny)
    return scored[0]

