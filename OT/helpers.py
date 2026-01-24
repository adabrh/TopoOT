import numpy as np

def score_and_sort(items, mode='and_product', wS=1.0, wP=0.5, tiny=1e-12):
    if not items: return items
    Smax = max(it['Sbest'] for it in items) or 1.0
    Pmax = max(it['pers']  for it in items) or 1.0

    def sort_default():
        items.sort(key=lambda it: (-it['Sbest'], it['Cbest'], -it['pers']))

    if mode == 'and_product':
        for it in items:
            it['score'] = (it['Sbest']/Smax) * (it['pers']/Pmax)
        items.sort(key=lambda it: (-it['score'], -it['Sbest'], it['Cbest'], -it['pers']))
        return items
    if mode == 'S_only':
        sort_default(); return items
    if mode == 'S_then_pers':
        sort_default(); return items
    if mode == 'hybrid_add':
        for it in items:
            it['score'] = wS*(it['Sbest']/Smax) + wP*(it['pers']/Pmax)
        items.sort(key=lambda it: (-it['score'], it['Cbest'])); return items
    if mode == 'hybrid_mult':
        for it in items:
            it['score'] = (it['Sbest']/Smax)**wS * (it['pers']/Pmax)**wP
        items.sort(key=lambda it: (-it['score'], -it['Sbest'], it['Cbest'])); return items
    sort_default(); return items

def pd_list(pd_dicts_for_steps, level_key):
    pd_list = []
    for pd_dict in pd_dicts_for_steps:
        if level_key == 'sub':
            P = pd_dict['sub_H1']; L = ['H1'] * len(P)
        else:
            P = pd_dict['sup_H0']; L = ['H0'] * len(P)
        if P.size == 0:
            P = np.empty((0, 2)); L = []
        pd_list.append({'points': P, 'labels': L})
    return pd_list
