import numpy as np
from gudhi.cubical_complex import CubicalComplex

def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    vmin, vmax = np.min(x), np.max(x)
    return (x - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(x, float)

def quantize_levels(arr: np.ndarray, k: int | None):
    if not k or k <= 1:
        return arr
    vmin, vmax = arr.min(), arr.max()
    if vmax <= vmin + 1e-12:
        return arr
    step = (vmax - vmin) / (k - 1)
    return np.round((arr - vmin) / step) * step + vmin

def _finite_valid(arr: np.ndarray) -> np.ndarray:
    if arr is None or arr.size == 0: return np.empty((0, 2), float)
    arr = np.asarray(arr, float)
    m = np.isfinite(arr).all(axis=1) & (arr[:, 1] > arr[:, 0])
    return arr[m]

def cubical_complex(anomaly_map: np.ndarray, k_levels: int | None = None):
    a = quantize_levels(anomaly_map, k_levels)

    cc_sub = CubicalComplex(dimensions=a.shape, top_dimensional_cells=a.flatten(order="F"))
    cc_sub.compute_persistence()
    sub_h1 = _finite_valid(cc_sub.persistence_intervals_in_dimension(1))

    inv = 1.0 - a
    cc_sup = CubicalComplex(dimensions=inv.shape, top_dimensional_cells=inv.flatten(order="F"))
    cc_sup.compute_persistence()
    raw_h0 = _finite_valid(cc_sup.persistence_intervals_in_dimension(0))
    if raw_h0.size:
        sup_h0 = np.stack([1.0 - raw_h0[:, 1], 1.0 - raw_h0[:, 0]], axis=1).astype(float)
        sup_h0 = sup_h0[(sup_h0[:, 1] > sup_h0[:, 0]) & np.isfinite(sup_h0).all(axis=1)]
    else:
        sup_h0 = np.empty((0, 2), float)

    def _sort_desc(arr):
        if arr.size == 0: return arr
        return arr[np.argsort((arr[:, 1] - arr[:, 0]))[::-1]]

    return {'sub_H1': _sort_desc(sub_h1), 'sup_H0': _sort_desc(sup_h0)}
