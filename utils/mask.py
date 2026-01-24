import numpy as np

def extract_region_for_feature(anomaly_map, homology_type, birth, death, persistence, filtration_type,
                               sub_h1_delta: float, sup_h0_delta: float):
    if filtration_type == 'sublevel': 
        thr = max(0.0, float(death) - float(sub_h1_delta))
    else:                             
        thr = max(0.0, float(death) - float(sup_h0_delta))
    return (anomaly_map >= thr)

def mask(anomaly_map, feature_item, sub_h1_delta: float, sup_h0_delta: float):
    if feature_item is None:
        return np.zeros_like(anomaly_map, dtype=np.uint8)
    hom = feature_item.get('label', 'H1' if feature_item.get('filt')=='sublevel' else 'H0')
    b   = feature_item['birth']; d = feature_item['death']; p = feature_item['pers']
    filt = feature_item.get('filt', 'sublevel')
    m = extract_region_for_feature(anomaly_map, hom, b, d, p, filt,
                                   sub_h1_delta=sub_h1_delta, sup_h0_delta=sup_h0_delta)
    return m.astype(np.uint8)
