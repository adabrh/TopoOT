# Test-Time Adaptation for Anomaly Segmentation via Topology-Aware Optimal Transport Chaining

## Abstract
Deep topological data analysis (TDA) offers a principled framework for capturing structural invariants such as connectivity and cycles that persist across scales, making it a natural fit for anomaly segmentation (AS). Unlike threshold-based binarisation, which produces brittle masks under distribution shift, TDA allows anomalies to be characterised as disruptions to global structure rather than local fluctuations. We introduce TopoOT, a topology-aware optimal transport (OT) framework that integrates multi-filtration persistence diagrams with test-time adaptation (TTA). Our key innovation is Optimal Transport Chaining, which sequentially aligns persistence diagrams (PDs) across thresholds and filtrations, yielding geodesic stability scores that identify features consistently preserved across scales. These stability-aware pseudo-labels supervise a lightweight head trained online with OT-consistency and contrastive objectives, ensuring robust adaptation under domain shift. Across standard 2D and 3D anomaly detection benchmarks, TopoOT achieves state-of-the-art performance, outperforming the most competitive methods by up to +24.1% mean F1 on 2D datasets and +10.2% on 3D anomaly segmentation benchmarks.

---

## 1) Environment

Create the conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
```

## 2) Datasets (Official Download Links)

* **[MVTec AD (2D)](https://www.mvtec.com/company/research/datasets/mvtec-ad)** 
* **[MVTec 3D-AD](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad)**
* **[VisA (2D)](https://github.com/amazon-research/spot-diff)**
* **[Real-IAD (2D)](https://realiad4ad.github.io/Real-IAD/)** 
* **[Anomaly-ShapeNet (3D)](https://github.com/Chopper-233/Anomaly-ShapeNet)**

After downloading, set the dataset path in config.

---

## 3) Repository structure

```
TopoOT/
├─ anomaly_scores/
│  └─ patchcore/
│     ├─ class1/
│     │  ├─ sample_000.npy
│     │  ├─ sample_001.npy
│     │  └─ ...
│     └─ class2/
│        └─ ...
├─ features/
│  └─ patchcore/
│     ├─ class1/
│     │  └─ features.pkl
│     └─ class2/
│        └─ features.pkl
└─ environment.yml
```

Example run (3D / CMM):
```bash
python patchcore-main.py "classname" i.e: "bottle"
```


## 9) Minimal run checklist

1. `conda env create -f environment.yml && conda activate <env-name>`  
2. Download a dataset (e.g., **MVTec-AD**) and set its path in `configs/patchore.py`.  
3. Export anomaly maps into `anomaly_scores/patchcore/<class>/*.npy`.  
4. Export features into `features/patchcore/<class>/features.pkl`.  
5. Run main script.
