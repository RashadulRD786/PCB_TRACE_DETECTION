# PCB Defect Detection with YOLO: A Comparative Study

> A systematic benchmarking of YOLO model families (YOLOv8, YOLO11, YOLO26) for automated Printed Circuit Board (PCB) defect detection using the HRIPCB dataset.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Defect Classes](#defect-classes)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
  - [Phase 1: Baseline — 20 Epochs](#phase-1-baseline--20-epochs)
  - [Phase 2: Extended Training — 50 Epochs](#phase-2-extended-training--50-epochs)
  - [Phase 3: Full Training — 100 Epochs](#phase-3-full-training--100-epochs)
  - [Overall Comparison](#overall-comparison)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Reproducing Experiments](#reproducing-experiments)
- [Environment](#environment)

---

## Overview

This project investigates the effectiveness of modern YOLO architectures for detecting manufacturing defects on Printed Circuit Boards (PCBs). PCB defect detection is a critical quality-control task in electronics manufacturing, where defects are often **small, subtle, and visually similar** to benign surface features.

We conducted **16 controlled experiments** across three model families, two input resolutions (640×640 and 960×960), and multiple training durations (20, 50, and 100 epochs) to answer:

- Which YOLO variant achieves the best detection accuracy on PCB defects?
- Does higher input resolution improve performance on small defects?
- How does training duration affect convergence and final accuracy?
- What is the best trade-off between model size and accuracy?

---

## Dataset

We use the **HRIPCB (High-Resolution Industrial PCB)** dataset — a widely used benchmark for PCB defect detection.

| Split      | Images     | Labels     |
|------------|-----------:|-----------:|
| Train      | 8,534      | 8,534      |
| Validation | 1,066      | 1,066      |
| Test       | 1,068      | 1,068      |
| **Total**  | **10,668** | **10,668** |

**Image Resolution:** 600×600 pixels  
**Annotation Format:** YOLO format — `class_id cx cy w h` (normalized coordinates)  
**Defect Characteristics:** Predominantly small objects (area < 32×32 px²), making this a challenging small-object detection benchmark.

---

## Defect Classes

The dataset contains **6 PCB defect categories**:

| ID | Class Name        | Description                                          |
|----|-------------------|------------------------------------------------------|
| 0  | `missing_hole`    | Required drill holes absent from the PCB            |
| 1  | `mouse_bite`      | Irregular material erosion along PCB edges          |
| 2  | `open_circuit`    | Broken or incomplete conductive traces              |
| 3  | `short`           | Unintended electrical connections between traces    |
| 4  | `spur`            | Excess copper protrusions on traces                 |
| 5  | `spurious_copper` | Unwanted copper deposits in non-conductive regions  |

---

## Experimental Setup

### Models

All experiments use pretrained Ultralytics YOLO models fine-tuned on the HRIPCB dataset:

| Model Family | Variants Tested                                             |
|--------------|-------------------------------------------------------------|
| YOLOv8       | Nano (n), Small (s), Medium (m), Large (l), XLarge (x)     |
| YOLO11       | Small (s), Medium (m)                                       |
| YOLO26       | Small (s)                                                   |

### Training Configuration

| Hyperparameter      | Value                    |
|---------------------|--------------------------|
| Optimizer           | SGD (auto)               |
| Initial LR (`lr0`)  | 0.01                     |
| Final LR (`lrf`)    | 0.01                     |
| Momentum            | 0.937                    |
| Weight Decay        | 0.0005                   |
| Warmup Epochs       | 3                        |
| Batch Size          | 16 (8 for 960×960 runs)  |
| Early Stopping      | patience = 20            |
| AMP                 | Enabled                  |
| Device              | GPU (CUDA 0)             |
| Seed                | 0 (deterministic)        |

### Data Augmentation

| Augmentation     | Value       |
|------------------|-------------|
| HSV Hue shift    | 0.015       |
| HSV Saturation   | 0.7         |
| HSV Value        | 0.4         |
| Translation      | 0.1         |
| Scale            | 0.5         |
| Horizontal Flip  | 0.5         |
| Mosaic           | 1.0         |
| Random Erasing   | 0.4         |
| Auto Augment     | RandAugment |

---

## Results

> **Metrics reported:** Best epoch `mAP50(B)` and `mAP50-95(B)` from `results.csv`.  
> mAP50-95 is the primary ranking metric — it is stricter and more informative than mAP50 alone.  
> **Bold** = best value within each phase.

---

### Phase 1: Baseline — 20 Epochs

Initial architecture sweep before committing to full training budgets.

| Experiment         | Model    | Img Size | Epochs | Precision | Recall | mAP50      | mAP50-95   |
|--------------------|----------|----------|--------|----------:|-------:|-----------:|-----------:|
| yolov8s_960        | YOLOv8-S | 960      | 20     | 0.9786    | 0.9900 | **0.9893** | **0.6245** |
| yolo26s_640        | YOLO26-S | 640      | 20     | 0.9744    | 0.9899 | 0.9883     | 0.6117     |
| yolo11m_640        | YOLO11-M | 640      | 20     | 0.9747    | 0.9873 | 0.9876     | 0.5986     |
| yolo11s_640        | YOLO11-S | 640      | 20     | 0.9730    | 0.9892 | 0.9883     | 0.5934     |

**Takeaway:** Even at 20 epochs all models exceed 98.7% mAP50. YOLOv8-S at 960×960 leads on mAP50-95, pointing to the benefit of higher resolution for small defects.

---

### Phase 2: Extended Training — 50 Epochs

Extending the top configurations to 50 epochs.

| Experiment               | Model    | Img Size | Epochs | Precision | Recall | mAP50      | mAP50-95   |
|--------------------------|----------|----------|--------|----------:|-------:|-----------:|-----------:|
| yolov8s_960_50_epochs    | YOLOv8-S | 960      | 50     | 0.9820    | 0.9921 | **0.9923** | **0.7242** |
| yolov8m_960_50_epochs    | YOLOv8-M | 960      | 50     | 0.9799    | 0.9934 | 0.9913     | 0.7215     |
| yolov8m_640_50_epochs    | YOLOv8-M | 640      | 50     | 0.9814    | 0.9899 | 0.9913     | 0.6885     |
| yolov8s_640_50_epochs    | YOLOv8-S | 640      | 50     | 0.9789    | 0.9923 | 0.9917     | 0.6753     |
| yolo26s_640_50_epochs    | YOLO26-S | 640      | 50     | 0.9800    | 0.9903 | 0.9918     | 0.6646     |

**Takeaway:** 50-epoch training yields **+4–6 pp** on mAP50-95 compared to 20 epochs. The 960×960 resolution variants consistently outperform their 640×640 counterparts on this metric.

---

### Phase 3: Full Training — 100 Epochs

Full training across all major model variants.

| Experiment                 | Model    | Img Size | Epochs | Precision  | Recall | mAP50      | mAP50-95   |
|----------------------------|----------|----------|--------|-----------:|-------:|-----------:|-----------:|
| **yolov8x_640_100_epochs** | YOLOv8-X | 640      | 100    | **0.9894** | 0.9899 | 0.9927     | **0.7665** |
| yolov8l_640_100_epochs     | YOLOv8-L | 640      | 100    | 0.9879     | 0.9887 | 0.9923     | 0.7499     |
| yolov8s_960_100_epochs     | YOLOv8-S | 960      | 100    | 0.9820     | 0.9902 | **0.9933** | 0.7275     |
| yolov8m_640_100_epochs     | YOLOv8-M | 640      | 100    | 0.9867     | 0.9871 | 0.9920     | 0.7281     |
| yolov11s_640_100_epochs    | YOLO11-S | 640      | 100    | 0.9865     | 0.9882 | 0.9929     | 0.6774     |
| yolov8s_640_100_epochs     | YOLOv8-S | 640      | 100    | 0.9853     | 0.9845 | 0.9921     | 0.6845     |
| yolov8n_640_100_epochs     | YOLOv8-N | 640      | 100    | 0.9823     | 0.9891 | 0.9919     | 0.6159     |

**Takeaway:** YOLOv8-X reaches the highest mAP50-95 (76.65%). YOLOv8-S at 960×960 achieves the best raw mAP50 (99.33%). YOLOv8-L offers the best accuracy-to-efficiency balance.

---

### Overall Comparison

All 16 experiments ranked by mAP50-95 (best epoch):

| Rank | Experiment                  | Model    | Img Size | Epochs | mAP50      | mAP50-95   | Precision | Recall |
|-----:|-----------------------------|----------|----------|-------:|-----------:|-----------:|----------:|-------:|
| 1    | yolov8x_640_100_epochs      | YOLOv8-X | 640      | 100    | 0.9927     | **0.7665** | 0.9894    | 0.9899 |
| 2    | yolov8l_640_100_epochs      | YOLOv8-L | 640      | 100    | 0.9923     | 0.7499     | 0.9879    | 0.9887 |
| 3    | yolov8s_960_100_epochs      | YOLOv8-S | 960      | 100    | **0.9933** | 0.7275     | 0.9820    | 0.9902 |
| 4    | yolov8m_640_100_epochs      | YOLOv8-M | 640      | 100    | 0.9920     | 0.7281     | 0.9867    | 0.9871 |
| 5    | yolov8m_960_50_epochs       | YOLOv8-M | 960      | 50     | 0.9913     | 0.7215     | 0.9799    | 0.9934 |
| 6    | yolov8s_960_50_epochs       | YOLOv8-S | 960      | 50     | 0.9923     | 0.7242     | 0.9820    | 0.9921 |
| 7    | yolov8s_640_100_epochs      | YOLOv8-S | 640      | 100    | 0.9921     | 0.6845     | 0.9853    | 0.9845 |
| 8    | yolov8m_640_50_epochs       | YOLOv8-M | 640      | 50     | 0.9913     | 0.6885     | 0.9814    | 0.9899 |
| 9    | yolov11s_640_100_epochs     | YOLO11-S | 640      | 100    | 0.9929     | 0.6774     | 0.9865    | 0.9882 |
| 10   | yolov8s_640_50_epochs       | YOLOv8-S | 640      | 50     | 0.9917     | 0.6753     | 0.9789    | 0.9923 |
| 11   | yolo26s_640_50_epochs       | YOLO26-S | 640      | 50     | 0.9918     | 0.6646     | 0.9800    | 0.9903 |
| 12   | yolov8s_960 (20ep)          | YOLOv8-S | 960      | 20     | 0.9893     | 0.6245     | 0.9786    | 0.9900 |
| 13   | yolov8n_640_100_epochs      | YOLOv8-N | 640      | 100    | 0.9919     | 0.6159     | 0.9823    | 0.9891 |
| 14   | yolo26s_640 (20ep)          | YOLO26-S | 640      | 20     | 0.9883     | 0.6117     | 0.9744    | 0.9899 |
| 15   | yolo11m_640 (20ep)          | YOLO11-M | 640      | 20     | 0.9876     | 0.5986     | 0.9747    | 0.9873 |
| 16   | yolo11s_640 (20ep)          | YOLO11-S | 640      | 20     | 0.9883     | 0.5934     | 0.9730    | 0.9892 |

---

## Key Findings

### 1. Model Size vs. Accuracy

Larger YOLOv8 variants consistently achieve higher mAP50-95. YOLOv8-X (76.65%) outperforms YOLOv8-N (61.59%) by ~15 pp. The ranking within YOLOv8 at 100 epochs follows the expected order: **X > L > M > S > N**. However, all models exceed **99.1% mAP50**, confirming YOLO is well-suited to this task regardless of model size.

### 2. Resolution Matters for Small Defects

Increasing input resolution from 640 to 960 delivers a consistent **+3–5 pp improvement in mAP50-95** with minimal change in mAP50. For example, YOLOv8-S at 960×960 (100 epochs) achieves **72.75%** vs. **68.45%** at 640×640. This directly confirms that higher resolution is beneficial when defects are predominantly small objects.

### 3. Training Duration

A clear progression is seen as training lengthens:
- **20 epochs:** ~59–62% mAP50-95
- **50 epochs:** ~66–72% mAP50-95
- **100 epochs:** ~68–77% mAP50-95

The early stopping mechanism (patience=20) prevents overfitting and makes 100-epoch runs both safe and effective.

### 4. Model Recommendations

| Use Case                              | Model                          | mAP50  | mAP50-95 |
|---------------------------------------|-------------------------------|-------:|---------:|
| Highest overall accuracy              | YOLOv8-X @ 640, 100 ep        | 0.9927 | 0.7665   |
| Best detection coverage (mAP50)       | YOLOv8-S @ 960, 100 ep        | 0.9933 | 0.7275   |
| Best accuracy / efficiency trade-off  | YOLOv8-L @ 640, 100 ep        | 0.9923 | 0.7499   |
| Lightweight / edge deployment         | YOLOv8-N @ 640, 100 ep        | 0.9919 | 0.6159   |

### 5. YOLO11 and YOLO26 Observations

- **YOLO11-S** at 100 epochs (mAP50: 0.9929, mAP50-95: 0.6774) is competitive with **YOLOv8-S** under the same conditions — a strong result for a newer, more compact architecture worth exploring further.
- **YOLO26-S** performs well on mAP50 (0.9918) at 50 epochs but lags on mAP50-95, suggesting it benefits from longer training or hyperparameter tuning specific to this dataset.

---

## Project Structure

```
PCB_YOLO/
├── datasets/
│   └── HRIPCB/
│       ├── data.yaml               # Dataset config (6 classes, paths)
│       ├── train/images & labels/  # 8,534 training samples
│       ├── val/images & labels/    # 1,066 validation samples
│       └── test/images & labels/   # 1,068 test samples
│
├── scripts/
│   ├── train.py                    # Training entry point (Ultralytics YOLO)
│   ├── validate.py                 # Evaluate model on val/test split
│   ├── plot_curves.py              # Plot mAP / loss curves from results.csv
│   ├── compute_ap_small.py         # Custom AP for small objects (< 32×32 px²)
│   └── runs/
│       └── experiments/            # Per-experiment output (weights, logs, plots)
│
├── analysis/
│   ├── class_distribution.py       # Bar chart of per-class instance counts
│   ├── dataset_sanity_check.py     # Validate label format and class ID ranges
│   └── defect_size_analysis.py     # BBox size distribution and aspect ratios
│
└── configs/
    └── train_config.yaml           # Reserved for config-driven training
```

---

## Reproducing Experiments

### Prerequisites

```bash
pip install ultralytics pandas matplotlib pyyaml
```

### Training

All 16 experiments are pre-configured in `configs/train_config.yaml`. Run any of them with a single command from the repo root:

```bash
# List all available experiments
python scripts/train.py --config configs/train_config.yaml --list

# Run the best overall model (YOLOv8-X, mAP50-95: 76.65%)
python scripts/train.py --config configs/train_config.yaml --name yolov8x_640_100_epochs

# Run the best mAP50 model (YOLOv8-S @ 960x960)
python scripts/train.py --config configs/train_config.yaml --name yolov8s_960_100_epochs

# Run any other experiment by name
python scripts/train.py --config configs/train_config.yaml --name yolov8l_640_100_epochs

# Override a parameter on the fly (e.g. reduce epochs for a quick test)
python scripts/train.py --config configs/train_config.yaml --name yolov8x_640_100_epochs --epochs 20
```

You can also run without a config file using explicit flags (original behaviour):

```bash
python scripts/train.py \
  --model yolov8x.pt \
  --data datasets/HRIPCB/data.yaml \
  --epochs 100 --imgsz 640 --batch 16 \
  --name my_custom_run
```

### Validation

```bash
python validate.py \
  --weights runs/experiments/yolov8x_640_100_epochs/weights/best.pt \
  --data ../datasets/HRIPCB/data.yaml \
  --imgsz 640
```

### Visualize Training Curves

```bash
python plot_curves.py \
  --csv runs/experiments/yolov8x_640_100_epochs/results.csv \
  --title "YOLOv8-X 640x640 100 Epochs"
```

### Small Object AP

```bash
# Edit pred_dir in compute_ap_small.py to point to your prediction labels directory
python compute_ap_small.py
```

---

## Environment

| Component        | Details                                                                     |
|------------------|-----------------------------------------------------------------------------|
| Framework        | [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)              |
| Python           | 3.x                                                                         |
| Hardware         | CUDA GPU (device 0)                                                         |
| Mixed Precision  | AMP enabled                                                                 |
| OS               | Linux (Ubuntu)                                                              |

---

*Experiments conducted by Rashadul Nafis Riyad.*
