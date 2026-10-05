# Comparison of Neural Network Architectures for the development of Microgesture Classification and Online Recognition

Code for my master's thesis on **skeleton-based micro-gesture recognition** with the **SMG dataset** (Spontaneous Micro-Gesture dataset). It covers two tasks:

1. **Offline classification** of pre-segmented micro-gesture clips with graph convolutional networks (ST-GCN, MS-G3D, and CTR-GCN).
2. **Online recognition** of micro-gestures in long, untrimmed recordings with a DNN-HMM pipeline: per-frame emission models (MLP and attention-based BiGRU), HMM priors and transitions, and Viterbi decoding.

**Author:** Diana Cristina Andrade Damian

> The thesis document is not included in this repository. The full text can be found at the University of Padua's archive in the following link: https://hdl.handle.net/20.500.12608/115817

---

## Background

Micro-gestures (MGs) are subtle, spontaneous body movements linked to emotional states such as stress. The SMG dataset (Chen et al., *International Journal of Computer Vision*, 2023) provides Kinect v2 skeleton recordings of spontaneous behaviour, labelled with **16 micro-gestures plus a non-micro-gesture class** (illustrative hand gestures), 17 classes in total. The micro-gesture classes are grouped by psychological basis (freezing, fighting, pacifying, fleeing).

Two properties make the data hard:
- **Extreme class imbalance.** In the training set, two classes (4 and 8) account for about 64% of the samples, while several classes have fewer than 15.
- **Small number of subjects.** Splits are made **by subject** to avoid leakage, since a person tends to repeat the same gestures.

---

## Repository contents

| File | Description |
|---|---|
| `smg-classifier.ipynb` | Offline classification: data inspection, ST-GCN, MS-G3D, and CTR-GCN, training and evaluation |
| `online-recognition.ipynb` | Online recognition: feature extraction, HMM statistics, emission models, Viterbi decoding, evaluation and ablations |

Both notebooks were developed on **Kaggle**, so their paths point to `/kaggle/input/...`. They also import helper modules that are not in this repository (see [Required helper files](#required-helper-files)).

---

## 1. Offline classification (`smg-classifier.ipynb`)

### Data
- Pre-processed **SMGskeleton** release from the original authors: `train_data.npy`, `train_label.pkl`, `test_data.npy`, `test_label.pkl`.
- Tensor layout `(N, C, T, V, M)` = (samples, 3 coordinates, 90 frames, 25 joints, 1 person). Sequences are zero-padded at the end to 90 frames, and coordinates are already centred on the body.
- Splits (by subject): **2,452 training** (first 30 subjects), **637 validation** (remaining 5 subjects of the original training set) and **610 test** samples.
- Includes a skeleton animation helper (Kinect v2 / NTU-RGB+D 25-joint layout).

### Models
| Model | Details |
|---|---|
| **ST-GCN** (own implementation) | 9 ST-GCN blocks (64→64→64→128→128→128→256→256→256 channels), temporal stride 2 in blocks 4 and 7, global average pooling, FC classifier; uniform, distance or spatial graph partitioning; ~3.04M parameters |
| **ST-GCN** (reference implementation) | Original model from the SMG authors' repository, ~3.09M parameters |
| **MS-G3D** | Multi-scale graph 3D network (Liu et al., CVPR 2020), ~3.18M parameters |
| **CTR-GCN** | Channel-wise Topology Refinement network (Chen et al., 2021), ~1.5M parameters |

### Handling class imbalance
- Class-frequency **weighted cross-entropy** (inverse or square-root weights), **focal loss**, and **weighted focal loss**, with optional label smoothing.
- **Weighted random sampler** for balanced mini-batches.
- Mixed-precision training (AMP), AdamW, cosine annealing, early stopping

### Results shown in the notebook (test set, 610 samples, 17 classes)

| Model | Accuracy | Top-5 |
|---|---|---|
| ST-GCN  | 0.4213 | 0.8689 | 
| MS-G3D  | 0.5984 | 0.8721 | 
| CTR-GCN  | 0.5967 | 0.8984 | 

**Observation:** Top-5 accuracy is much higher than top-1 for all models. This reflects the imbalance and the difficulty of generalising to new subjects. Models do well on frequent classes and poorly on rare ones.

---

## 2. Online recognition (`online-recognition.ipynb`)

The task is to detect **where** each micro-gesture starts and ends in a long recording, and **which** gesture it is. It follows the online-recognition baseline of the SMG authors, re-implemented in PyTorch.

### Pipeline

```
skeleton stream -> per-frame features (2,775-d) -> emission model -> state probabilities (86)
                -> attention emission transform -> Viterbi with HMM prior/transitions
                -> (label, begin, end) segments -> matching against ground truth
```

1. **Features per frame (2,775 dimensions):** a spatial pose descriptor of pairwise joint differences within the frame (900) plus a motion descriptor of joint differences between consecutive frames (1,875). Features are standardised using statistics from the training subjects only.
2. **HMM states (86):** each of the 17 classes is a left-to-right chain of 5 sub-states, plus one background state for "no gesture".
3. **Emission model:** a neural network that maps each frame's features to a probability over the 86 states.
4. **HMM prior and transitions:** estimated from the training subjects only.
5. **Decoding:** attention emission transform (μ = −2.5, λ = 2.1), then Viterbi decoding, then cleaning the state path into gesture segments.
6. **Evaluation:** segments are matched to ground truth with an overlap threshold of 0.3, and per-class and overall recall, precision and F1 are computed.

**Subject split:** subjects 0–29 for training, 30–34 for validation, 35–39 for testing (5 test subjects, 610 ground-truth gestures).

### Emission models
| Model | Description |
|---|---|
| **MLP** | 2 hidden layers of 1,000 units, dropout 0.5, batch normalisation |
| **MLP (weighted)** | Same MLP trained with class-weighted cross-entropy |
| **AE-BiLSTM** | Features reshaped to 5 time steps × 555, BiGRU, temporal attention, BiGRU, spatial attention, dense head |
| **AE-BiLSTM (weighted)** | Same model with class-weighted loss |
| **Ablation variants** | Single GRU, single BiGRU, BiGRU with temporal-only or spatial-only attention, stacked BiGRU without attention, and swapped attention order. The ablation run is commented out in the saved version. |

Includes plotting utilities for emission posteriorgrams and for predicted vs ground-truth gesture segments.

### Results (overall, 5 test subjects, overlap threshold 0.3)

| Model | Recall | Precision | F1 |
|---|---|---|---|
| MLP | 0.0311 | 0.3065 | 0.0565 |
| MLP (weighted) | 0.0541 | 0.4177 | 0.0958 |
| AE-BiLSTM | 0.1066 | 0.3403 | **0.1623** |
| AE-BiLSTM (weighted) | 0.1230 | 0.1572 | 0.1380 |

**Observations:**
- The attention-based recurrent model clearly outperforms the MLP baseline.
- Online recognition on this dataset is very hard: only the most frequent classes (mainly class 8, and class 11 for the BiLSTM models) are detected with any reliability, and most rare classes are never recovered.
- Class weighting improves the MLP but raises false positives for the BiLSTM (precision drops from 0.34 to 0.16), so it does not improve F1 there.

---

## Required helper files

The notebooks import code that is not part of this repository. Place these next to the notebooks before running:

- `stgcn_utils.py`: `ConvTemporalGraphical`, `Graph` and `st_gcn`, taken from the SMG authors' repository.
- `msg3d_utils.py`: `AdjMatrixGraph`, `MultiScale_GraphConv`, `MultiScale_TemporalConv`, `SpatialTemporal_MS_GCN`, `UnfoldTemporalWindows`, `MLP`, `activation_factory` and `count_params`, from the [MS-G3D repository](https://github.com/kenziyuliu/MS-G3D).
- `SMGaccessSample.py` (`GestureSample`): the online-recognition notebook clones the [SMG repository](https://github.com/mikecheninoulu/SMG) and copies this file from `online recognition/SMG/`.

## Data

The SMG dataset belongs to its original authors. Obtain it from the [SMG repository](https://github.com/mikecheninoulu/SMG) and follow their terms of use.

- **Offline notebook:** needs the four pre-processed files (`train_data.npy`, `train_label.pkl`, `test_data.npy`, `test_label.pkl`).
- **Online notebook:** needs the raw skeleton recordings, one folder per subject (`experimentWell_skeleton`, 40 folders, each with data, fine-label and skeleton text files).

The notebooks currently read these from Kaggle (`/kaggle/input/datasets/...`) and download a saved ST-GCN checkpoint with `kagglehub`. Update the paths to your own copies.

## How to run

1. Obtain the data and the helper files above.
2. Open the notebook in Kaggle, Colab or Jupyter and update the data paths (`train_path`, `DATA`, `parsers['data']`, and so on).
3. Run the cells in order. A GPU is recommended for the offline notebook. Feature extraction in the online notebook is slow because it processes the full recordings.
4. Cells for the Optuna study and the ablation experiments are commented out. Uncomment them to reproduce those experiments.

### Requirements

- Python 3.10+
- `torch`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`
- `tensorflow` (only used for seeding in the offline notebook), `opencv-python`, `kagglehub`
- `optuna`, `optuna-dashboard`, `plotly` (for the Optuna plots)

```bash
pip install torch numpy scipy pandas scikit-learn matplotlib tensorflow opencv-python kagglehub optuna optuna-dashboard plotly
```

## Notes and limitations

- Results are from a single run with fixed seeds, with no repeated runs or confidence intervals.
- The test sets are small (610 gestures from 5 subjects), and several classes have only a handful of test samples, so per-class metrics are noisy.

## References

- Chen, H., Shi, H., Liu, X., Li, X., & Zhao, G. (2023). SMG: A Micro-gesture Dataset Towards Spontaneous Body Gestures for Emotional Stress State Analysis. *International Journal of Computer Vision*.
- Yan, S., Xiong, Y., & Lin, D. (2018). Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition. *AAAI*.
- Liu, Z., Zhang, H., Chen, Z., Wang, Z., & Ouyang, W. (2020). Disentangling and Unifying Graph Convolutions for Skeleton-Based Action Recognition. *CVPR*.
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. *ICCV*.
- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. *KDD*.
- Chen, Y., Zhang, Z., Yuan, C., Li, B., Deng, Y., & Hu, W. (2021). Channel-wise topology refinement graph convolution for skeleton-based action recognition. arXiv. doi.org.
