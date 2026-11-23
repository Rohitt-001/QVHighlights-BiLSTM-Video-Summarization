# QVHighlights: Query-Focused Video Summarization using BiLSTM

A lightweight recurrent baseline for **query-focused video highlight detection** on the QVHighlights dataset. The model combines pre-extracted CLIP video features with a natural-language query representation and predicts a saliency score for each video clip. The highest-scoring clips are selected as query-relevant highlights.

> **Project type:** Academic minor project / research baseline  
> **Framework:** PyTorch  
> **Execution environment:** Kaggle Notebook with GPU support

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Training Configuration](#training-configuration)
- [Results](#results)
- [Generating Highlights](#generating-highlights)
- [Repository Structure](#repository-structure)
- [Setup and Usage](#setup-and-usage)
- [Limitations](#limitations)
- [Acknowledgements](#acknowledgements)

## Overview

Long videos often contain more information than a viewer needs. This project explores how a natural-language query can guide a model to identify the video clips most relevant to that query.

Given:

1. A video represented by pre-extracted clip-level visual features.
2. A natural-language query represented by a pre-extracted text feature vector.

The model predicts a saliency score for every clip. Clips with the highest scores are selected to form a query-focused highlight sequence.

This implementation uses a **BiLSTM baseline** rather than a large transformer-based moment retrieval architecture. It focuses on learning temporal relationships between video clips while conditioning the sequence on the query representation.

## Problem Statement

Given a video and a natural-language query, estimate the relevance or highlight-worthiness of each video clip with respect to that query. The predicted clip scores are then used to select a shorter set of highlights.

## Dataset

The project uses the **QVHighlights** dataset and its pre-extracted feature files.

| Split | Number of samples |
|---|---:|
| Training | 7,218 |
| Validation | 1,550 |
| Test | 1,542 |
| **Total** | **10,310** |

The notebook uses these feature directories:

- `clip_features`: clip-level CLIP video representations (512 dimensions per clip).
- `clip_text_features`: query representations; the notebook uses `pooler_output` (512 dimensions).

The dataset package also contains other feature directories, including SlowFast, CLIP subtitle and PANN features. **The current model uses CLIP video and CLIP text features only.**

The test annotations supplied to this notebook contain query/video metadata but no ground-truth highlight labels. Test predictions are therefore generated for inference, not for calculating labelled test metrics.

## Methodology

1. **Load annotations:** Read the training, validation and test JSONL annotation files.
2. **Load pretrained features:** Load the CLIP video feature matrix and the pooled CLIP query representation for each sample.
3. **Query conditioning:** Repeat the 512-dimensional query vector for every clip in its video.
4. **Feature fusion:** Concatenate each 512-dimensional video clip vector with the repeated 512-dimensional query vector, producing a 1,024-dimensional input per clip.
5. **Temporal modelling:** Pass the sequence through a two-layer bidirectional LSTM.
6. **Saliency prediction:** Apply dropout and a linear layer followed by sigmoid to produce a score between 0 and 1 for each clip.
7. **Training:** Use masked mean squared error so that loss is calculated only for annotated clips.
8. **Highlight selection:** Rank clips by predicted saliency and select the top 15% for each test video.

## Model Architecture

```text
CLIP Video Features (512-D)       CLIP Query Feature (512-D)
              |                              |
              |                       Repeat for each clip
              |                              |
              +---------- Concatenate -------+
                             |
                    Fused Features (1024-D)
                             |
                2-Layer Bidirectional LSTM
                    Hidden Size: 256
                             |
                       Dropout (0.3)
                             |
                    Fully Connected Layer
                             |
                          Sigmoid
                             |
                 Clip-wise Saliency Scores
                             |
                 Select Top 15% of Clips
                             |
                 Query-Focused Highlights
```

**Model configuration**

| Parameter | Value |
|---|---:|
| Input dimension | 1,024 |
| LSTM hidden dimension | 256 |
| LSTM layers | 2 |
| Direction | Bidirectional |
| Dropout | 0.3 |
| Trainable parameters | 4,203,009 |
| Output | One saliency score per clip |

## Training Configuration

| Setting | Value |
|---|---:|
| Epochs | 20 |
| Batch size | 32 |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Gradient clipping | 1.0 |
| Loss function | Masked Mean Squared Error (MSE) |
| Training device used in recorded run | NVIDIA Tesla T4 GPU |

The best checkpoint is selected using validation loss.

## Results

The following values are from the recorded notebook run. They are **validation results from the best validation-loss checkpoint**, evaluated using a 0.5 threshold to convert saliency scores into binary labels.

| Validation metric | Score |
|---|---:|
| Accuracy | 0.9098 |
| Precision | 0.9101 |
| Recall | 0.9996 |
| F1-score | 0.9527 |
| Mean Absolute Error (MAE) | 0.1345 |
| Best validation loss | 0.0297 |

The notebook also generates predictions for all **1,542 test samples**. Since the test split has no ground-truth highlight labels in the supplied annotations, no labelled test accuracy, precision, recall or F1-score is reported.

### Highlight selection

For the fixed-ratio inference step, the notebook selects the top 15% of clips ranked by predicted saliency for each test sample. The recorded average compression ratio is approximately **0.1599** (about 16%).

The notebook additionally contains an adaptive threshold experiment using a saliency threshold of 0.76. This is a separate analysis from the fixed top-15% selection strategy.

### Generated outputs

The notebook saves the following artifacts in the Kaggle working directory:

- `qvhighlight_bilstm_best.pth` — best model state dictionary.
- `loss_graph.png` — training and validation loss curves.
- `metrics_graph.png` — validation metric curves across epochs.
- `training_history.csv` — per-epoch training and validation history.
- `test_highlights.csv` — compact test highlight predictions.
- `test_highlights.json` — detailed test predictions and saliency scores.
- `single_video_demo.png` — example clip-level prediction visualization (generated in the demonstration cell).
- `single_video_highlights.csv` — example selected highlight timestamps.

These files are generated by running the notebook; they are not automatically included in this repository unless you upload them.

## Generating Highlights

For each test video, the model:

1. Computes a saliency score for every clip.
2. Selects `ceil(0.15 × number of clips)` clips with the highest scores (at least one clip).
3. Sorts the selected clip indices into video order.
4. Converts selected clip indices into approximate timestamps using the video duration and number of clips.

The output includes the video/query identifiers, selected clip indices, saliency scores and compression ratio.

## Repository Structure

A suggested GitHub repository layout is:

```text
QVHighlights-BiLSTM-Video-Summarization/
├── README.md
├── minor-project-final.ipynb
├── requirements.txt
├── .gitignore
├── results/
│   ├── loss_graph.png
│   ├── metrics_graph.png
│   ├── single_video_demo.png
│   ├── training_history.csv
│   └── test_highlights.csv
└── report/
    └── project_report.pdf
```

Only include result files and the report if you have them available and are permitted to share them. The full dataset is not included.

## Setup and Usage

### 1. Clone the repository

```bash
git clone https://github.com/<YOUR-USERNAME>/QVHighlights-BiLSTM-Video-Summarization.git
cd QVHighlights-BiLSTM-Video-Summarization
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Attach the dataset

The notebook was developed in Kaggle and expects the QVHighlights dataset to be attached as a Kaggle input. Its current base path is:

```python
BASE = "/kaggle/input/datasets/kanzallahhoussam/qvhighlight/QVHighlight"
```

If your dataset is stored elsewhere, update `BASE` in the notebook to point to the directory containing the `annotations` and `features` folders.

### 4. Run the notebook

Open `minor-project-final.ipynb` in Kaggle or Jupyter and execute the cells in order. A CUDA-enabled GPU is recommended for training; the notebook automatically selects CUDA when available and otherwise uses CPU.

**Note:** The notebook's output/checkpoint paths use `/kaggle/working/`. If running outside Kaggle, change those paths to a writable local directory.

## Limitations

- The implementation uses pre-extracted CLIP features; it does not extract visual or text features from raw videos itself.
- The current model uses CLIP video and query features only; the other feature types available in the dataset are not fused into this baseline.
- The supplied test annotations have no ground-truth highlight labels, so the test split is used for prediction generation rather than labelled metric evaluation.
- The top-15% selection rule is a simple fixed-ratio highlight selection strategy and does not guarantee semantic diversity between selected clips.
- The validation classification metrics depend on thresholding continuous saliency scores at 0.5 and should be interpreted in that context.

## Acknowledgements

- The QVHighlights dataset and its authors for the benchmark and annotations.
- The CLIP feature representations provided with the dataset.
- The PyTorch and scikit-learn communities for the tools used in this implementation.

## References

Please add the full bibliographic citation for the QVHighlights paper and any other sources required by your project report before publishing the repository.

---

**Disclaimer:** This repository presents an academic baseline implementation for research and educational purposes. It is not a production video summarization system.
