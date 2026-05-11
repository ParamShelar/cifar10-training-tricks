# CIFAR-10 Training Tricks Ablation Study

This project tests how common deep learning training techniques affect a small convolutional neural network on CIFAR-10 under limited compute.

The research question is:

> How much do data augmentation, batch normalization, dropout, and label smoothing improve a small CNN on CIFAR-10 under limited compute?

## Report

The final report is available here:

[report/CIFAR10_training_tricks_report.pdf](report/CIFAR10_training_tricks_report.pdf)

The bibliography entries used for the report are kept in `report/references.bib`.

Generated figure images used while preparing the report are intentionally not tracked in Git. They can be recreated locally with:

```bash
python src/generate_plots.py
```

## Experiments

| Experiment | Description |
|---|---|
| Baseline | Small CNN with no extra training tricks |
| Augmentation | Adds random crop and horizontal flip |
| Batch Normalization | Adds batch normalization after convolution layers |
| Dropout | Adds dropout before the final classifier |
| Label Smoothing | Uses cross-entropy loss with label smoothing |
| All Tricks | Combines augmentation, batch normalization, dropout, and label smoothing |

## Results

| Experiment | Train Acc | Test Acc | Gap | Test Loss |
|---|---:|---:|---:|---:|
| Baseline | 98.61% | 72.77% | 25.84% | 2.286 |
| Augmentation | 84.48% | 81.14% | 3.34% | 0.575 |
| Batch Normalization | 98.79% | 77.40% | 21.39% | 1.432 |
| Dropout | 92.73% | 76.33% | 16.40% | 1.321 |
| Label Smoothing | 100.00% | 74.39% | 25.61% | 1.166 |
| All Tricks | 74.92% | 81.96% | -7.04% | 0.969 |

The best test accuracy came from combining all tricks, while augmentation alone was the strongest individual improvement over the baseline.

## Dataset

This project uses the CIFAR-10 dataset, which contains 60,000 color images across 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## Model

The baseline model is a small convolutional neural network:

```text
Conv -> ReLU -> MaxPool
Conv -> ReLU -> MaxPool
Conv -> ReLU
Flatten
Linear -> ReLU
Linear -> 10 classes
```

The model is intentionally simple so the effect of each training technique can be measured clearly.

## Training Setup

```text
Dataset: CIFAR-10
Epochs: 30
Batch size: 128
Optimizer: Adam
Learning rate: 0.001
Loss: Cross-entropy
Seed: 0
```

## Repository Structure

```text
cifar10-training-tricks/
|-- README.md
|-- requirements.txt
|-- report/
|   |-- CIFAR10_training_tricks_report.pdf
|   `-- references.bib
|-- src/
|   |-- train.py
|   |-- run_experiments.py
|   |-- generate_plots.py
|   |-- model.py
|   |-- data.py
|   |-- evaluate.py
|   `-- utils.py
|-- configs/
|   |-- baseline.yaml
|   |-- augmentation.yaml
|   |-- batchnorm.yaml
|   |-- dropout.yaml
|   |-- label_smoothing.yaml
|   `-- all_tricks.yaml
|-- results/
|   |-- matrix.csv
|   |-- history/
|   `-- confusion_matrices/
`-- notebooks/
    `-- analysis.ipynb
```

## Installation

```bash
pip install -r requirements.txt
```

## How to Run

Run one experiment:

```bash
python src/train.py --config configs/baseline.yaml
```

Run all experiments:

```bash
python src/run_experiments.py
```

Quick debug run:

```bash
python src/run_experiments.py --epochs 1
```

Generate local plots and the confusion matrix CSV:

```bash
python src/generate_plots.py
```

## Outputs

Experiment summaries are saved in `results/matrix.csv`.

Per-epoch training histories are saved in `results/history/`.

Generated plots and report figure images are kept out of Git so the repository stays focused on code, data tables, and the final report PDF.
