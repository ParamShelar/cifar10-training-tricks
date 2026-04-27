# CIFAR-10 Training Tricks Ablation Study

This project investigates which common deep learning training techniques improve the performance of a small convolutional neural network on CIFAR-10.

The main research question is:

> How much do data augmentation, batch normalization, dropout, and label smoothing improve a small CNN on CIFAR-10 under limited compute?

## Project Idea

Instead of only training one CNN, this project performs an ablation study. A baseline CNN is trained first, then common training techniques are added one at a time to measure their individual effects.

The goal is to compare each method fairly by keeping the model, optimizer, learning rate, batch size, number of epochs, and random seed fixed across experiments.

## Experiments

| Experiment | Description |
|---|---|
| Baseline | Small CNN with no extra training tricks |
| Augmentation | Adds random crop and horizontal flip |
| Batch Normalization | Adds batch normalization after convolution layers |
| Dropout | Adds dropout before the final classifier |
| Label Smoothing | Uses cross-entropy loss with label smoothing |
| All Tricks | Combines augmentation, batch normalization, dropout, and label smoothing |

## Dataset

This project uses the CIFAR-10 dataset.

CIFAR-10 contains 60,000 color images across 10 classes:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

The dataset includes 50,000 training images and 10,000 test images.

## Model

The baseline model is a small convolutional neural network with the following general structure:

```text
Conv → ReLU → MaxPool
Conv → ReLU → MaxPool
Conv → ReLU
Flatten
Linear → ReLU
Linear → 10 classes
```

The model is intentionally kept simple so that the effect of each training technique can be measured more clearly.

## Training Setup

The planned training setup is:

```text
Dataset: CIFAR-10
Epochs: 30
Batch size: 128
Optimizer: Adam
Learning rate: 0.001
Loss: Cross-entropy
```

## Evaluation Metrics

Each experiment will track:

```text
Train accuracy
Test accuracy
Train loss
Test loss
Training time
Train-test accuracy gap
```

The train-test accuracy gap is calculated as:

```text
Gap = Train accuracy - Test accuracy
```

A smaller gap usually means the model is overfitting less.

## Repository Structure

```text
cifar10-training-tricks/
├── README.md
├── requirements.txt
├── report/
│   ├── main.tex
│   └── references.bib
├── src/
│   ├── train.py
│   ├── run_experiments.py
│   ├── model.py
│   ├── data.py
│   ├── evaluate.py
│   └── utils.py
├── configs/
│   ├── baseline.yaml
│   ├── augmentation.yaml
│   ├── batchnorm.yaml
│   ├── dropout.yaml
│   ├── label_smoothing.yaml
│   └── all_tricks.yaml
├── results/
│   ├── metrics.csv
│   ├── history/
│   ├── plots/
│   └── confusion_matrices/
└── notebooks/
    └── analysis.ipynb
```

## Installation

Install the required Python packages:

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

Colab usage:

```bash
!git clone <repo-url>
%cd cifar10-training-tricks
!pip install -r requirements.txt
!python src/run_experiments.py
```

## Results

Experiment results will be saved in:

```text
results/metrics.csv
```

Per-epoch training curves will be saved in:

```text
results/history/{experiment}_seed{seed}.csv
```

The main results table will compare:

| Experiment | Train Acc | Test Acc | Gap | Test Loss |
|---|---:|---:|---:|---:|
| Baseline | 98.61% | 72.77% | 25.84% | 2.286 |
| Augmentation | 84.48% | 81.14% | 3.34% | 0.575 |
| Batch Normalization | 98.79% | 77.40% | 21.39% | 1.432 |
| Dropout | 92.73% | 76.33% | 16.40% | 1.321 |
| Label Smoothing | 100.00% | 74.39% | 25.61% | 1.166 |
| All Tricks | 74.92% | 81.96% | -7.04% | 0.969 |

## Planned Plots

The project will include:

1. Test accuracy by experiment
2. Training and test loss curves
3. Confusion matrix for the best model

Plots will be saved in:

```text
results/plots/
```

Confusion matrices will be saved in:

```text
results/confusion_matrices/
```

## Report

The final LaTeX report will be written in:

```text
report/main.tex
```

The report will include:

```text
Introduction
Background
Methods
Experiments
Results
Discussion
Limitations
Conclusion
```

## Project Status

Project setup in progress.
