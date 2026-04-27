import random
import numpy as np
import torch
import csv
import os


METRIC_COLUMNS = [
    "experiment",
    "seed",
    "train_acc",
    "test_acc",
    "train_loss",
    "test_loss",
    "epochs",
    "time_minutes",
]

HISTORY_COLUMNS = [
    "epoch",
    "train_loss",
    "train_acc",
    "test_loss",
    "test_acc",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_parent_dir(file_path):
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def ensure_csv_header(file_path, fieldnames):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return True

    with open(file_path, mode="r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if rows and rows[0] == fieldnames:
        return False

    with open(file_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        writer.writerows(rows)

    return False


def save_metrics(results_file, row):
    ensure_parent_dir(results_file)
    write_header = ensure_csv_header(results_file, METRIC_COLUMNS)

    with open(results_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def save_history(history_file, rows):
    ensure_parent_dir(history_file)

    with open(history_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
