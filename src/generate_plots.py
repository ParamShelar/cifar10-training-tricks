import csv
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from data import get_dataloaders
from model import SmallCNN
from train import build_criterion, build_optimizer, evaluate, load_config, train_one_epoch
from utils import set_seed


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
HISTORY_DIR = RESULTS_DIR / "history"
PLOTS_DIR = RESULTS_DIR / "plots"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"
MODELS_DIR = RESULTS_DIR / "models"

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

EXPERIMENT_ORDER = [
    "baseline",
    "augmentation",
    "batchnorm",
    "dropout",
    "label_smoothing",
    "all_tricks",
]

DISPLAY_NAMES = {
    "baseline": "Baseline",
    "augmentation": "Augmentation",
    "batchnorm": "BatchNorm",
    "dropout": "Dropout",
    "label_smoothing": "Label smoothing",
    "all_tricks": "All tricks",
}

HISTORY_LOGS = {
    "baseline": """epoch,train_loss,train_acc,test_loss,test_acc
1,1.3584,0.5102,1.0820,0.6154
2,0.9347,0.6717,0.8859,0.6893
3,0.7551,0.7368,0.7819,0.7252
4,0.6251,0.7808,0.7725,0.7284
5,0.5152,0.8200,0.7781,0.7370
6,0.4126,0.8554,0.8057,0.7412
7,0.3126,0.8905,0.8140,0.7496
8,0.2261,0.9226,0.9282,0.7463
9,0.1558,0.9450,1.1126,0.7379
10,0.1207,0.9579,1.1888,0.7437
11,0.0977,0.9664,1.3603,0.7392
12,0.0757,0.9743,1.3968,0.7357
13,0.0740,0.9744,1.7438,0.7217
14,0.0736,0.9740,1.6210,0.7305
15,0.0721,0.9750,1.5528,0.7381
16,0.0577,0.9804,1.6343,0.7358
17,0.0554,0.9814,1.7435,0.7343
18,0.0588,0.9797,1.7633,0.7278
19,0.0504,0.9829,1.8221,0.7347
20,0.0602,0.9787,2.0174,0.7298
21,0.0463,0.9842,1.9091,0.7340
22,0.0416,0.9861,1.9827,0.7375
23,0.0466,0.9843,2.1151,0.7390
24,0.0501,0.9830,2.0965,0.7337
25,0.0475,0.9848,2.1496,0.7254
26,0.0458,0.9850,2.1424,0.7327
27,0.0364,0.9878,2.2353,0.7296
28,0.0386,0.9874,2.2247,0.7318
29,0.0368,0.9876,2.2486,0.7345
30,0.0423,0.9861,2.2857,0.7277
""",
    "augmentation": """epoch,train_loss,train_acc,test_loss,test_acc
1,1.5758,0.4237,1.2568,0.5427
2,1.2037,0.5688,1.0225,0.6355
3,1.0413,0.6307,0.9001,0.6825
4,0.9351,0.6687,0.8931,0.6893
5,0.8627,0.6966,0.7793,0.7255
6,0.8118,0.7124,0.7701,0.7350
7,0.7640,0.7333,0.7440,0.7415
8,0.7328,0.7440,0.6821,0.7665
9,0.7016,0.7555,0.6740,0.7679
10,0.6718,0.7640,0.6741,0.7625
11,0.6539,0.7706,0.6379,0.7800
12,0.6289,0.7793,0.6428,0.7796
13,0.6090,0.7875,0.6872,0.7636
14,0.5933,0.7927,0.6235,0.7903
15,0.5775,0.7978,0.5895,0.7990
16,0.5704,0.8019,0.6129,0.7895
17,0.5611,0.8039,0.5885,0.7988
18,0.5426,0.8092,0.5729,0.8057
19,0.5395,0.8108,0.5983,0.7992
20,0.5189,0.8182,0.5927,0.8016
21,0.5093,0.8221,0.5865,0.7985
22,0.5015,0.8251,0.5788,0.8016
23,0.4910,0.8288,0.5970,0.8058
24,0.4896,0.8293,0.5776,0.8044
25,0.4836,0.8304,0.5525,0.8169
26,0.4706,0.8352,0.5705,0.8149
27,0.4701,0.8351,0.5746,0.8088
28,0.4605,0.8373,0.5591,0.8172
29,0.4536,0.8418,0.5686,0.8138
30,0.4448,0.8448,0.5745,0.8114
""",
    "batchnorm": """epoch,train_loss,train_acc,test_loss,test_acc
1,1.1650,0.5845,0.9012,0.6823
2,0.7955,0.7203,0.8141,0.7191
3,0.6645,0.7674,0.7505,0.7441
4,0.5709,0.7992,0.7628,0.7381
5,0.4843,0.8307,0.7328,0.7542
6,0.4145,0.8546,0.8008,0.7478
7,0.3438,0.8795,0.7621,0.7570
8,0.2853,0.9014,0.7667,0.7665
9,0.2265,0.9220,0.7983,0.7614
10,0.1829,0.9357,0.8346,0.7681
11,0.1403,0.9534,0.9262,0.7640
12,0.1210,0.9583,0.9982,0.7591
13,0.0979,0.9666,1.0432,0.7619
14,0.0873,0.9705,1.0358,0.7651
15,0.0740,0.9747,1.0755,0.7701
16,0.0685,0.9775,1.1973,0.7580
17,0.0591,0.9793,1.3268,0.7433
18,0.0660,0.9767,1.1847,0.7743
19,0.0604,0.9802,1.3448,0.7487
20,0.0528,0.9821,1.2705,0.7613
21,0.0489,0.9837,1.3510,0.7576
22,0.0522,0.9823,1.3181,0.7606
23,0.0470,0.9837,1.2706,0.7791
24,0.0371,0.9870,1.4273,0.7631
25,0.0415,0.9852,1.3696,0.7676
26,0.0388,0.9871,1.3774,0.7747
27,0.0454,0.9846,1.3729,0.7662
28,0.0304,0.9900,1.4859,0.7663
29,0.0366,0.9876,1.5322,0.7568
30,0.0353,0.9879,1.4316,0.7740
""",
    "dropout": """epoch,train_loss,train_acc,test_loss,test_acc
1,1.5597,0.4332,1.1980,0.5651
2,1.1845,0.5818,0.9827,0.6524
3,1.0254,0.6414,0.8852,0.6912
4,0.9100,0.6839,0.8516,0.7017
5,0.8263,0.7135,0.8216,0.7169
6,0.7460,0.7389,0.7758,0.7304
7,0.6868,0.7589,0.7231,0.7500
8,0.6222,0.7803,0.7344,0.7469
9,0.5696,0.7979,0.7677,0.7537
10,0.5235,0.8150,0.7821,0.7440
11,0.4798,0.8267,0.7485,0.7637
12,0.4425,0.8391,0.8152,0.7518
13,0.4171,0.8497,0.8023,0.7558
14,0.3859,0.8575,0.8388,0.7599
15,0.3563,0.8699,0.8697,0.7580
16,0.3296,0.8790,0.8877,0.7578
17,0.3152,0.8816,0.9213,0.7551
18,0.2937,0.8910,0.9631,0.7588
19,0.2804,0.8947,0.9871,0.7541
20,0.2600,0.9012,1.0740,0.7632
21,0.2598,0.9012,1.0799,0.7579
22,0.2411,0.9085,1.0803,0.7633
23,0.2390,0.9107,1.1558,0.7558
24,0.2253,0.9156,1.1247,0.7590
25,0.2260,0.9154,1.1373,0.7596
26,0.2217,0.9157,1.1341,0.7594
27,0.2080,0.9207,1.2744,0.7586
28,0.2029,0.9245,1.1919,0.7587
29,0.1931,0.9267,1.2726,0.7585
30,0.1951,0.9273,1.3210,0.7633
""",
    "label_smoothing": """epoch,train_loss,train_acc,test_loss,test_acc
1,1.5605,0.5177,1.3346,0.6295
2,1.2309,0.6823,1.1824,0.7083
3,1.0950,0.7509,1.1230,0.7315
4,1.0006,0.7968,1.1137,0.7411
5,0.9209,0.8387,1.0993,0.7534
6,0.8465,0.8790,1.1058,0.7570
7,0.7723,0.9178,1.1082,0.7569
8,0.7124,0.9529,1.1240,0.7553
9,0.6605,0.9784,1.1341,0.7585
10,0.6245,0.9920,1.1521,0.7586
11,0.6000,0.9973,1.1384,0.7616
12,0.5836,0.9986,1.1307,0.7618
13,0.5706,0.9997,1.1587,0.7520
14,0.5628,0.9998,1.1490,0.7549
15,0.5573,0.9998,1.1511,0.7559
16,0.5543,0.9998,1.1574,0.7558
17,0.5515,0.9999,1.1630,0.7533
18,0.5510,0.9999,1.1601,0.7470
19,0.5509,0.9999,1.1656,0.7520
20,0.5469,0.9999,1.1658,0.7487
21,0.5442,1.0000,1.1604,0.7501
22,0.5403,1.0000,1.1662,0.7486
23,0.5379,1.0000,1.1577,0.7473
24,0.5377,1.0000,1.1634,0.7446
25,0.5399,1.0000,1.1769,0.7448
26,0.5376,1.0000,1.1537,0.7499
27,0.5338,1.0000,1.1517,0.7492
28,0.5310,1.0000,1.1507,0.7476
29,0.5302,1.0000,1.1640,0.7432
30,0.5301,1.0000,1.1664,0.7439
""",
    "all_tricks": """epoch,train_loss,train_acc,test_loss,test_acc
1,1.8457,0.3688,1.5322,0.5258
2,1.6438,0.4776,1.4148,0.5953
3,1.5700,0.5161,1.3152,0.6474
4,1.5127,0.5498,1.3108,0.6591
5,1.4718,0.5752,1.2884,0.6577
6,1.4433,0.5894,1.2735,0.6708
7,1.4149,0.6039,1.1653,0.7336
8,1.3912,0.6182,1.1674,0.7203
9,1.3762,0.6232,1.1453,0.7348
10,1.3575,0.6346,1.1621,0.7181
11,1.3436,0.6452,1.1119,0.7503
12,1.3292,0.6504,1.0900,0.7705
13,1.3185,0.6569,1.1131,0.7552
14,1.3033,0.6620,1.1051,0.7606
15,1.2960,0.6652,1.0749,0.7714
16,1.2844,0.6773,1.0521,0.7860
17,1.2706,0.6834,1.0311,0.7902
18,1.2648,0.6864,1.0921,0.7653
19,1.2516,0.6968,1.0391,0.7933
20,1.2447,0.6995,1.0187,0.7987
21,1.2313,0.7067,1.0150,0.7995
22,1.2193,0.7120,1.0134,0.7991
23,1.2056,0.7226,1.0110,0.8021
24,1.1982,0.7238,0.9948,0.8074
25,1.1886,0.7324,0.9881,0.8086
26,1.1834,0.7372,0.9997,0.8070
27,1.1758,0.7380,0.9853,0.8155
28,1.1652,0.7433,1.0068,0.8037
29,1.1567,0.7486,0.9835,0.8114
30,1.1558,0.7492,0.9693,0.8196
""",
}


def ensure_dirs():
    for directory in [HISTORY_DIR, PLOTS_DIR, CONFUSION_DIR, MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def history_dataframe(experiment):
    return pd.read_csv(io.StringIO(HISTORY_LOGS[experiment]))


def write_history_csvs():
    for experiment in EXPERIMENT_ORDER:
        df = history_dataframe(experiment)
        df.to_csv(HISTORY_DIR / f"{experiment}_seed0.csv", index=False)


def load_results_matrix():
    matrix_path = RESULTS_DIR / "matrix.csv"
    if matrix_path.exists():
        matrix = pd.read_csv(matrix_path)
        return matrix.set_index("experiment").loc[EXPERIMENT_ORDER].reset_index()

    rows = []
    for experiment in EXPERIMENT_ORDER:
        final = history_dataframe(experiment).iloc[-1]
        rows.append(
            {
                "experiment": experiment,
                "seed": 0,
                "train_acc": final["train_acc"],
                "test_acc": final["test_acc"],
                "train_loss": final["train_loss"],
                "test_loss": final["test_loss"],
                "epochs": 30,
                "time_minutes": np.nan,
            }
        )
    return pd.DataFrame(rows)


def save_test_accuracy_plot(matrix):
    labels = [DISPLAY_NAMES[name] for name in matrix["experiment"]]
    accuracy = matrix["test_acc"].to_numpy() * 100
    colors = ["#5b8def", "#35a16b", "#8e6ccf", "#d17a22", "#c44e52", "#222222"]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.bar(labels, accuracy, color=colors, edgecolor="#222222", linewidth=0.7)

    ax.set_title("Final Test Accuracy by Experiment", pad=12, weight="bold")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 90)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=20)

    for bar, value in zip(bars, accuracy):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "test_accuracy_by_experiment.png", dpi=300)
    fig.savefig(PLOTS_DIR / "test_accuracy_by_experiment.pdf")
    plt.close(fig)


def save_loss_curves_plot():
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharex=True)
    axes = axes.ravel()

    for ax, experiment in zip(axes, EXPERIMENT_ORDER):
        df = history_dataframe(experiment)
        ax.plot(df["epoch"], df["train_loss"], color="#315f9f", linewidth=2, label="Train")
        ax.plot(df["epoch"], df["test_loss"], color="#d17a22", linewidth=2, label="Test")
        ax.set_title(DISPLAY_NAMES[experiment], fontsize=11, weight="bold")
        ax.set_xlim(1, 30)
        ax.grid(alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[3:]:
        ax.set_xlabel("Epoch")

    for ax in axes[::3]:
        ax.set_ylabel("Loss")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Training and Test Loss Curves", y=0.995, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(PLOTS_DIR / "loss_curves.png", dpi=300)
    fig.savefig(PLOTS_DIR / "loss_curves.pdf")
    plt.close(fig)


def train_or_load_all_tricks_model():
    checkpoint_path = MODELS_DIR / "all_tricks_seed0.pt"
    config = load_config(REPO_ROOT / "configs" / "all_tricks.yaml")
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SmallCNN(
        use_batchnorm=config.get("use_batchnorm", False),
        use_dropout=config.get("use_dropout", False),
        dropout_p=config.get("dropout_p", 0.5),
    ).to(device)

    if checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        return model, device, config, checkpoint_path

    train_loader, test_loader = get_dataloaders(
        data_dir=str(REPO_ROOT / config["data_dir"]),
        batch_size=config["batch_size"],
        use_augmentation=config.get("use_augmentation", False),
    )
    criterion = build_criterion(config)
    optimizer = build_optimizer(model, config)

    print("No all_tricks checkpoint found; training all_tricks for confusion matrix.")
    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(
            f"Epoch {epoch:02d}/{config['epochs']} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

    torch.save(model.state_dict(), checkpoint_path)
    return model, device, config, checkpoint_path


def collect_predictions(model, device, config):
    _, test_loader = get_dataloaders(
        data_dir=str(REPO_ROOT / config["data_dir"]),
        batch_size=config["batch_size"],
        use_augmentation=config.get("use_augmentation", False),
    )
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(predicted.tolist())
            y_true.extend(labels.numpy().tolist())

    return np.array(y_true), np.array(y_pred)


def save_confusion_matrix_plot():
    model, device, config, checkpoint_path = train_or_load_all_tricks_model()
    y_true, y_pred = collect_predictions(model, device, config)

    counts = confusion_matrix(y_true, y_pred, labels=np.arange(len(CIFAR10_CLASSES)))
    normalized = counts / counts.sum(axis=1, keepdims=True)
    accuracy = (y_true == y_pred).mean()

    csv_path = CONFUSION_DIR / "all_tricks_confusion_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true_class", *CIFAR10_CLASSES])
        for class_name, row in zip(CIFAR10_CLASSES, counts):
            writer.writerow([class_name, *row.tolist()])

    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    image = ax.imshow(normalized, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Share of true class", rotation=-90, va="bottom")

    ax.set_title(f"All Tricks Confusion Matrix (accuracy: {accuracy * 100:.2f}%)", weight="bold")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(np.arange(len(CIFAR10_CLASSES)))
    ax.set_yticks(np.arange(len(CIFAR10_CLASSES)))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CIFAR10_CLASSES)

    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            value = normalized[i, j]
            text_color = "white" if value > 0.55 else "#222222"
            ax.text(j, i, f"{value * 100:.0f}", ha="center", va="center", color=text_color, fontsize=8)

    ax.set_xticks(np.arange(-0.5, len(CIFAR10_CLASSES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CIFAR10_CLASSES), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()
    fig.savefig(CONFUSION_DIR / "all_tricks_confusion_matrix.png", dpi=300)
    fig.savefig(CONFUSION_DIR / "all_tricks_confusion_matrix.pdf")
    plt.close(fig)

    return checkpoint_path, csv_path


def main():
    ensure_dirs()
    write_history_csvs()
    matrix = load_results_matrix()
    save_test_accuracy_plot(matrix)
    save_loss_curves_plot()
    checkpoint_path, csv_path = save_confusion_matrix_plot()

    print("Saved plots:")
    print(f"- {PLOTS_DIR / 'test_accuracy_by_experiment.png'}")
    print(f"- {PLOTS_DIR / 'loss_curves.png'}")
    print(f"- {CONFUSION_DIR / 'all_tricks_confusion_matrix.png'}")
    print(f"Saved confusion counts: {csv_path}")
    print(f"Saved all_tricks checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
