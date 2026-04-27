import argparse
import os
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import SmallCNN
from data import get_dataloaders
from utils import set_seed, save_metrics, save_history


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy



def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    return config


def build_criterion(config):
    loss_name = config.get("loss", "CrossEntropyLoss")
    if loss_name != "CrossEntropyLoss":
        raise ValueError(f"Unsupported loss: {loss_name}")

    label_smoothing = (
        config.get("label_smoothing", 0.0)
        if config.get("use_label_smoothing", False)
        else 0.0
    )

    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def build_optimizer(model, config):
    optimizer_name = config.get("optimizer", "Adam")
    if optimizer_name != "Adam":
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    return optim.Adam(
        model.parameters(),
        lr=config["learning_rate"]
    )


def get_history_file(config):
    results_dir = os.path.dirname(config["results_file"]) or "."
    file_name = f"{config['experiment_name']}_seed{config['seed']}.csv"
    return os.path.join(results_dir, "history", file_name)


def run_training(config):
    if config["epochs"] < 1:
        raise ValueError("epochs must be at least 1")

    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning experiment: {config['experiment_name']}")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        use_augmentation=config.get("use_augmentation", False)
    )

    model = SmallCNN(
        use_batchnorm=config.get("use_batchnorm", False),
        use_dropout=config.get("use_dropout", False),
        dropout_p=config.get("dropout_p", 0.5)
    ).to(device)

    criterion = build_criterion(config)
    optimizer = build_optimizer(model, config)

    start_time = time.time()
    history = []

    final_train_loss = None
    final_train_acc = None
    final_test_loss = None
    final_test_acc = None

    for epoch in range(1, config["epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        test_loss, test_acc = evaluate(
            model,
            test_loader,
            criterion,
            device
        )

        final_train_loss = train_loss
        final_train_acc = train_acc
        final_test_loss = test_loss
        final_test_acc = test_acc

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc
        })

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.4f}")

    end_time = time.time()
    time_minutes = (end_time - start_time) / 60

    row = {
        "experiment": config["experiment_name"],
        "seed": config["seed"],
        "train_acc": final_train_acc,
        "test_acc": final_test_acc,
        "train_loss": final_train_loss,
        "test_loss": final_test_loss,
        "epochs": config["epochs"],
        "time_minutes": time_minutes
    }

    save_metrics(config["results_file"], row)
    save_history(get_history_file(config), history)

    print("\nFinal Results")
    print(f"Experiment: {config['experiment_name']}")
    print(f"Final train accuracy: {final_train_acc:.4f}")
    print(f"Final test accuracy: {final_test_acc:.4f}")
    print(f"Final train loss: {final_train_loss:.4f}")
    print(f"Final test loss: {final_test_loss:.4f}")
    print(f"Training time: {time_minutes:.2f} minutes")

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
