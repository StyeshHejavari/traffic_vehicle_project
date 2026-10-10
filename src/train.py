from pathlib import Path
import json

import torch


# ============================================================
# Project paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
RESULTS_DIR = PROJECT_DIR / "results"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = RESULTS_DIR / "experiment_results.json"


# ============================================================
# Train one epoch
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(logits, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = logits.argmax(dim=1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total

    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)

            predictions = logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

    epoch_loss = running_loss / total

    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# Save checkpoint
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    val_accuracy,
    val_loss,
    path,
    class_to_idx,
    seed,
    config,
    threshold=None
):

    checkpoint = {
        "epoch": epoch,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "val_accuracy": val_accuracy,

        "val_loss": val_loss,

        "class_to_idx": class_to_idx,

        "seed": seed,

        "config": config,

        "threshold": threshold
    }

    torch.save(checkpoint, path)


# ============================================================
# Save experiment result to JSON
# ============================================================

def save_experiment_result(
    model_name,
    best_epoch,
    best_val_accuracy,
    best_val_loss,
    seed,
    class_to_idx,
    config,
    checkpoint_path,
    threshold=None
):

    if RESULTS_FILE.exists():

        with open(
            RESULTS_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            results = json.load(f)

    else:

        results = {}

    results[model_name] = {

        "best_epoch": best_epoch,

        "best_val_accuracy": best_val_accuracy,

        "best_val_loss": best_val_loss,

        "seed": seed,

        "class_to_idx": class_to_idx,

        "config": config,

        "threshold": threshold,

        "checkpoint": str(
            checkpoint_path.relative_to(PROJECT_DIR)
        )
    }

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# Main training function
# ============================================================
def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    model_name,
    class_to_idx,
    seed,
    config,
    threshold=None,
    scheduler=None,
    early_stopping=False,
    patience=3,
    min_delta=0.0
):
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": []
    }

    best_val_accuracy = float("-inf")
    best_val_loss = None
    best_epoch = None

    # Counter for Early Stopping
    epochs_without_improvement = 0

    checkpoint_path = CHECKPOINT_DIR / f"{model_name}_best.pt"

    for epoch in range(epochs):

        # -------------------------
        # Train
        # -------------------------
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        # -------------------------
        # Validation
        # -------------------------
        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        current_lr = optimizer.param_groups[0]["lr"]

        # -------------------------
        # Save history
        # -------------------------
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["learning_rate"].append(current_lr)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"--- Train Acc: {train_accuracy:.4f} "
            f"--- Val Loss: {val_loss:.4f} "
            f"--- Val Acc: {val_accuracy:.4f} "
            f"--- LR: {current_lr:.6f}"
        )

        # -------------------------
        # Check improvement
        # -------------------------
        improved = val_accuracy > best_val_accuracy + min_delta

        if improved:

            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            best_epoch = epoch + 1

            # Reset Early Stopping counter
            epochs_without_improvement = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                val_accuracy=val_accuracy,
                val_loss=val_loss,
                path=checkpoint_path,
                class_to_idx=class_to_idx,
                seed=seed,
                config=config,
                threshold=threshold
            )

            print("  → Validation accuracy improved. Checkpoint saved.")

        else:

            if early_stopping:
                epochs_without_improvement += 1

                print(
                    f"  → No improvement "
                    f"({epochs_without_improvement}/{patience})"
                )

        # -------------------------
        # Scheduler
        # -------------------------
        if scheduler is not None:
            scheduler.step(val_loss)

        # -------------------------
        # Early Stopping
        # -------------------------
        if (
            early_stopping
            and epochs_without_improvement >= patience
        ):
            print(
                f"\nEarly stopping triggered at epoch {epoch+1}."
            )
            print(
                f"Best validation accuracy: "
                f"{best_val_accuracy:.4f} "
                f"at epoch {best_epoch}"
            )
            break

    # -------------------------
    # Save experiment result
    # -------------------------
    save_experiment_result(
        model_name=model_name,
        best_epoch=best_epoch,
        best_val_accuracy=best_val_accuracy,
        best_val_loss=best_val_loss,
        seed=seed,
        class_to_idx=class_to_idx,
        config=config,
        checkpoint_path=checkpoint_path,
        threshold=threshold
    )

    return history, best_val_accuracy


# BCE Training Functions
import torch
import torch.nn.functional as F


def train_one_epoch_bce(
    model,
    loader,
    criterion,
    optimizer,
    device,
    num_classes
):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        targets = F.one_hot(
            labels,
            num_classes=num_classes
        ).float()

        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = logits.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def evaluate_bce(
    model,
    loader,
    criterion,
    device,
    num_classes
):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            targets = F.one_hot(
                labels,
                num_classes=num_classes
            ).float()

            loss = criterion(logits, targets)

            running_loss += loss.item() * images.size(0)

            predictions = logits.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def train_model_bce(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    model_name,
    class_to_idx,
    seed,
    config,
    num_classes,
    threshold=None,
    scheduler=None
):
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": []
    }

    best_val_accuracy = float("-inf")
    best_val_loss = None
    best_epoch = None

    checkpoint_path = CHECKPOINT_DIR / f"{model_name}_best.pt"

    for epoch in range(epochs):

        train_loss, train_accuracy = train_one_epoch_bce(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            num_classes
        )

        val_loss, val_accuracy = evaluate_bce(
            model,
            val_loader,
            criterion,
            device,
            num_classes
        )

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["learning_rate"].append(current_lr)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"--- Train Acc: {train_accuracy:.4f} "
            f"--- Val Loss: {val_loss:.4f} "
            f"--- Val Acc: {val_accuracy:.4f} "
            f"--- LR: {current_lr:.6f}"
        )

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            best_epoch = epoch + 1

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                val_accuracy=val_accuracy,
                val_loss=val_loss,
                path=checkpoint_path,
                class_to_idx=class_to_idx,
                seed=seed,
                config=config,
                threshold=threshold
            )

        # Scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

    save_experiment_result(
        model_name=model_name,
        best_epoch=best_epoch,
        best_val_accuracy=best_val_accuracy,
        best_val_loss=best_val_loss,
        seed=seed,
        class_to_idx=class_to_idx,
        config=config,
        checkpoint_path=checkpoint_path,
        threshold=threshold
    )

    return history, best_val_accuracy
