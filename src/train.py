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
    threshold=None
):

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    # --------------------------------------------------------
    # Best validation result
    # --------------------------------------------------------

    best_val_accuracy = float("-inf")

    best_val_loss = None

    best_epoch = None

    # --------------------------------------------------------
    # Checkpoint path
    # --------------------------------------------------------

    checkpoint_path = (
        CHECKPOINT_DIR /
        f"{model_name}_best.pt"
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(epochs):

        # -------------------------
        # Train
        # -------------------------

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        # -------------------------
        # Validation
        # -------------------------

        val_loss, val_accuracy = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device
        )

        # -------------------------
        # Store history
        # -------------------------

        history["train_loss"].append(train_loss)

        history["train_accuracy"].append(train_accuracy)

        history["val_loss"].append(val_loss)

        history["val_accuracy"].append(val_accuracy)

        # -------------------------
        # Print results
        # -------------------------

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} --- "
            f"Train Acc: {train_accuracy:.4f} --- "
            f"Val Loss: {val_loss:.4f} --- "
            f"Val Acc: {val_accuracy:.4f}"
        )

        # -------------------------
        # Check best model
        # -------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            best_val_loss = val_loss

            best_epoch = epoch + 1

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=best_epoch,
                val_accuracy=best_val_accuracy,
                val_loss=best_val_loss,
                path=checkpoint_path,
                class_to_idx=class_to_idx,
                seed=seed,
                config=config,
                threshold=threshold
            )

    # ========================================================
    # Save experiment summary
    # ========================================================

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

    # ========================================================
    # Final messages
    # ========================================================

    print()
    print("Best checkpoint saved:")
    print(checkpoint_path)

    print()
    print("Experiment results updated:")
    print(RESULTS_FILE)

    return history, best_val_accuracy