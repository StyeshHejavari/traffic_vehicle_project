# Confusion Matrix


import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def get_predictions(model, loader, device):
    """
    Get true labels and predictions for all samples in a DataLoader.
    """

    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            logits = model(images)

            predictions = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    return np.array(all_labels), np.array(all_predictions)


def plot_confusion_matrix(
    model,
    loader,
    class_names,
    device,
    normalize=None
):
    """
    Compute and plot the confusion matrix.

    normalize:
        None       -> raw counts
        "true"     -> row-normalized
        "pred"     -> column-normalized
        "all"      -> normalized over all samples
    """

    true_labels, predictions = get_predictions(
        model,
        loader,
        device
    )

    cm = confusion_matrix(
        true_labels,
        predictions,
        labels=np.arange(len(class_names)),
        normalize=normalize
    )

    fig, ax = plt.subplots(figsize=(9, 8))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(
        ax=ax,
        xticks_rotation=45,
        values_format=".2f" if normalize else "d"
    )

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")

    plt.tight_layout()
    plt.show()

    return cm

# ***********
# Classification Report


from sklearn.metrics import classification_report


def get_classification_report(
    model,
    loader,
    class_names,
    device
):
    true_labels, predictions = get_predictions(
        model,
        loader,
        device
    )

    report = classification_report(
        true_labels,
        predictions,
        target_names=class_names,
        digits=4
    )

    print(report)

    return report


# Plotting Curves

def plot_training_history(history):

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Train Accuracy"
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.show()


def set_seed(seed=42):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)