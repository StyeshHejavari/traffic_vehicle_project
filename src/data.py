import numpy as np

from sklearn.model_selection import train_test_split

from torchvision import datasets

from torch.utils.data import Subset, DataLoader

from PIL import Image


# =========================
# Paths
# =========================

TRAIN_DIR = "../dataset/merged_train"
TEST_DIR = "../dataset/test"

SEED = 42


# =========================
# Letterbox
# =========================

def letterbox(
    image,
    target_size=(224, 224),
    fill=(0, 0, 0)
):
    target_width, target_height = target_size

    original_width, original_height = image.size

    scale = min(
        target_width / original_width,
        target_height / original_height
    )

    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.BILINEAR
    )

    new_image = Image.new(
        "RGB",
        (target_width, target_height),
        fill
    )

    left = (target_width - new_width) // 2
    top = (target_height - new_height) // 2

    new_image.paste(
        image,
        (left, top)
    )

    return new_image


# =========================
# Dataset
# =========================
# Test/Evaluation dataset

def create_train_val_datasets(
    data_dir,
    train_transform,
    val_transform,
    test_size=0.2,
    seed=42
):

    # Full training dataset
    train_dataset_full = datasets.ImageFolder(
        data_dir,
        transform=train_transform
    )
    class_to_idx = train_dataset_full.class_to_idx
    
    test_dataset = datasets.ImageFolder(
        "../dataset/test",
        transform=val_transform
    )
    # Make sure train and test use the same class mapping
    assert class_to_idx == test_dataset.class_to_idx

    # Same images and same indices,
    # but with validation transform
    val_dataset_full = datasets.ImageFolder(
        data_dir,
        transform=val_transform
    )

    indices = np.arange(
        len(train_dataset_full)
    )

    labels = np.array(
        train_dataset_full.targets
    )

    train_indices, val_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=labels
    )

    train_dataset = Subset(
        train_dataset_full,
        train_indices
    )

    val_dataset = Subset(
        val_dataset_full,
        val_indices
    )

    return (
        train_dataset,
        val_dataset,
        train_indices,
        val_indices,
        class_to_idx
    )


# =========================
# DataLoaders
# =========================

def create_dataloaders(
    train_dataset,
    val_dataset,
    batch_size=32,
    shuffle_train=True
):

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader