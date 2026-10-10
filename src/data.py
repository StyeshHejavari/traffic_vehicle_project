import numpy as np

from sklearn.model_selection import train_test_split

from torchvision import datasets

from torch.utils.data import Subset, DataLoader,Sampler , Dataset

from PIL import Image

import random
import torch
from PIL import Image
import torch.nn.functional as F
from pathlib import Path


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

# ****************************

# BalancedBatchSampler

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, num_classes, seed=42, num_batches=None):
        if batch_size % num_classes != 0:
            raise ValueError("batch_size must be divisible by num_classes.")
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.seed = seed
        self.samples_per_class = batch_size // num_classes
        self.class_to_indices = {c: [] for c in range(num_classes)}
        for idx, label in enumerate(labels):
            self.class_to_indices[int(label)].append(idx)
        self.num_batches = num_batches or len(labels) // batch_size
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        pools = {c: [] for c in range(self.num_classes)}

        def draw(c, k):
            out = []
            while len(out) < k:
                if not pools[c]:
                    pools[c] = self.class_to_indices[c].copy()
                    rng.shuffle(pools[c])
                out.append(pools[c].pop())
            return out

        for _ in range(self.num_batches):
            batch = []
            for c in range(self.num_classes):
                batch.extend(draw(c, self.samples_per_class))
            rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches



def create_simulated_imbalance(
    train_indices,
    train_labels,
    class_to_idx,
    selected_classes,
    retain_fraction,
    seed
):
    rng = np.random.default_rng(seed)

    selected_class_ids = {
        class_to_idx[class_name]
        for class_name in selected_classes
    }

    retained_indices = []

    for class_id in range(len(class_to_idx)):

        class_positions = np.where(
            train_labels == class_id
        )[0]

        if class_id in selected_class_ids:

            n_keep = max(
                1,
                int(len(class_positions) * retain_fraction)
            )

            selected_positions = rng.choice(
                class_positions,
                size=n_keep,
                replace=False
            )

            retained_indices.extend(
                train_indices[selected_positions]
            )

        else:

            retained_indices.extend(
                train_indices[class_positions]
            )

    return np.array(retained_indices)



# Classify unseen Folder

class UnlabeledImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.paths = list(Path(root).glob("*"))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]

        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, path.name


def classify_unseen_folder(folder_path, model, threshold, class_names, transform, device, batch_size=32):
    dataset = UnlabeledImageFolder(folder_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    filenames, predictions, confidences = [], [], []

    with torch.no_grad():
        idx = 0
        for images, names in loader:
            images = images.to(device)
            probs = F.softmax(model(images), dim=1)
            conf, pred_idx = probs.max(dim=1)

            for name, c, p in zip(names, conf.cpu().numpy(), pred_idx.cpu().numpy()):
                label = "Unknown" if c < threshold else class_names[p]
                filenames.append(name)
                predictions.append(label)
                confidences.append(c)

    return pd.DataFrame({
        "filename": filenames,
        "predicted_class": predictions,
        "confidence": confidences,
    })


# =================
# Predict Images
# =================
def predict_images(
    model,
    image_dir,
    transform,
    class_names,
    threshold,
    device
):
    image_dir = Path(image_dir)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    image_paths = [
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in image_extensions
    ]

    model.eval()

    results = []

    with torch.no_grad():
        for image_path in image_paths:

            # Load image
            image = Image.open(image_path).convert("RGB")

            # Preprocess
            image_tensor = transform(image).unsqueeze(0).to(device)

            # Model prediction
            logits = model(image_tensor)
            probabilities = F.softmax(logits, dim=1)

            confidence, prediction = probabilities.max(dim=1)

            confidence = confidence.item()
            prediction = prediction.item()

            predicted_class = class_names[prediction]

            # Unknown / known decision
            if confidence < threshold:
                final_prediction = "Unknown"
                needs_review = True
            else:
                final_prediction = predicted_class
                needs_review = False

            results.append({
                "filename": image_path.name,
                "predicted_class": final_prediction,
                "model_class": predicted_class,
                "confidence": confidence,
                "needs_review": needs_review,
                "probabilities": {
                    class_name: probability.item()
                    for class_name, probability
                    in zip(class_names, probabilities[0])
                }
            })

    return results