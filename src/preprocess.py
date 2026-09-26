from pathlib import Path
from PIL import Image
from collections import defaultdict
import hashlib
import pandas as pd
import shutil


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

if (BASE_DIR / "dataset").exists():
    DATASET_DIR = BASE_DIR / "dataset"

elif (BASE_DIR.parent / "dataset").exists():
    DATASET_DIR = BASE_DIR.parent / "dataset"

else:
    raise FileNotFoundError(
        "Could not find the dataset folder."
    )


UNCLEAN_DIR = DATASET_DIR / "unclean"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"




for folder in (
    UNCLEAN_DIR,
    TRAIN_DIR,
    TEST_DIR
):
    if not folder.exists():
        raise FileNotFoundError(
            f"Folder not found: {folder}"
        )


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# File discovery
# ============================================================

def get_image_files(root):
    root = Path(root)

    return [
        p
        for p in root.rglob("*")
        if (
            p.is_file()
            and p.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]


# ============================================================
# MD5 / duplicate detection
# ============================================================

def calculate_md5(file_path):
    """Calculate MD5 hash of a file."""

    md5 = hashlib.md5()

    with open(file_path, "rb") as f:

        for chunk in iter(
            lambda: f.read(8192),
            b""
        ):
            md5.update(chunk)

    return md5.hexdigest()


def build_hash_index(folder):
    """
    Return:
        {md5: [file_paths]}
    """

    hash_index = defaultdict(list)

    folder = Path(folder)

    for file_path in folder.rglob("*"):

        if (
            file_path.is_file()
            and file_path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            md5 = calculate_md5(file_path)

            hash_index[md5].append(
                file_path
            )

    return hash_index


def find_duplicate_groups(hash_index):
    """Find duplicate groups inside one dataset."""

    return {
        md5: files
        for md5, files in hash_index.items()
        if len(files) > 1
    }


def find_cross_dataset_duplicates(
    source_hashes,
    target_hashes
):
    """Find exact duplicates between two datasets."""

    duplicates = []

    for md5, source_files in source_hashes.items():

        if md5 not in target_hashes:
            continue

        for source_file in source_files:

            for target_file in target_hashes[md5]:

                duplicates.append(
                    (source_file, target_file)
                )

    return duplicates


# ============================================================
# Image audit
# ============================================================

def audit_image_file(path):

    result = {
        "path": str(path),
        "extension": path.suffix.lower(),
        "readable": False,
        "format": None,
        "width": None,
        "height": None,
        "mode": None,
        "error": None
    }

    if (
        path.suffix.lower()
        not in IMAGE_EXTENSIONS
    ):

        result["error"] = "image_extension"

        return result

    try:

        with Image.open(path) as img:
            img.verify()

        with Image.open(path) as img:

            result["readable"] = True
            result["format"] = img.format
            result["width"], result["height"] = img.size
            result["mode"] = img.mode

    except Exception as e:

        result["error"] = (
            f"{type(e).__name__}: {e}"
        )

    return result


def add_hash_column(audit_df):

    audit_df = audit_df.copy()

    audit_df["hash"] = pd.NA

    for idx, row in audit_df.iterrows():

        if row["readable"]:

            try:

                audit_df.at[
                    idx,
                    "hash"
                ] = calculate_md5(
                    Path(row["path"])
                )

            except Exception as e:

                audit_df.at[
                    idx,
                    "error"
                ] = (
                    f"{type(e).__name__}: {e}"
                )

    return audit_df


def find_too_small_images(
    audit_df,
    min_width=32,
    min_height=32
):

    return audit_df[
        (audit_df["readable"]) &
        (
            (audit_df["width"] < min_width) |
            (audit_df["height"] < min_height)
        )
    ]


def add_aspect_ratio(audit_df):

    audit_df = audit_df.copy()

    audit_df["aspect_ratio"] = (
        audit_df["width"] /
        audit_df["height"]
    )

    return audit_df


# ============================================================
# Label conflict
# ============================================================

def build_hash_label_index(folder):

    hash_to_files = defaultdict(list)

    folder = Path(folder)

    for file_path in folder.rglob("*"):

        if (
            file_path.is_file()
            and file_path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            md5 = calculate_md5(file_path)

            label = file_path.parent.name

            hash_to_files[md5].append(
                (file_path, label)
            )

    return hash_to_files


def find_label_conflicts(hash_to_files):

    label_conflicts = {}

    for md5, files in hash_to_files.items():

        labels = {
            label
            for _, label in files
        }

        if len(labels) > 1:

            label_conflicts[md5] = files

    return label_conflicts


# ============================================================
# Cleaning report
# ============================================================

def build_cleaning_report(
    audit_df,
    train_hashes,
    test_hashes
):

    report = audit_df.copy()

    report["class"] = report["path"].apply(
        lambda x: Path(x).parent.name
    )

    report["duplicate_in_unclean"] = (
        report["hash"].notna() &
        report["hash"].duplicated(
            keep=False
        )
    )

    report["duplicate_in_train"] = (
        report["hash"].notna() &
        report["hash"].isin(train_hashes)
    )

    report["duplicate_in_test"] = (
        report["hash"].notna() &
        report["hash"].isin(test_hashes)
    )

    return report


# ============================================================
# Cleaning
# ============================================================

def clean_unclean_dataset(
    cleaning_report,
    train_hashes,
    test_hashes,
    output_dir
):

    output_dir = Path(output_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    seen_hashes = set()

    kept_files = []
    removed_files = []

    for _, row in cleaning_report.iterrows():

        path = Path(row["path"])
        file_hash = row["hash"]

        if not row["readable"]:

            removed_files.append(
                (path, "unreadable_or_invalid")
            )

            continue

        if pd.isna(file_hash):

            removed_files.append(
                (path, "missing_hash")
            )

            continue

        if file_hash in train_hashes:

            removed_files.append(
                (path, "duplicate_in_train")
            )

            continue

        if file_hash in test_hashes:

            removed_files.append(
                (path, "duplicate_in_test")
            )

            continue

        if file_hash in seen_hashes:

            removed_files.append(
                (path, "duplicate_in_unclean")
            )

            continue

        seen_hashes.add(file_hash)

        kept_files.append(path)

    for path in kept_files:

        class_name = path.parent.name

        destination_dir = (
            output_dir / class_name
        )

        destination_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(
            path,
            destination_dir / path.name
        )

    return kept_files, removed_files


# ============================================================
# Merge train + cleaned unclean
# ============================================================

def create_merged_train_dataset(
    train_dir,
    cleaned_unclean_dir,
    output_dir
):

    train_dir = Path(train_dir)
    cleaned_unclean_dir = Path(
        cleaned_unclean_dir
    )
    output_dir = Path(output_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Original train
    for class_dir in train_dir.iterdir():

        if not class_dir.is_dir():
            continue

        destination_dir = (
            output_dir / class_dir.name
        )

        destination_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for path in class_dir.iterdir():

            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            ):

                shutil.copy2(
                    path,
                    destination_dir / path.name
                )

    # Cleaned unclean
    for class_dir in cleaned_unclean_dir.iterdir():

        if not class_dir.is_dir():
            continue

        destination_dir = (
            output_dir / class_dir.name
        )

        destination_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for path in class_dir.iterdir():

            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            ):

                destination = (
                    destination_dir / path.name
                )

                if destination.exists():

                    stem = path.stem
                    suffix = path.suffix
                    counter = 1

                    while destination.exists():

                        destination = (
                            destination_dir /
                            f"{stem}_unclean_{counter}{suffix}"
                        )

                        counter += 1

                shutil.copy2(
                    path,
                    destination

                )

    return output_dir

# ============================================================
# Class statistics
# ============================================================

def get_class_counts(dataset_dir):

    dataset_dir = Path(dataset_dir)

    class_counts = {}

    for class_dir in sorted(
        dataset_dir.iterdir()
    ):

        if class_dir.is_dir():

            class_counts[class_dir.name] = sum(
                1
                for p in class_dir.iterdir()
                if (
                    p.is_file()
                    and p.suffix.lower()
                    in IMAGE_EXTENSIONS
                )
            )

    return class_counts