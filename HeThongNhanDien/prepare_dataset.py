import argparse
import random
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset structure from raw dataset folder")
    parser.add_argument("--source", default="dataset_raw", help="Source folder containing raw .jpg and .txt files")
    parser.add_argument("--dest", default="dataset", help="Destination dataset root folder")
    parser.add_argument("--train", type=float, default=0.8, help="Fraction of images for training")
    parser.add_argument("--val", type=float, default=0.1, help="Fraction of images for validation")
    parser.add_argument("--test", type=float, default=0.1, help="Fraction of images for testing")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for splitting")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")
    return parser.parse_args()


def make_folder(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def prepare_split(source: Path, dest: Path, train_frac: float, val_frac: float, test_frac: float, seed: int, move: bool):
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"Source folder not found: {source}")

    image_files = sorted(source.glob("*.jpg")) + sorted(source.glob("*.jpeg")) + sorted(source.glob("*.png"))
    image_files = [p for p in image_files if p.is_file()]
    if not image_files:
        raise RuntimeError(f"No image files found in {source}")

    random.seed(seed)
    random.shuffle(image_files)

    total = len(image_files)
    train_count = int(total * train_frac)
    val_count = int(total * val_frac)
    test_count = total - train_count - val_count

    splits = {
        "train": image_files[:train_count],
        "val": image_files[train_count:train_count + val_count],
        "test": image_files[train_count + val_count:],
    }

    for split_name, images in splits.items():
        image_dest = dest / "images" / split_name
        label_dest = dest / "labels" / split_name
        make_folder(image_dest)
        make_folder(label_dest)

        for image_path in images:
            label_path = image_path.with_suffix(".txt")
            target_image = image_dest / image_path.name
            target_label = label_dest / label_path.name

            if move:
                shutil.move(str(image_path), str(target_image))
            else:
                shutil.copy2(str(image_path), str(target_image))

            if label_path.exists():
                if move:
                    shutil.move(str(label_path), str(target_label))
                else:
                    shutil.copy2(str(label_path), str(target_label))
            else:
                target_label.write_text("")

    print(f"Split {total} images into:")
    print(f"  train: {len(splits['train'])}")
    print(f"  val:   {len(splits['val'])}")
    print(f"  test:  {len(splits['test'])}")
    print(f"Prepared dataset at: {dest}")


def main():
    args = parse_args()
    if round(args.train + args.val + args.test, 6) != 1.0:
        raise ValueError("train + val + test fractions must sum to 1.0")

    source = Path(args.source)
    dest = Path(args.dest)
    prepare_split(source, dest, args.train, args.val, args.test, args.seed, args.move)


if __name__ == "__main__":
    main()
