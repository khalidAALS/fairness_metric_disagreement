from __future__ import annotations

from pathlib import Path
from typing import Set

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

from config import DEVICE, IMAGE_SIZE, METADATA_DIR, PROCESSED_DIR


PAIRS_CSV = METADATA_DIR / "lfw_pairs.csv"
EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"


def load_model() -> InceptionResnetV1:
    return InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)


def get_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )


def embedding_output_path(image_path: Path) -> Path:
    # preserves uniqueness with relative-style flat filename
    safe_name = "__".join(image_path.parts[-3:])
    safe_name = safe_name.replace(".jpg", ".npy").replace(".png", ".npy")
    return EMBEDDINGS_DIR / safe_name


def extract_one(model, transform, image_path: Path, save_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        embedding = model(tensor).detach().cpu().numpy()[0]

    np.save(save_path, embedding)


def main() -> None:
    if not PAIRS_CSV.exists():
        raise FileNotFoundError(f"Pairs CSV not found: {PAIRS_CSV}")

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PAIRS_CSV)

    all_images: Set[str] = set(df["image1"].tolist()) | set(df["image2"].tolist())
    image_paths = [Path(p) for p in all_images]

    model = load_model()
    transform = get_transform()

    print(f"Total unique images to process: {len(image_paths)}")

    processed = 0
    skipped = 0
    failed = 0

    for image_path in image_paths:
        save_path = embedding_output_path(image_path)

        if save_path.exists():
            skipped += 1
            continue

        try:
            extract_one(model, transform, image_path, save_path)
            processed += 1

            if processed % 100 == 0:
                print(f"Processed {processed} images")

        except Exception as e:
            failed += 1
            print(f"Failed: {image_path} -> {e}")

    print("\nDone.")
    print(f"Processed: {processed}")
    print(f"Skipped existing: {skipped}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()