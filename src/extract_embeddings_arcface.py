from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import insightface
from insightface.app import FaceAnalysis

from config import METADATA_DIR, PROCESSED_DIR


PAIRS_CSV = METADATA_DIR / "lfw_pairs.csv"
EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings_arcface"


def main() -> None:
    df = pd.read_csv(PAIRS_CSV)

    image_paths = set(df["image1"]).union(set(df["image2"]))
    image_paths = [Path(p) for p in image_paths]

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(320, 320))

    processed = 0
    skipped = 0
    failed = 0

    print(f"Total unique images: {len(image_paths)}")

    for img_path in image_paths:
        safe_name = "__".join(img_path.parts[-3:])
        out_path = EMBEDDINGS_DIR / safe_name.replace(".jpg", ".npy")

        if out_path.exists():
            skipped += 1
            continue

        try:
            img = np.array(Image.open(img_path).convert("RGB"))
            faces = app.get(img)

            if len(faces) == 0:
                failed += 1
                continue

            emb = faces[0].embedding
            np.save(out_path, emb)

            processed += 1

            if processed % 25 == 0:
                print(f"Processed {processed}")

        except Exception:
            failed += 1

    print("\nDone.")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()