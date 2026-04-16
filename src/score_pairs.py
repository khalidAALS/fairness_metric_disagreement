from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import METADATA_DIR, PROCESSED_DIR
from utils import cosine_similarity


PAIRS_CSV = METADATA_DIR / "lfw_pairs.csv"
EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"
OUTPUT_CSV = PROCESSED_DIR / "lfw_scores.csv"
MISSING_CSV = PROCESSED_DIR / "lfw_missing_scores.csv"


def embedding_output_path(image_path: Path) -> Path:
    safe_name = "__".join(image_path.parts[-3:])
    safe_name = safe_name.replace(".jpg", ".npy").replace(".png", ".npy")
    return EMBEDDINGS_DIR / safe_name


def main() -> None:
    df = pd.read_csv(PAIRS_CSV)

    scores = []
    missing_rows = []

    for _, row in df.iterrows():
        img1 = Path(row["image1"])
        img2 = Path(row["image2"])

        emb1_path = embedding_output_path(img1)
        emb2_path = embedding_output_path(img2)

        if not emb1_path.exists() or not emb2_path.exists():
            scores.append(np.nan)
            missing_rows.append(
                {
                    "group": row["group"],
                    "image1": str(img1),
                    "image2": str(img2),
                    "label": row["label"],
                    "emb1_exists": emb1_path.exists(),
                    "emb2_exists": emb2_path.exists(),
                    "emb1_path": str(emb1_path),
                    "emb2_path": str(emb2_path),
                }
            )
            continue

        emb1 = np.load(emb1_path)
        emb2 = np.load(emb2_path)

        score = cosine_similarity(emb1, emb2)
        scores.append(score)

    df["score"] = scores
    df.to_csv(OUTPUT_CSV, index=False)

    missing_df = pd.DataFrame(missing_rows)
    missing_df.to_csv(MISSING_CSV, index=False)

    print(f"Saved scores to: {OUTPUT_CSV}")
    print(f"Total rows: {len(df)}")
    print(f"Missing scores: {len(missing_rows)}")
    if len(missing_rows) > 0:
        print(f"Missing details saved to: {MISSING_CSV}")


if __name__ == "__main__":
    main()