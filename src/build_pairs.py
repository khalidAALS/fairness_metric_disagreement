from __future__ import annotations

from pathlib import Path
import random
import pandas as pd

from config import RAW_DIR, METADATA_DIR


LFW_DIR = RAW_DIR / "lfw"
OUTPUT_CSV = METADATA_DIR / "lfw_pairs.csv"


def get_all_people() -> list[Path]:
    return [p for p in LFW_DIR.iterdir() if p.is_dir()]


def get_images(person_path: Path) -> list[Path]:
    return sorted(person_path.glob("*.jpg"))


def build_pairs() -> None:
    rows = []
    people = get_all_people()

    if not people:
        raise FileNotFoundError(f"No person folders found in {LFW_DIR}")


    # genuine pairs
    for person in people:
        images = get_images(person)
        if len(images) < 2:
            continue

        for i in range(len(images) - 1):
            rows.append(
                {
                    "group": person.name[0],
                    "image1": str(images[i]),
                    "image2": str(images[i + 1]),
                    "label": 1,
                }
            )

    genuine_count = len(rows)


    # impostor pairs
    valid_people = [p for p in people if len(get_images(p)) > 0]

    for _ in range(genuine_count):
        p1, p2 = random.sample(valid_people, 2)

        img1 = random.choice(get_images(p1))
        img2 = random.choice(get_images(p2))

        rows.append(
            {
                "group": "mixed",
                "image1": str(img1),
                "image2": str(img2),
                "label": 0,
            }
        )

    df = pd.DataFrame(rows)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved {len(df)} pairs to {OUTPUT_CSV}")
    print(f"Genuine pairs: {genuine_count}")
    print(f"Impostor pairs: {len(df) - genuine_count}")


if __name__ == "__main__":
    build_pairs()