"""
Reconstruct label_map.json from the Aslense CSV.

Replicates train.py:discover_labels — alphabetic sort of words present in the
training set. The teammate's training set is unknown, so we filter the CSV by
video count and truncate to match the checkpoint's class count (1740).

When the teammate's real label_map.json arrives, just drop it in models/ and
delete the one this script writes.

Usage:
    python scripts/build_label_map.py [--target-classes 1740]
"""

import argparse
import json
from pathlib import Path

import pandas as pd

CSV_PATH = Path("data") / "Aslense Dataset.csv"
OUT_PATH = Path("models") / "label_map.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-classes", type=int, default=1740)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    df = pd.read_csv(CSV_PATH)
    counts = df.groupby("word").size().sort_values(ascending=False)

    # Take top-N by video count (matches the most likely teammate filter),
    # then sort alphabetically (matches train.py:discover_labels).
    top = counts.head(args.target_classes).index.tolist()
    words = sorted(top)

    label_map = {word: idx for idx, word in enumerate(words)}

    with open(args.out, "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"Wrote {len(label_map)} labels to {args.out}")
    print(
        f"Sample: index 0 = {words[0]!r}, index 308 = {words[308]!r}, "
        f"index {len(words) - 1} = {words[-1]!r}"
    )


if __name__ == "__main__":
    main()
