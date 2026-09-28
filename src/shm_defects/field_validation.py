"""Image-level external validation from manually verified confusion counts.

The 44 field images are not redistributable; this module reproduces the reported metrics
from the counts in `results/field_validation.csv` (dissertation Table 7).

    python -m shm_defects.field_validation results/field_validation.csv
"""
from __future__ import annotations

import argparse
import csv

from .metrics import Counts


def summarise(path: str) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            c = Counts(tp=int(r["tp"]), fp=int(r["fp"]), fn=int(r["fn"]), tn=int(r["tn"]))
            specificity = c.tn / (c.tn + c.fp) if (c.tn + c.fp) else 0.0
            rows.append({"model": r["model"], "images": c.tp + c.fp + c.fn + c.tn,
                         "accuracy": round(c.accuracy, 4), "precision": round(c.precision, 3),
                         "recall": round(c.recall, 3), "specificity": round(specificity, 3)})
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", nargs="?", default="results/field_validation.csv")
    args = ap.parse_args(argv)
    rows = summarise(args.csv)
    print(f"{'model':<14}{'n':>4}{'acc':>9}{'prec':>7}{'rec':>7}{'spec':>7}")
    for r in rows:
        print(f"{r['model']:<14}{r['images']:>4}{r['accuracy']:>9.2%}{r['precision']:>7.3f}{r['recall']:>7.3f}{r['specificity']:>7.3f}")


if __name__ == "__main__":
    main()
