"""
extract_landmarks.py — CLI helper: bangun dataset landmark → X.npy / y.npy.

Jalankan: python scripts/extract_landmarks.py
Bungkus tipis di atas src.preprocessing.build_sequences.build_dataset.
"""

import sys
from pathlib import Path

# pastikan root proyek di sys.path saat dijalankan langsung
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.build_sequences import build_dataset


def main():
    build_dataset()


if __name__ == "__main__":
    main()
