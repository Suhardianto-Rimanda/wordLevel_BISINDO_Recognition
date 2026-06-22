"""
run_inference.py — CLI: jalankan inference webcam real-time (debug, tanpa web).

Jalankan: python scripts/run_inference.py
Tekan 'q' di jendela untuk keluar. Butuh model terlatih di models/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.inference_engine import run_webcam


def main():
    run_webcam(source=0)


if __name__ == "__main__":
    main()
