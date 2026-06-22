"""
compare_extractors.py — GATE PARITY: bandingkan fitur Holistic vs Hands.

Tujuan: memutuskan apakah aman set ``config.FEATURE_EXTRACTOR = "hands"`` (lebih
cepat) TANPA merusak prediksi LSTM yang dilatih pada fitur Holistic.

Cara kerja: ambil sejumlah frame IDENTIK, jalankan KEDUA ekstraktor, lalu ukur
selisih vektor fitur (126-dim: [tangan_kiri 63 | tangan_kanan 63]). Hanya frame
yang Holistic mendeteksi tangan (vektor non-nol) yang dihitung agar tidak bias
oleh frame kosong (0 == 0).

Sumber frame (urut prioritas):
  --webcam N         : tangkap N frame dari webcam (uji kondisi nyata).
  --videos-dir DIR   : sampling frame dari video .mp4 (default config.RAW_DIR).

Jalankan:
  .venv/Scripts/python.exe scripts/compare_extractors.py
  .venv/Scripts/python.exe scripts/compare_extractors.py --webcam 120
  .venv/Scripts/python.exe scripts/compare_extractors.py --videos-dir data/raw --frames-per-video 8

CATATAN: skrip ini TIDAK mengubah config. Ia hanya melapor angka + verdikt.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.preprocessing.extract_landmarks import (  # noqa: E402
    HolisticLandmarkExtractor,
    HandsLandmarkExtractor,
)
from src.preprocessing.extract_frames import read_frames  # noqa: E402

# Ambang verdikt: max-abs-diff per koordinat dianggap "cocok" bila <= ini.
# Koordinat MediaPipe ter-normalisasi [0,1] untuk x,y → 1e-2 ≈ 1% lebar frame.
PARITY_MAX_ABS = 1e-2


def _frames_from_webcam(n):
    """Tangkap ``n`` frame BGR dari webcam (source 0)."""
    import cv2

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Webcam tak bisa dibuka (source=0).")

    frames = []
    print(f"Menangkap {n} frame dari webcam — gerakkan tangan…")
    try:
        while len(frames) < n:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def _frames_from_videos(videos_dir, max_videos, frames_per_video):
    """Sampling frame merata dari beberapa video .mp4 di ``videos_dir``."""
    videos_dir = Path(videos_dir)
    if not videos_dir.exists():
        raise FileNotFoundError(
            f"Folder video tak ada: {videos_dir}\n"
            "Beri --videos-dir yang valid, atau pakai --webcam N."
        )
    vids = sorted(videos_dir.rglob("*.mp4"))[:max_videos]
    if not vids:
        raise FileNotFoundError(f"Tak ada .mp4 di {videos_dir}.")

    frames = []
    for v in vids:
        all_frames = read_frames(v)
        if not all_frames:
            continue
        # ambil ``frames_per_video`` indeks merata
        idxs = np.linspace(0, len(all_frames) - 1, frames_per_video).astype(int)
        frames.extend(all_frames[i] for i in np.unique(idxs))
    if not frames:
        raise RuntimeError("Tak ada frame terbaca dari video.")
    print(f"Sampling {len(frames)} frame dari {len(vids)} video.")
    return frames


def main():
    ap = argparse.ArgumentParser(description="Uji parity fitur Holistic vs Hands.")
    ap.add_argument("--webcam", type=int, default=0,
                    help="tangkap N frame dari webcam (0 = pakai video).")
    ap.add_argument("--videos-dir", default=str(config.RAW_DIR),
                    help="folder video .mp4 (default config.RAW_DIR).")
    ap.add_argument("--max-videos", type=int, default=5)
    ap.add_argument("--frames-per-video", type=int, default=10)
    args = ap.parse_args()

    if args.webcam > 0:
        frames = _frames_from_webcam(args.webcam)
    else:
        frames = _frames_from_videos(
            args.videos_dir, args.max_videos, args.frames_per_video
        )

    # Ekstrak fitur kedua metode pada frame yang SAMA.
    holistic = HolisticLandmarkExtractor(static_image_mode=False)
    hands = HandsLandmarkExtractor(static_image_mode=False)
    diffs, cosines = [], []
    n_holistic_hand = 0
    try:
        for frame in frames:
            vh = holistic.extract(frame)
            va = hands.extract(frame)
            if not np.any(vh):      # Holistic tak deteksi tangan → lewati
                continue
            n_holistic_hand += 1
            diffs.append(np.abs(vh - va))
            denom = (np.linalg.norm(vh) * np.linalg.norm(va)) or 1.0
            cosines.append(float(np.dot(vh, va) / denom))
    finally:
        holistic.close()
        hands.close()

    print("\n===== HASIL PARITY (Holistic vs Hands) =====")
    print(f"frame total           : {len(frames)}")
    print(f"frame dgn tangan (Hol): {n_holistic_hand}")
    if not diffs:
        print("Tak ada frame bertangan untuk dibandingkan. "
              "Ulangi dgn --webcam dan gerakkan tangan di depan kamera.")
        return

    diffs = np.stack(diffs)            # (n_frame, 126)
    mean_abs = float(diffs.mean())
    max_abs = float(diffs.max())
    p95_abs = float(np.percentile(diffs, 95))
    mean_cos = float(np.mean(cosines))
    print(f"mean abs diff         : {mean_abs:.5f}")
    print(f"p95  abs diff         : {p95_abs:.5f}")
    print(f"max  abs diff         : {max_abs:.5f}")
    print(f"mean cosine sim       : {mean_cos:.5f}")
    print(f"ambang aman (max abs) : {PARITY_MAX_ABS}")

    print("\n----- VERDIKT -----")
    if max_abs <= PARITY_MAX_ABS:
        print("AMAN: fitur Hands ~ identik dgn Holistic.")
        print('-> boleh set config.FEATURE_EXTRACTOR = "hands" untuk FPS lebih tinggi.')
    else:
        print("BEDA: fitur Hands menyimpang dari Holistic (data training).")
        print("-> JANGAN ganti ke 'hands' begitu saja; prediksi LSTM bisa rusak.")
        print("   Opsi: retrain LSTM pada fitur Hands, atau tetap Holistic.")


if __name__ == "__main__":
    main()
