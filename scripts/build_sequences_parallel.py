"""
build_sequences_parallel.py — Versi PARALEL dari build_sequences.py.

Tujuan: percepat ekstraksi landmark dengan memproses banyak video SECARA
PARALEL antar-proses (ProcessPoolExecutor), memanfaatkan seluruh core CPU.
Ekstraksi MediaPipe Holistic bersifat CPU-bound, jadi paralelisasi level video
memberi speedup mendekati jumlah core.

PRINSIP UTAMA — HASIL HARUS IDENTIK DGN build_sequences.py:
    - Logika numerik (normalize_sequence, resample_sequence) di-IMPORT ulang dari
      build_sequences.py. TIDAK ada duplikasi → tidak ada risiko menyimpang.
    - Split (split_videos_stratified), verifikasi anti-leakage (verify_split),
      dan penulisan manifest (_write_manifest) juga di-IMPORT ulang.
    - records DIURUTKAN ulang (by video_id) sebelum split & sebelum dirakit ke
      array, sehingga isi X_*.npy / y_*.npy byte-identik dengan versi serial,
      terlepas dari urutan penyelesaian proses yang non-deterministik.

Tiap WORKER membuka SATU HolisticLandmarkExtractor sekali (mahal bila per-video)
lalu memproses banyak video. Objek MediaPipe tidak bisa di-pickle / di-share
antar-proses → tiap proses punya extractor sendiri (lazy-init per proses).

CATATAN PLATFORM:
    - Jalankan di LINUX/Ubuntu. start method "fork" (default Linux) paling mulus.
    - Di Windows/macOS (spawn) MediaPipe bisa bermasalah; tidak disarankan.

Jalankan:
    python scripts/build_sequences_parallel.py                  # auto: cpu_count-1
    python scripts/build_sequences_parallel.py --workers 6      # paksa 6 worker
    python scripts/build_sequences_parallel.py --raw-dir ... --out-dir ...

Output: SAMA dengan build_sequences.py (X/y per split, label_map.json,
manifest.csv) ke config.PROCESSED_DIR.
"""

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

import config
from src.preprocessing.extract_frames import read_frames
from src.preprocessing.extract_landmarks import (
    HolisticLandmarkExtractor,
    extract_sequence,
)
# Pakai-ULANG semua logika dari modul serial — JANGAN tulis ulang.
from src.preprocessing.build_sequences import (
    normalize_sequence,
    resample_sequence,
    split_videos_stratified,
    verify_split,
    _write_manifest,
    _list_videos,
    _SPLITS,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# WORKER — satu extractor per PROSES (bukan per video)
# --------------------------------------------------------------------------- #
# Extractor disimpan di level-modul tiap proses worker. ProcessPoolExecutor
# menjalankan initializer sekali per proda; extractor dibuka di situ dan dipakai
# ulang untuk semua video yang ditangani proses tersebut.
_WORKER_EXTRACTOR = None


def _worker_init():
    """Initializer per proses: buka satu HolisticLandmarkExtractor."""
    global _WORKER_EXTRACTOR
    # static_image_mode=False → SAMA dengan build_sequences.py (mode video).
    _WORKER_EXTRACTOR = HolisticLandmarkExtractor(static_image_mode=False)


def _process_one_video(task):
    """Proses SATU video → record dict, atau tandai gagal.

    Langkah identik build_sequences.build_dataset (per video):
        read_frames -> extract_sequence -> (cek ada tangan)
                    -> normalize_sequence -> resample_sequence(30)

    Args:
        task: tuple (video_id, video_path_str, label, label_idx).

    Returns:
        dict record bila sukses:
            {"video_id", "label", "label_idx", "seqs": [seq]}
        atau {"video_id", "failed": True} bila video gagal / tak ada tangan.
    """
    global _WORKER_EXTRACTOR
    video_id, video_path, label, label_idx = task

    frames = read_frames(video_path)
    if frames is None:
        return {"video_id": video_id, "failed": True}

    seq = extract_sequence(frames, _WORKER_EXTRACTOR)        # (T,126)
    if not np.any(seq):                                      # tak ada tangan sama sekali
        return {"video_id": video_id, "failed": True}

    seq = normalize_sequence(seq)
    seq = resample_sequence(seq, config.SEQUENCE_LENGTH)     # (30,126)

    return {
        "video_id": video_id,
        "label": label,
        "label_idx": label_idx,
        "seqs": [seq],
    }


# --------------------------------------------------------------------------- #
# BUILD DATASET (paralel)
# --------------------------------------------------------------------------- #
def build_dataset_parallel(raw_dir=None, out_dir=None, n_workers=None):
    """Versi paralel dari build_dataset. Output identik dgn versi serial.

    Args:
        raw_dir   : folder dataset mentah (default config.RAW_DIR).
        out_dir   : folder output (default config.PROCESSED_DIR).
        n_workers : jumlah proses. Default = max(1, cpu_count - 1).

    Returns:
        dict {"train": (X, y), "val": (X, y), "test": (X, y)}.

    Raises:
        RuntimeError: bila tak ada sequence valid, atau anti-leakage GAGAL.
    """
    raw_dir = raw_dir or config.RAW_DIR
    out_dir = out_dir or config.PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    labels = config.get_labels_from_dataset(raw_dir)
    label_to_idx = {name: i for i, name in enumerate(labels)}
    logger.info(
        "Mulai build dataset PARALEL. %d kelas dari %s | workers=%d",
        len(labels), raw_dir, n_workers,
    )

    # --- kumpulkan daftar tugas (deterministik: kelas terurut, video terurut) ---
    tasks = []
    for name in labels:
        class_dir = raw_dir / name
        if not class_dir.is_dir():
            logger.warning("Folder kelas tak ada: %s", class_dir)
            continue
        for video_path in _list_videos(class_dir):
            video_id = video_path.relative_to(raw_dir).as_posix()
            tasks.append((video_id, str(video_path), name, label_to_idx[name]))

    if not tasks:
        raise RuntimeError(f"Tidak ada video ditemukan di {raw_dir}")

    logger.info("Total video antri: %d", len(tasks))

    # --- eksekusi paralel ---
    records = []
    failed = []
    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=_worker_init
    ) as executor:
        futures = {executor.submit(_process_one_video, t): t[0] for t in tasks}
        for fut in tqdm(
            as_completed(futures), total=len(futures), unit="vid", desc="extract"
        ):
            res = fut.result()
            if res.get("failed"):
                failed.append(res["video_id"])
            else:
                records.append(res)

    if not records:
        raise RuntimeError(f"Tidak ada sequence valid dari {raw_dir}")

    # --- KUNCI DETERMINISME: urutkan records by video_id ---
    # Proses selesai dalam urutan non-deterministik. Mengurutkan di sini menjamin
    # isi X_*.npy / y_*.npy byte-identik dengan versi serial build_sequences.py.
    records.sort(key=lambda r: r["video_id"])

    # --- split level video + verifikasi anti-leakage (logika di-reuse) ---
    split_map = split_videos_stratified(records)
    passed = verify_split(records, split_map, labels)

    # --- rakit array per split (urutan records sudah deterministik) ---
    buckets = {sp: ([], []) for sp in _SPLITS}
    for rec in records:
        Xs, ys = buckets[split_map[rec["video_id"]]]
        for seq in rec["seqs"]:
            Xs.append(seq)
            ys.append(rec["label_idx"])
    arrays = {
        sp: (np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64))
        for sp, (Xs, ys) in buckets.items()
    }

    # --- simpan artefak (sama persis dgn versi serial) ---
    file_paths = {
        "train": (out_dir / config.X_TRAIN_PATH.name, out_dir / config.Y_TRAIN_PATH.name),
        "val": (out_dir / config.X_VAL_PATH.name, out_dir / config.Y_VAL_PATH.name),
        "test": (out_dir / config.X_TEST_PATH.name, out_dir / config.Y_TEST_PATH.name),
    }
    for sp in _SPLITS:
        X, y = arrays[sp]
        xp, yp = file_paths[sp]
        np.save(xp, X)
        np.save(yp, y)

    idx_to_label = {i: name for name, i in label_to_idx.items()}
    (out_dir / config.PROCESSED_LABEL_MAP_PATH.name).write_text(
        json.dumps(idx_to_label, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest_path = out_dir / config.MANIFEST_PATH.name
    _write_manifest(records, split_map, manifest_path)

    # --- ringkasan logging ---
    logger.info("=== Ringkasan dataset (ter-split level video, PARALEL) ===")
    for sp in _SPLITS:
        X, y = arrays[sp]
        logger.info("  %-5s : X=%s y=%s", sp, X.shape, y.shape)
    logger.info("Total video  : %d | video gagal : %d", len(records), len(failed))
    for fp in failed:
        logger.info("    [gagal] %s", fp)
    logger.info("Disimpan ke  : %s", out_dir)

    if not passed:
        raise RuntimeError(
            "LEAKAGE CHECK FAIL: ada video lintas split (lihat output di atas)."
        )

    return arrays


def main():
    parser = argparse.ArgumentParser(
        description="Ekstraksi landmark PARALEL → dataset ter-split (.npy)."
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Jumlah proses paralel. Default: cpu_count - 1.",
    )
    parser.add_argument(
        "--raw-dir", type=str, default=None,
        help="Folder dataset mentah. Default: config.RAW_DIR.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Folder output. Default: config.PROCESSED_DIR.",
    )
    args = parser.parse_args()

    from pathlib import Path
    raw_dir = Path(args.raw_dir) if args.raw_dir else None
    out_dir = Path(args.out_dir) if args.out_dir else None

    build_dataset_parallel(
        raw_dir=raw_dir, out_dir=out_dir, n_workers=args.workers
    )


if __name__ == "__main__":
    main()
