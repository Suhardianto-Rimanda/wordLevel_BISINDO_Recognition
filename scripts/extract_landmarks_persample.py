"""
extract_landmarks_persample.py — Ekstraksi landmark PER-SAMPLE (1 video → 1 .npy).

Berbeda dari build_sequences.py / extract_landmarks_split.py yang menyatukan banyak
video ke array GABUNGAN per split (X_train.npy dst), skrip ini menulis SATU file
.npy untuk SETIAP video, TANPA split & TANPA penggabungan. Tujuannya: ekstraksi
MediaPipe (mahal) cukup dijalankan SEKALI; pembagian dataset jadi soal menyusun
ulang file (murah) — lihat scripts/assemble_dataset.py.

Logika per video IDENTIK build_sequences.py (fungsi di-IMPORT, bukan ditulis ulang):
    read_frames -> extract_sequence (T,126) -> normalize_sequence
                -> resample_sequence (SEQUENCE_LENGTH,126)
sehingga (SEQUENCE_LENGTH,126) per video sama persis dengan dataset lama.

ADITIF — tidak menyentuh build_sequences.py, extract_landmarks_split.py, maupun
pipeline lain.

Output (ke --out-dir, default config.PROCESSED_DIR/"landmarks_persample"):
    <out_dir>/<kelas>/<nama_video>.npy   : array (SEQUENCE_LENGTH, 126), 1 per video
    <out_dir>/manifest_persample.csv     : npy_path,video_id,label,signer,jumlah_sequence
    <out_dir>/label_map.json             : {index: kata} (urutan konsisten dataset lama)

Jalankan:
    python scripts/extract_landmarks_persample.py
    python scripts/extract_landmarks_persample.py --raw-dir data/raw_signer03 --skip-existing
    python scripts/extract_landmarks_persample.py --out-dir data/persample --yes

Opsi:
    --raw-dir PATH   Folder sumber video <kelas>/<video>. Default config.RAW_DIR.
    --out-dir PATH   Folder output. Default config.PROCESSED_DIR/"landmarks_persample".
    --skip-existing  Lewati video yang .npy-nya sudah ada (incremental, tak ekstrak ulang).
    --yes            Lewati konfirmasi awal.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# root proyek + folder scripts/ ke sys.path (agar bisa import config, src, & sibling)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from src.preprocessing.extract_frames import read_frames
from src.preprocessing.extract_landmarks import make_extractor, extract_sequence
# reuse logika numerik yang SAMA agar hasil identik dgn dataset lama
from src.preprocessing.build_sequences import normalize_sequence, resample_sequence
# reuse util sibling (deteksi signer, daftar video, konfirmasi) — jangan tulis ulang
from extract_landmarks_split import _detect_signer, _list_videos, _confirm
from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_OUT_NAME = "landmarks_persample"
_MANIFEST_NAME = "manifest_persample.csv"


def extract_persample(raw_dir=None, out_dir=None, skip_existing=False, assume_yes=False):
    """Ekstrak landmark per-video → 1 .npy/video + manifest + label_map.

    Args:
        raw_dir: folder sumber video (default config.RAW_DIR).
        out_dir: folder output (default config.PROCESSED_DIR/"landmarks_persample").
        skip_existing: lewati video yang .npy-nya sudah ada (tetap dicatat di manifest).
        assume_yes: lewati konfirmasi awal.

    Returns:
        dict ringkasan {"ok","skip","fail","out_dir"} atau None bila dibatalkan.
    """
    raw_dir = Path(raw_dir) if raw_dir else config.RAW_DIR
    out_dir = Path(out_dir) if out_dir else (config.PROCESSED_DIR / _DEFAULT_OUT_NAME)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = config.get_labels_from_dataset(raw_dir)
    label_to_idx = {name: i for i, name in enumerate(labels)}

    print("=" * 70)
    print(f"Sumber video  : {raw_dir}")
    print(f"Output        : {out_dir}")
    print(f"Kelas         : {len(labels)} | skip-existing={skip_existing}")
    print("=" * 70)
    if not assume_yes and not _confirm("Lanjut ekstraksi per-sample?"):
        print("Dibatalkan. Tidak ada file yang ditulis.")
        return None

    records = []            # baris manifest (termasuk yang di-skip)
    n_ok = n_skip = 0
    failed = []

    # satu extractor untuk seluruh proses (pola build_dataset). static_image_mode=False
    # → SAMA dengan build_sequences.py / extract_landmarks_split.py (mode video).
    with make_extractor(static_image_mode=False) as extractor:
        for name in labels:
            class_dir = raw_dir / name
            if not class_dir.is_dir():
                logger.warning("Folder kelas tak ada: %s", class_dir)
                continue
            out_class = out_dir / name
            out_class.mkdir(parents=True, exist_ok=True)

            for video_path in tqdm(_list_videos(class_dir), desc=name, unit="vid"):
                video_id = video_path.relative_to(raw_dir).as_posix()
                npy_path = out_class / (video_path.stem + ".npy")

                # incremental: .npy sudah ada → lewati ekstraksi, TETAP catat di manifest
                if skip_existing and npy_path.exists():
                    n_skip += 1
                    records.append({
                        "npy_path": npy_path.relative_to(out_dir).as_posix(),
                        "video_id": video_id,
                        "label": name,
                        "signer": _detect_signer(video_id),
                        "jumlah_sequence": 1,
                    })
                    continue

                frames = read_frames(video_path)
                if frames is None:
                    failed.append(video_id)
                    continue
                seq = extract_sequence(frames, extractor)             # (T,126)
                if not np.any(seq):                                   # tak ada tangan sama sekali
                    logger.warning("Tak ada tangan terdeteksi: %s", video_path)
                    failed.append(video_id)
                    continue
                seq = normalize_sequence(seq)                         # identik training
                seq = resample_sequence(seq, config.SEQUENCE_LENGTH)  # (SEQUENCE_LENGTH,126)

                np.save(npy_path, seq)
                n_ok += 1
                records.append({
                    "npy_path": npy_path.relative_to(out_dir).as_posix(),
                    "video_id": video_id,
                    "label": name,
                    "signer": _detect_signer(video_id),
                    "jumlah_sequence": 1,
                })

    # --- manifest_persample.csv (ditimpa tiap run, mencakup ok + skip) ---
    manifest_path = out_dir / _MANIFEST_NAME
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["npy_path", "video_id", "label", "signer", "jumlah_sequence"])
        for rec in sorted(records, key=lambda r: r["npy_path"]):
            writer.writerow([
                rec["npy_path"], rec["video_id"], rec["label"],
                rec["signer"], rec["jumlah_sequence"],
            ])

    # --- label_map.json (urutan indeks konsisten dataset lama) ---
    idx_to_label = {i: name for name, i in label_to_idx.items()}
    (out_dir / config.PROCESSED_LABEL_MAP_PATH.name).write_text(
        json.dumps(idx_to_label, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- ringkasan ---
    logger.info("=== Ringkasan ekstraksi per-sample ===")
    logger.info("  sukses  : %d", n_ok)
    logger.info("  dilewati: %d (skip-existing)", n_skip)
    logger.info("  gagal   : %d", len(failed))
    for fp in failed:
        logger.info("    [gagal] %s", fp)
    logger.info("Output    : %s", out_dir)
    logger.info("manifest  : %s", manifest_path)
    logger.info("label_map : %s", out_dir / config.PROCESSED_LABEL_MAP_PATH.name)

    if n_ok == 0 and n_skip == 0:
        logger.warning("Tidak ada sample tertulis (cek raw-dir & folder kelas).")

    return {"ok": n_ok, "skip": n_skip, "fail": len(failed), "out_dir": str(out_dir)}


def main():
    parser = argparse.ArgumentParser(
        description="Ekstraksi landmark per-sample: 1 video → 1 .npy (tanpa split)."
    )
    parser.add_argument("--raw-dir", default=None,
                        help="Folder sumber video <kelas>/<video>. Default config.RAW_DIR.")
    parser.add_argument("--out-dir", default=None,
                        help="Folder output. Default config.PROCESSED_DIR/'landmarks_persample'.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Lewati video yang .npy-nya sudah ada (incremental).")
    parser.add_argument("--yes", action="store_true", help="Lewati konfirmasi awal.")
    args = parser.parse_args()

    extract_persample(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        skip_existing=args.skip_existing,
        assume_yes=args.yes,
    )


if __name__ == "__main__":
    main()
