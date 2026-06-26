"""
extract_landmarks_split.py — Ekstraksi landmark → split FLEKSIBEL (pilih sendiri).

Berbeda dari build_sequences.py yang selalu membagi train/val/test sekaligus,
skrip ini memproses video lalu menyimpannya ke SATU ATAU BEBERAPA split yang
DIPILIH PENGGUNA, dengan rasio yang ditentukan sendiri. Cocok untuk:
  - menambah data signer baru khusus sebagai test (1 split), atau
  - mengekstrak hanya train+val dengan rasio tertentu (2 split), atau
  - kombinasi apa pun (train+test, val+test, bahkan train+val+test).

Metode pembagian SAMA dengan build_sequences.py: split di LEVEL VIDEO, stratified
per kelas, ber-seed (config.RANDOM_SEED). Logika preprocessing (normalize_sequence,
resample_sequence, ekstraksi 126 fitur) juga di-reuse dari modul yang sama, sehingga
landmark yang dihasilkan kompatibel dengan dataset lama.

Jalankan (mode interaktif — akan ada konfirmasi split & rasio):
    python scripts/extract_landmarks_split.py

Atau langsung via argumen:
    python scripts/extract_landmarks_split.py --split test
    python scripts/extract_landmarks_split.py --split train,val --ratio 80:20
    python scripts/extract_landmarks_split.py --split train,val,test --ratio 70:15:15
    python scripts/extract_landmarks_split.py --split val,test --ratio 50:50 --raw-dir data/raw_signer03

Opsi:
    --split LIST    Daftar split dipisah koma: train,val,test (urutan bebas).
                    Bila kosong → ditanyakan di terminal.
    --ratio LIST    Rasio dipisah titik dua, sesuai URUTAN --split. Mis. 80:20.
                    Wajib bila split > 1. Jumlah harus = 100. Untuk 1 split,
                    rasio diabaikan (semua → split itu).
    --raw-dir PATH  Folder sumber video (default: config.RAW_DIR).
    --out-dir PATH  Folder output (default: config.PROCESSED_DIR).
    --append        Gabungkan ke X_<split>.npy yang sudah ada (bukan timpa).
    --yes           Lewati konfirmasi akhir (untuk batch).

Output (ke out_dir), untuk tiap split terpilih:
    X_<split>.npy : (n, 30, 126)
    y_<split>.npy : (n,)
    label_map.json            : {index: kata}
    manifest_extract.csv      : video_id,label,split,signer,jumlah_sequence
"""

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

# pastikan root proyek di sys.path saat dijalankan langsung
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.preprocessing.extract_frames import read_frames
from src.preprocessing.extract_landmarks import make_extractor, extract_sequence
# reuse logika normalisasi & resample yang SAMA agar hasil identik dgn dataset lama
from src.preprocessing.build_sequences import normalize_sequence, resample_sequence
from src.utils.logger import get_logger

logger = get_logger(__name__)

_ALL_SPLITS = ("train", "val", "test")
_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


# --------------------------------------------------------------------------- #
# UTIL UMUM
# --------------------------------------------------------------------------- #
def _list_videos(class_dir):
    """Daftar file video dalam satu folder kelas (terurut)."""
    return sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _VIDEO_EXTS
    )


def _detect_signer(video_id):
    """Ambil identitas signer dari video_id bila ada pola 'SignerXX'."""
    m = re.search(r"(Signer\d+)", video_id, flags=re.IGNORECASE)
    return m.group(1) if m else ""


def _split_paths(out_dir):
    """Peta split → (X_path, y_path) sesuai nama dari config."""
    return {
        "train": (out_dir / config.X_TRAIN_PATH.name, out_dir / config.Y_TRAIN_PATH.name),
        "val":   (out_dir / config.X_VAL_PATH.name,   out_dir / config.Y_VAL_PATH.name),
        "test":  (out_dir / config.X_TEST_PATH.name,  out_dir / config.Y_TEST_PATH.name),
    }


# --------------------------------------------------------------------------- #
# PARSING & VALIDASI SPLIT + RASIO
# --------------------------------------------------------------------------- #
def _parse_splits(raw):
    """'train,val' → ['train','val'] (validasi nama & duplikat)."""
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("Daftar split kosong.")
    seen = []
    for p in parts:
        if p not in _ALL_SPLITS:
            raise ValueError(f"Split '{p}' tidak dikenal (pilih: train/val/test).")
        if p in seen:
            raise ValueError(f"Split '{p}' tertulis lebih dari sekali.")
        seen.append(p)
    return seen


def _parse_ratios(raw, n):
    """'80:20' → [0.8, 0.2]. Validasi: jumlah harus = 100."""
    parts = [p.strip() for p in raw.split(":") if p.strip() != ""]
    if len(parts) != n:
        raise ValueError(
            f"Jumlah angka rasio ({len(parts)}) tidak sama dengan jumlah split ({n})."
        )
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"Rasio harus berupa angka, dapat: '{raw}'.")
    if any(v <= 0 for v in vals):
        raise ValueError("Setiap rasio harus lebih besar dari 0.")
    total = sum(vals)
    if abs(total - 100.0) > 1e-6:
        raise ValueError(f"Jumlah rasio harus = 100, sekarang {total:g}.")
    return [v / 100.0 for v in vals]


def _ask_splits():
    """Tanya daftar split lewat terminal sampai valid."""
    menu = (
        "\nPilih split yang ingin dihasilkan (boleh lebih dari satu, pisahkan koma).\n"
        "  Contoh: 'test'            -> hanya test\n"
        "          'train,val'       -> train dan val\n"
        "          'train,val,test'  -> ketiganya\n"
        "Masukkan pilihan: "
    )
    while True:
        try:
            return _parse_splits(input(menu))
        except ValueError as e:
            print(f"  {e} Coba lagi.")


def _ask_ratios(splits):
    """Tanya rasio sesuai urutan splits (hanya bila >1 split)."""
    label = ":".join(splits)
    contoh = "80:20" if len(splits) == 2 else "70:15:15"
    prompt = (
        f"\nMasukkan rasio untuk {label} (jumlah harus 100).\n"
        f"  Contoh untuk {len(splits)} split: {contoh}\n"
        f"Rasio {label} = "
    )
    while True:
        try:
            return _parse_ratios(input(prompt), len(splits))
        except ValueError as e:
            print(f"  {e} Coba lagi.")


def _confirm(prompt):
    """Konfirmasi ya/tidak."""
    while True:
        ans = input(f"{prompt} [y/t]: ").strip().lower()
        if ans in ("y", "ya", "yes"):
            return True
        if ans in ("t", "tidak", "n", "no"):
            return False
        print("  Jawab 'y' atau 't'.")


# --------------------------------------------------------------------------- #
# SPLIT LEVEL VIDEO, STRATIFIED PER KELAS (mengikuti build_sequences.py)
# --------------------------------------------------------------------------- #
def split_videos_stratified_general(records, splits, ratios, seed=None):
    """Peta video_id → split untuk DAFTAR split & rasio sembarang.

    Mengikuti pendekatan build_sequences.py: per kelas, urutkan deterministik →
    shuffle ber-seed → iris menurut rasio. Bila satu kelas punya video cukup
    (>= jumlah split), tiap split dijamin minimal 1 video. Sisa pembulatan
    didistribusikan mulai dari split pertama.

    Returns:
        dict {video_id: split}.
    """
    seed = config.RANDOM_SEED if seed is None else seed
    n_split = len(splits)

    by_class = defaultdict(list)
    for rec in records:
        by_class[rec["label"]].append(rec["video_id"])

    rng = random.Random(seed)
    split_map = {}
    for label in sorted(by_class):
        vids = sorted(by_class[label])      # deterministik sebelum shuffle
        rng.shuffle(vids)
        n = len(vids)

        counts = [int(n * r) for r in ratios]       # floor awal
        if n >= n_split:                            # jamin tiap split >=1
            for i in range(n_split):
                if counts[i] == 0:
                    counts[i] = 1

        # sesuaikan agar total == n
        diff = n - sum(counts)
        i = 0
        while diff != 0:
            idx = i % n_split
            if diff > 0:
                counts[idx] += 1
                diff -= 1
            else:
                if counts[idx] > 1:
                    counts[idx] -= 1
                    diff += 1
            i += 1
            if i > 10 * n_split:                    # pengaman loop
                break

        assigned = []
        for sp, c in zip(splits, counts):
            assigned += [sp] * c
        # jaga panjang assigned == n (akibat pengaman di atas)
        assigned = assigned[:n] + [splits[0]] * max(0, n - len(assigned))
        for vid, sp in zip(vids, assigned):
            split_map[vid] = sp
    return split_map


# --------------------------------------------------------------------------- #
# EKSTRAKSI
# --------------------------------------------------------------------------- #
def extract_landmarks(splits, ratios, raw_dir=None, out_dir=None,
                      append=False, assume_yes=False):
    """Proses video → simpan ke split-split terpilih sesuai rasio.

    Returns:
        dict {split: (X, y)} yang disimpan, atau None bila dibatalkan.
    """
    raw_dir = Path(raw_dir) if raw_dir else config.RAW_DIR
    out_dir = Path(out_dir) if out_dir else config.PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = config.get_labels_from_dataset(raw_dir)
    label_to_idx = {name: i for i, name in enumerate(labels)}
    logger.info("Mulai ekstraksi → split %s (rasio %s). %d kelas dari %s",
                splits, [round(r, 3) for r in ratios], len(labels), raw_dir)

    records = []
    failed = []

    with make_extractor(static_image_mode=False) as extractor:
        for name in labels:
            class_dir = raw_dir / name
            if not class_dir.is_dir():
                logger.warning("Folder kelas tak ada: %s", class_dir)
                continue
            for video_path in tqdm(_list_videos(class_dir), desc=name, unit="vid"):
                video_id = video_path.relative_to(raw_dir).as_posix()
                frames = read_frames(video_path)
                if frames is None:
                    failed.append(video_id)
                    continue
                seq = extract_sequence(frames, extractor)            # (T,126)
                if not np.any(seq):
                    logger.warning("Tak ada tangan terdeteksi: %s", video_path)
                    failed.append(video_id)
                    continue
                seq = normalize_sequence(seq)                        # identik
                seq = resample_sequence(seq, config.SEQUENCE_LENGTH) # (30,126)
                records.append({
                    "video_id": video_id,
                    "label": name,
                    "label_idx": label_to_idx[name],
                    "signer": _detect_signer(video_id),
                    "seqs": [seq],
                })

    if not records:
        raise RuntimeError(f"Tidak ada sequence valid dari {raw_dir}")

    # --- tentukan split tiap video ---
    if len(splits) == 1:
        split_map = {rec["video_id"]: splits[0] for rec in records}
    else:
        split_map = split_videos_stratified_general(records, splits, ratios)

    # --- rakit array per split terpilih ---
    buckets = {sp: ([], []) for sp in splits}
    for rec in records:
        sp = split_map[rec["video_id"]]
        Xs, ys = buckets[sp]
        for seq in rec["seqs"]:
            Xs.append(seq)
            ys.append(rec["label_idx"])
    new_arrays = {
        sp: (np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64))
        for sp, (Xs, ys) in buckets.items()
    }

    paths = _split_paths(out_dir)

    # --- ringkasan + (bila append) hitung total akhir ---
    print("=" * 70)
    print(f"Sumber video    : {raw_dir}")
    print(f"Video terproses : {len(records)} | gagal: {len(failed)}")
    print(f"Split terpilih  : {', '.join(splits)}")
    if len(splits) > 1:
        print("Rasio           : " +
              ", ".join(f"{sp}={r*100:g}%" for sp, r in zip(splits, ratios)))
    print("-" * 70)
    final = {}
    for sp in splits:
        X_new, y_new = new_arrays[sp]
        xp, yp = paths[sp]
        if append and xp.exists() and yp.exists():
            X_old = np.load(xp); y_old = np.load(yp)
            if X_old.shape[1:] != X_new.shape[1:]:
                raise RuntimeError(
                    f"[{sp}] bentuk tak cocok untuk append: "
                    f"lama {X_old.shape[1:]} vs baru {X_new.shape[1:]}"
                )
            X_fin = np.concatenate([X_old, X_new], 0)
            y_fin = np.concatenate([y_old, y_new], 0)
            mode = f"APPEND ({len(X_old)}+{len(X_new)}={len(X_fin)})"
        else:
            X_fin, y_fin = X_new, y_new
            mode = f"TULIS BARU ({len(X_new)})"
        final[sp] = (X_fin, y_fin)
        print(f"  {sp:<5} -> {xp.name:<14} {str(X_new.shape):<18} | {mode}")
    print("=" * 70)

    if not assume_yes and not _confirm("Lanjut menyimpan?"):
        print("Dibatalkan. Tidak ada file yang ditulis.")
        return None

    # --- simpan tiap split ---
    for sp in splits:
        X_fin, y_fin = final[sp]
        xp, yp = paths[sp]
        np.save(xp, X_fin)
        np.save(yp, y_fin)

    # --- label_map.json ---
    idx_to_label = {i: name for name, i in label_to_idx.items()}
    (out_dir / config.PROCESSED_LABEL_MAP_PATH.name).write_text(
        json.dumps(idx_to_label, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- manifest (mencatat split hasil pembagian + signer) ---
    manifest_path = out_dir / "manifest_extract.csv"
    write_header = not (append and manifest_path.exists())
    mode = "a" if (append and manifest_path.exists()) else "w"
    with open(manifest_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["video_id", "label", "split", "signer", "jumlah_sequence"])
        for rec in sorted(records, key=lambda r: r["video_id"]):
            writer.writerow([
                rec["video_id"], rec["label"], split_map[rec["video_id"]],
                rec["signer"], len(rec["seqs"]),
            ])

    logger.info("=== Ringkasan tersimpan ===")
    for sp in splits:
        X_fin, _ = final[sp]
        logger.info("  %-5s : %s -> %s", sp, X_fin.shape, paths[sp][0].name)
    logger.info("label_map : %s", out_dir / config.PROCESSED_LABEL_MAP_PATH.name)
    logger.info("manifest  : %s", manifest_path)
    if failed:
        logger.info("Video gagal (%d):", len(failed))
        for fp in failed:
            logger.info("    [gagal] %s", fp)

    return final


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Ekstraksi landmark ke split fleksibel (1+ split, rasio sendiri)."
    )
    parser.add_argument("--split", default=None,
                        help="Daftar split dipisah koma (mis. 'train,val'). "
                             "Kosong -> ditanya di terminal.")
    parser.add_argument("--ratio", default=None,
                        help="Rasio dipisah ':' sesuai urutan --split (mis. '80:20'). "
                             "Jumlah harus 100. Wajib bila split > 1.")
    parser.add_argument("--raw-dir", default=None, help="Folder sumber video.")
    parser.add_argument("--out-dir", default=None, help="Folder output.")
    parser.add_argument("--append", action="store_true",
                        help="Gabungkan ke file split yang sudah ada.")
    parser.add_argument("--yes", action="store_true", help="Lewati konfirmasi akhir.")
    args = parser.parse_args()

    # --- tentukan split ---
    if args.split:
        try:
            splits = _parse_splits(args.split)
        except ValueError as e:
            parser.error(str(e))
    else:
        splits = _ask_splits()

    # --- tentukan rasio ---
    if len(splits) == 1:
        ratios = [1.0]
        if args.ratio:
            print("(Info: hanya 1 split dipilih, --ratio diabaikan.)")
    else:
        if args.ratio:
            try:
                ratios = _parse_ratios(args.ratio, len(splits))
            except ValueError as e:
                parser.error(str(e))
        else:
            ratios = _ask_ratios(splits)

    extract_landmarks(
        splits=splits,
        ratios=ratios,
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        append=args.append,
        assume_yes=args.yes,
    )


if __name__ == "__main__":
    main()