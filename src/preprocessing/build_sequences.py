"""
build_sequences.py — Rakit dataset training dari video → data TER-SPLIT (.npy).

Pipeline per video:
    read_frames -> extract_sequence (T,126) -> normalize_sequence
                -> resample_sequence (30,126) -> simpan sebagai record per-video

Split dilakukan di LEVEL VIDEO (stratified per kelas, rasio & seed dari config):
satu video utuh hanya masuk ke SATU split (train/val/test). Bila satu video
menghasilkan >1 sequence, SEMUA sequence-nya ikut split video induknya. Tiap
sequence mempertahankan jejak asal (video_id) supaya bisa diaudit.

Util utama:
    normalize_sequence       : wrist-relative + scale (invarian posisi signer)
    resample_sequence        : standardisasi panjang ke SEQUENCE_LENGTH (30)
    split_videos_stratified  : peta video_id -> split (level video, per kelas)
    verify_split             : cek anti-leakage (irisan video antar split kosong)

Output (ke config.PROCESSED_DIR):
    X_train/X_val/X_test .npy : (n, 30, 126)
    y_train/y_val/y_test .npy : (n,) index kelas
    label_map.json            : {index: kata}
    manifest.csv              : video_id,label,split,jumlah_sequence

Output gabungan lama (landmarks/X.npy, y.npy) TIDAK lagi ditulis.

Logging: jumlah video & sequence per split per kelas, video gagal, hasil
verifikasi anti-leakage, shape akhir.
"""

import csv
import json
import random
from collections import Counter, defaultdict

import numpy as np
from tqdm import tqdm

import config
from src.preprocessing.extract_frames import read_frames
from src.preprocessing.extract_landmarks import (
    HolisticLandmarkExtractor,
    extract_sequence,
    HAND_FEATURE_LEN,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

_EPS = 1e-6
_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


# --------------------------------------------------------------------------- #
# NORMALISASI
# --------------------------------------------------------------------------- #
def _normalize_hand_block(block):
    """Normalisasi satu blok tangan (T, 63) wrist-relative + scale.

    - Geser tiap titik relatif ke wrist (landmark 0).
    - Skala dengan ukuran tangan = jarak euclidean terjauh titik→wrist per frame.
    - Frame all-zero (tangan tak terdeteksi) dibiarkan zero (tak dinormalisasi).
    """
    T = block.shape[0]
    pts = block.reshape(T, config.HAND_LANDMARKS, config.HAND_DIMS)  # (T,21,3)

    # frame mana yang punya tangan (tidak semua nol)
    present = np.any(pts != 0, axis=(1, 2))  # (T,)

    out = pts.copy()
    if np.any(present):
        wrist = pts[present, 0:1, :]                 # (n,1,3)
        centered = pts[present] - wrist              # geser ke wrist
        # ukuran tangan per frame = jarak terjauh titik ke wrist
        dist = np.linalg.norm(centered, axis=2)      # (n,21)
        scale = dist.max(axis=1, keepdims=True)[:, :, None]  # (n,1,1)
        out[present] = centered / np.maximum(scale, _EPS)

    return out.reshape(T, HAND_FEATURE_LEN)


def normalize_sequence(seq):
    """Normalisasi sequence (T, 126): tiap tangan diproses terpisah."""
    seq = np.asarray(seq, dtype=np.float32)
    left = _normalize_hand_block(seq[:, :HAND_FEATURE_LEN])
    right = _normalize_hand_block(seq[:, HAND_FEATURE_LEN:])
    return np.concatenate([left, right], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# RESAMPLE PANJANG
# --------------------------------------------------------------------------- #
def resample_sequence(seq, target=None):
    """Standardisasi sequence (T, F) → (target, F) via linear interpolation.

    - T == 0       : kembalikan zeros (target, F).
    - T == 1       : tile satu frame sebanyak target.
    - T == target  : kembalikan apa adanya.
    - selainnya    : interpolasi tiap kolom fitur sepanjang sumbu waktu.
    """
    target = target or config.SEQUENCE_LENGTH
    seq = np.asarray(seq, dtype=np.float32)
    T = seq.shape[0]
    F = seq.shape[1] if seq.ndim == 2 else config.FEATURE_DIM

    if T == 0:
        return np.zeros((target, F), dtype=np.float32)
    if T == 1:
        return np.repeat(seq, target, axis=0).astype(np.float32)
    if T == target:
        return seq.astype(np.float32)

    orig_idx = np.linspace(0.0, T - 1, num=T)
    new_idx = np.linspace(0.0, T - 1, num=target)
    out = np.empty((target, F), dtype=np.float32)
    for f in range(F):
        out[:, f] = np.interp(new_idx, orig_idx, seq[:, f])
    return out


# --------------------------------------------------------------------------- #
# BUILD DATASET
# --------------------------------------------------------------------------- #
def _list_videos(class_dir):
    """Daftar file video dalam satu folder kelas (terurut)."""
    return sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _VIDEO_EXTS
    )


# --------------------------------------------------------------------------- #
# SPLIT LEVEL VIDEO + VERIFIKASI ANTI-LEAKAGE
# --------------------------------------------------------------------------- #
_SPLITS = ("train", "val", "test")


def split_videos_stratified(records, ratios=None, seed=None):
    """Peta ``video_id`` → split, stratified per kelas, di LEVEL VIDEO.

    Tiap video utuh hanya masuk SATU split. Per kelas: urutkan (deterministik) →
    shuffle ber-seed → iris menurut rasio (floor; sisa pembulatan → train). Bila
    jumlah video kelas >= 3, val & test dijamin minimal 1 (selama train >= 1).

    Args:
        records: list dict berisi minimal ``{"video_id", "label"}``.
        ratios : (train, val, test). Default dari config (jumlah harus ~1.0).
        seed   : int seed shuffle. Default ``config.RANDOM_SEED``.

    Returns:
        dict ``{video_id: split}``.
    """
    if ratios is None:
        ratios = (config.SPLIT_TRAIN_RATIO, config.SPLIT_VAL_RATIO,
                  config.SPLIT_TEST_RATIO)
    seed = config.RANDOM_SEED if seed is None else seed
    _, r_val, r_test = ratios

    by_class = defaultdict(list)
    for rec in records:
        by_class[rec["label"]].append(rec["video_id"])

    rng = random.Random(seed)
    split_map = {}
    for label in sorted(by_class):
        vids = sorted(by_class[label])      # deterministik sebelum shuffle
        rng.shuffle(vids)
        n = len(vids)

        n_val = int(n * r_val)
        n_test = int(n * r_test)
        if n >= 3:                          # jamin val & test terwakili
            n_val = max(1, n_val)
            n_test = max(1, n_test)
        n_train = n - n_val - n_test
        while n_train < 1 and (n_val > 0 or n_test > 0):  # kelas sangat kecil
            if n_test >= n_val and n_test > 0:
                n_test -= 1
            elif n_val > 0:
                n_val -= 1
            n_train = n - n_val - n_test

        assigned = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
        for vid, sp in zip(vids, assigned):
            split_map[vid] = sp
    return split_map


def verify_split(records, split_map, labels):
    """Cek anti-leakage + cetak ringkasan. Kembalikan ``True`` bila LOLOS.

    - Irisan himpunan ``video_id`` antar split HARUS kosong.
    - Cetak jumlah video & sequence per split per kelas (tabel ringkas).
    - Peringatan bila ada kelas tak terwakili di salah satu split.
    """
    vids_per_split = {sp: set() for sp in _SPLITS}
    n_vid = {sp: Counter() for sp in _SPLITS}
    n_seq = {sp: Counter() for sp in _SPLITS}
    for rec in records:
        sp = split_map[rec["video_id"]]
        vids_per_split[sp].add(rec["video_id"])
        n_vid[sp][rec["label"]] += 1
        n_seq[sp][rec["label"]] += len(rec["seqs"])

    # --- 1. overlap check (level video) ---
    offenders = set()
    for i, a in enumerate(_SPLITS):
        for b in _SPLITS[i + 1:]:
            offenders |= vids_per_split[a] & vids_per_split[b]
    passed = not offenders

    print("=" * 64)
    if passed:
        print("LEAKAGE CHECK (video overlap): PASS")
    else:
        print("LEAKAGE CHECK (video overlap): FAIL")
        print(f"  {len(offenders)} video bocor lintas split:")
        for vid in sorted(offenders):
            print(f"    [BOCOR] {vid}")

    # --- 2. tabel video/sequence per split per kelas ---
    print("-" * 64)
    print(f"{'kelas':<22} | {'train':>11} | {'val':>11} | {'test':>11}")
    print(f"{'(vid/seq)':<22} | {'vid/seq':>11} | {'vid/seq':>11} | {'vid/seq':>11}")
    print("-" * 64)
    for name in labels:
        cells = " | ".join(
            f"{n_vid[sp][name]:>4}/{n_seq[sp][name]:<6}" for sp in _SPLITS
        )
        print(f"{name:<22} | {cells}")
    print("-" * 64)
    total = " | ".join(
        f"{sum(n_vid[sp].values()):>4}/{sum(n_seq[sp].values()):<6}"
        for sp in _SPLITS
    )
    print(f"{'TOTAL':<22} | {total}")
    print("-" * 64)

    # --- 3. kelas tak terwakili ---
    missing = False
    for name in labels:
        for sp in _SPLITS:
            if n_vid[sp][name] == 0:
                print(f"WARNING: kelas '{name}' tidak ada di split '{sp}'")
                missing = True
    if not missing:
        print("Semua kelas terwakili di train/val/test.")
    print("=" * 64)

    return passed


def _write_manifest(records, split_map, path):
    """Tulis manifest.csv: video_id,label,split,jumlah_sequence (terurut)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "label", "split", "jumlah_sequence"])
        for rec in sorted(records, key=lambda r: r["video_id"]):
            writer.writerow([
                rec["video_id"], rec["label"],
                split_map[rec["video_id"]], len(rec["seqs"]),
            ])


def build_dataset(raw_dir=None, out_dir=None):
    """Proses video → simpan dataset TER-SPLIT (level video) ke ``out_dir``.

    Split stratified per kelas (rasio & seed dari config). Satu video utuh hanya
    masuk satu split; semua sequence-nya ikut. Output gabungan lama tak ditulis.

    Returns:
        dict ``{"train": (X, y), "val": (X, y), "test": (X, y)}``.

    Raises:
        RuntimeError: bila tak ada sequence valid, atau verifikasi anti-leakage
        GAGAL (ada video lintas split).
    """
    raw_dir = raw_dir or config.RAW_DIR
    out_dir = out_dir or config.PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = config.get_labels_from_dataset(raw_dir)
    label_to_idx = {name: i for i, name in enumerate(labels)}
    logger.info("Mulai build dataset. %d kelas dari %s", len(labels), raw_dir)

    records = []     # [{"video_id", "label", "label_idx", "seqs": [seq, ...]}]
    failed = []

    # satu extractor dibuka untuk seluruh proses (lebih efisien)
    with HolisticLandmarkExtractor(static_image_mode=False) as extractor:
        for name in labels:
            class_dir = raw_dir / name
            if not class_dir.is_dir():
                logger.warning("Folder kelas tak ada: %s", class_dir)
                continue

            videos = _list_videos(class_dir)
            for video_path in tqdm(videos, desc=name, unit="vid"):
                video_id = video_path.relative_to(raw_dir).as_posix()
                frames = read_frames(video_path)
                if frames is None:
                    failed.append(video_id)
                    continue

                seq = extract_sequence(frames, extractor)        # (T,126)
                # tak ada tangan sama sekali di seluruh video → buang
                if not np.any(seq):
                    logger.warning("Tak ada tangan terdeteksi: %s", video_path)
                    failed.append(video_id)
                    continue

                seq = normalize_sequence(seq)
                seq = resample_sequence(seq, config.SEQUENCE_LENGTH)  # (30,126)

                # 1 sequence per video saat ini; list menjaga aturan "semua
                # sequence dari satu video ikut split yang sama" bila kelak >1.
                records.append({
                    "video_id": video_id,
                    "label": name,
                    "label_idx": label_to_idx[name],
                    "seqs": [seq],
                })

    if not records:
        raise RuntimeError(f"Tidak ada sequence valid dari {raw_dir}")

    # --- split di LEVEL VIDEO + verifikasi anti-leakage (cetak hasil) ---
    split_map = split_videos_stratified(records)
    passed = verify_split(records, split_map, labels)

    # --- rakit array per split (semua sequence satu video ikut splitnya) ---
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

    # --- simpan artefak (ke out_dir; nama file dari config) ---
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
    logger.info("=== Ringkasan dataset (ter-split level video) ===")
    for sp in _SPLITS:
        X, y = arrays[sp]
        logger.info("  %-5s : X=%s y=%s", sp, X.shape, y.shape)
    logger.info("Total video  : %d | video gagal : %d", len(records), len(failed))
    for fp in failed:
        logger.info("    [gagal] %s", fp)
    logger.info("Disimpan ke  : %s", out_dir)
    logger.info("label_map    : %s", out_dir / config.PROCESSED_LABEL_MAP_PATH.name)
    logger.info("manifest     : %s", manifest_path)

    if not passed:
        raise RuntimeError(
            "LEAKAGE CHECK FAIL: ada video lintas split (lihat output di atas)."
        )

    return arrays


if __name__ == "__main__":
    build_dataset()
