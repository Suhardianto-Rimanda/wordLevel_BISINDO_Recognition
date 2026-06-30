"""
assemble_dataset.py — Susun file landmark PER-SAMPLE → dataset X_/y_ siap-training.

Pasangan dari extract_landmarks_persample.py: dari folder per-sample (.npy/video +
manifest_persample.csv + label_map.json), rakit X_<split>.npy / y_<split>.npy sesuai
SKENARIO yang dipilih — TANPA mengekstrak ulang (cepat, hanya menyusun file).

Pembagian SELALU di LEVEL VIDEO (anti-leakage) & ber-seed (config.RANDOM_SEED).
Logika split di-IMPORT dari extract_landmarks_split.py (tidak ditulis ulang).
Format output SAMA dengan dataset lama: X (n, SEQUENCE_LENGTH, 126) float32,
y (n,) int64, label_map.json {index: kata}.

ADITIF — tidak menyentuh build_sequences.py / extract_landmarks_split.py / training.

MODE:
  auto   — split otomatis stratified per kelas (seperti extract_landmarks_split):
           --split train,val,test --ratio 70:15:15  (kombinasi & rasio bebas, jumlah=100)
  signer — pisah berdasar SIGNER (signer-independent):
           --trainval-signers Signer01,Signer02 --test-signers Signer03 [--val-ratio 0.2]
           (trainval & test harus DISJOIN; --val-ratio memecah train/val dari trainval)

Jalankan:
    python scripts/assemble_dataset.py --mode auto --split train,val,test --ratio 70:15:15
    python scripts/assemble_dataset.py --mode signer --trainval-signers Signer01 --test-signers Signer02
    python scripts/assemble_dataset.py --mode signer --trainval-signers Signer01,Signer02 \
        --test-signers Signer03 --val-ratio 0.2 --out-dir data/processed_signerindep

Opsi:
    --mode {auto,signer}        Skenario pembagian (default: auto).
    --src-dir PATH              Folder hasil Skrip 1. Default config.PROCESSED_DIR/"landmarks_persample".
    --out-dir PATH              Folder output X_/y_. Default config.PROCESSED_DIR (konfirmasi bila menimpa).
    --seed INT                  Seed shuffle. Default config.RANDOM_SEED.
    --yes                       Lewati konfirmasi (mis. saat menimpa).
  [mode auto]
    --split LIST                train,val,test (urutan/kombinasi bebas).
    --ratio LIST                rasio ':' sesuai --split, jumlah=100 (mis. 70:15:15).
  [mode signer]
    --trainval-signers LIST     signer untuk train(+val), pisah koma.
    --test-signers LIST         signer untuk test, pisah koma.
    --val-ratio FLOAT           pecah val dari trainval (0..1, default 0 = tanpa val).
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# root proyek + folder scripts/ ke sys.path (import config, src, & sibling)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
# reuse logika split LEVEL VIDEO + util parsing dari skrip yang sudah ada
from extract_landmarks_split import (
    split_videos_stratified_general,
    _parse_splits,
    _parse_ratios,
    _split_paths,
    _confirm,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

_MANIFEST_NAME = "manifest_persample.csv"
_ALL_SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# MUAT SUMBER (manifest + label_map)
# --------------------------------------------------------------------------- #
def _load_labels(src_dir):
    """Urutan label dari label_map.json src-dir (konsisten indeks dataset lama).

    Returns list[str] (index→kata) atau None bila file tak ada.
    """
    p = src_dir / config.PROCESSED_LABEL_MAP_PATH.name
    if not p.exists():
        return None
    m = json.loads(p.read_text(encoding="utf-8"))
    return [m[str(i)] for i in range(len(m))]


def _load_records(src_dir):
    """Baca manifest_persample.csv → list record dict.

    Tiap record: {npy_path, video_id, label, signer}.
    """
    mp = src_dir / _MANIFEST_NAME
    if not mp.exists():
        raise FileNotFoundError(
            f"Manifest tak ditemukan: {mp}\n"
            "Jalankan dulu scripts/extract_landmarks_persample.py untuk src-dir ini."
        )
    records = []
    with open(mp, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append({
                "npy_path": row["npy_path"],
                "video_id": row["video_id"],
                "label": row["label"],
                "signer": (row.get("signer") or "").strip(),
            })
    if not records:
        raise RuntimeError(f"Manifest kosong: {mp}")
    return records


# --------------------------------------------------------------------------- #
# PEMETAAN SPLIT
# --------------------------------------------------------------------------- #
def _split_map_auto(records, splits, ratios, seed):
    """Mode auto: split_map {video_id: split} stratified per kelas (logika reuse)."""
    if len(splits) == 1:
        return {rec["video_id"]: splits[0] for rec in records}
    return split_videos_stratified_general(records, splits, ratios, seed=seed)


def _norm_set(raw):
    """'Signer01,Signer02' → set lowercase (pencocokan tak peka huruf besar/kecil)."""
    return {p.strip().lower() for p in raw.split(",") if p.strip()}


def _split_map_signer(records, trainval_signers, test_signers, val_ratio, seed):
    """Mode signer: peta split berbasis signer (signer-independent).

    Returns (split_map, ignored_count). trainval & test DISJOIN (divalidasi pemanggil).
    """
    tv_set = _norm_set(trainval_signers)
    te_set = _norm_set(test_signers)

    tv_records = [r for r in records if r["signer"].lower() in tv_set]
    te_records = [r for r in records if r["signer"].lower() in te_set]
    used = {r["video_id"] for r in tv_records} | {r["video_id"] for r in te_records}
    ignored = sum(1 for r in records if r["video_id"] not in used)

    split_map = {r["video_id"]: "test" for r in te_records}

    if val_ratio and val_ratio > 0 and tv_records:
        sub = split_videos_stratified_general(
            tv_records, ["train", "val"], [1.0 - val_ratio, val_ratio], seed=seed
        )
        split_map.update(sub)
    else:
        for r in tv_records:
            split_map[r["video_id"]] = "train"

    return split_map, ignored


# --------------------------------------------------------------------------- #
# MODE CUSTOM — porsi train:val:test PER SIGNER (dialog interaktif / --config)
# Helper TERPISAH; TIDAK menyentuh split_videos_stratified_general (mode auto/signer).
# --------------------------------------------------------------------------- #
_CUSTOM_SPLITS = ("train", "val", "test")


def _detect_signers(records):
    """Daftar (signer, jumlah_video) terurut jumlah desc lalu nama.

    Signer kosong (tanpa pola SignerXX) dikelompokkan key "" (tampil '(tanpa-signer)').
    """
    cnt = Counter(r["signer"] for r in records)   # records = 1 per video
    return sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))


def _signer_label(signer):
    return signer if signer else "(tanpa-signer)"


def _validate_triplet(nums):
    """Validasi [train, val, test]: 3 angka, non-negatif, jumlah TEPAT 100.

    Returns list[float] persen. Raise ValueError dgn pesan jelas bila salah.
    """
    if len(nums) != 3:
        raise ValueError(f"Butuh TEPAT 3 angka (train:val:test), dapat {len(nums)}.")
    try:
        vals = [float(x) for x in nums]
    except (TypeError, ValueError):
        raise ValueError(f"Porsi harus angka, dapat: {nums}.")
    if any(v < 0 for v in vals):
        raise ValueError("Porsi tak boleh negatif (0 diizinkan).")
    total = sum(vals)
    if abs(total - 100.0) > 1e-6:
        raise ValueError(f"Jumlah harus TEPAT 100, sekarang {total:g}.")
    return vals


def _parse_custom_triplet(raw):
    """'70:15:15' -> [70.0, 15.0, 15.0] (persen, 0 diizinkan, jumlah=100)."""
    parts = [p.strip() for p in raw.split(":") if p.strip() != ""]
    return _validate_triplet(parts)


def _ask_custom_ratios(signers):
    """Tanya porsi train:val:test tiap signer (re-ask bila salah).

    Args: signers = list key signer (urutan tampil).
    Returns: dict {signer: [t, v, te]} persen.
    """
    ratios = {}
    print("\nMasukkan porsi train:val:test (jumlah=100, 0 diizinkan) untuk tiap signer.")
    for s in signers:
        while True:
            raw = input(f"  Porsi untuk {_signer_label(s)} (mis. 70:15:15) = ")
            try:
                ratios[s] = _parse_custom_triplet(raw)
                break
            except ValueError as e:
                print(f"    {e} Coba lagi.")
    return ratios


def _load_custom_config(path, present_signers):
    """Baca JSON {signer: [t,v,te]} -> dict tervalidasi (pencocokan tak peka huruf)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config harus objek JSON {signer: [train,val,test]}.")
    lut = {s.lower(): s for s in present_signers}   # nama kanonik signer yang ADA
    out = {}
    for key, nums in data.items():
        canon = lut.get(str(key).lower())
        if canon is None:
            raise ValueError(f"Signer '{key}' di config tak ada di dataset.")
        out[canon] = _validate_triplet(nums)
    return out


def _save_custom_config(path, ratios):
    """Tulis dict {signer: [t,v,te]} ke JSON (untuk reproduksi/ulang)."""
    Path(path).write_text(
        json.dumps(dict(ratios), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _allocate_counts(n, fracs):
    """Bagi n item ke 3 split menurut fracs (jumlah=1.0), KONSERVASI total.

    floor tiap split + sisa pembulatan dibagikan ke split ber-frac>0 (urut bagian
    pecahan desc). Split ber-frac 0 TETAP 0. sum(counts)==n persis -> tak ada video
    hilang/dobel akibat pembulatan.
    """
    raw = [n * f for f in fracs]
    counts = [int(x) for x in raw]                 # floor (x >= 0)
    rem = n - sum(counts)
    frac_part = [raw[i] - counts[i] for i in range(3)]
    eligible = sorted(
        [i for i in range(3) if fracs[i] > 0],
        key=lambda i: frac_part[i], reverse=True,
    )
    k = 0
    while rem > 0 and eligible:
        counts[eligible[k % len(eligible)]] += 1
        rem -= 1
        k += 1
    return counts


def _split_signer_by_ratio(signer_records, triplet_pct, seed):
    """Split SATU signer -> {video_id: split}: per kelas, level video, exact-ratio.

    Konsisten dgn pendekatan lama (per kelas: urut deterministik -> shuffle ber-seed
    -> iris) TAPI menghormati porsi 0 (tanpa jaminan min-1) — khusus mode custom.
    """
    fracs = [p / 100.0 for p in triplet_pct]
    by_class = defaultdict(list)
    for r in signer_records:
        by_class[r["label"]].append(r["video_id"])

    rng = random.Random(seed)
    split_map = {}
    for label in sorted(by_class):
        vids = sorted(by_class[label])             # deterministik sebelum shuffle
        rng.shuffle(vids)
        counts = _allocate_counts(len(vids), fracs)
        idx = 0
        for sp, c in zip(_CUSTOM_SPLITS, counts):
            for vid in vids[idx:idx + c]:
                split_map[vid] = sp
            idx += c
    return split_map


def _split_map_custom(records, ratios_by_signer, seed):
    """Gabung split tiap signer -> split_map global {video_id: split}."""
    by_signer = defaultdict(list)
    for r in records:
        by_signer[r["signer"]].append(r)

    split_map = {}
    for signer, triplet in ratios_by_signer.items():
        split_map.update(_split_signer_by_ratio(by_signer.get(signer, []), triplet, seed))
    return split_map


def _print_custom_plan(signers_counts, ratios_by_signer):
    """Tabel rencana: signer x (train/val/test) persen + perkiraan jumlah video."""
    counts = dict(signers_counts)
    print("=" * 72)
    print(f"{'signer':<18}{'n':>6}   {'train%':>8}{'val%':>8}{'test%':>8}   (est vid t/v/te)")
    print("-" * 72)
    for signer, _ in signers_counts:
        t, v, te = ratios_by_signer[signer]
        n = counts[signer]
        est = f"{round(n*t/100)}/{round(n*v/100)}/{round(n*te/100)}"
        print(f"{_signer_label(signer):<18}{n:>6}   {t:>8g}{v:>8g}{te:>8g}   {est}")
    print("=" * 72)


def _print_signer_dependent_notes(ratios_by_signer):
    """Catatan metodologis: signer dgn train/val>0 DAN test>0 -> signer-dependent."""
    for s, (t, v, te) in ratios_by_signer.items():
        if (t > 0 or v > 0) and te > 0:
            print(f"CATATAN: {_signer_label(s)} signer-DEPENDENT - sebagian datanya "
                  f"dilatih DAN diuji (bukan signer-independent). (catatan saja)")


# --------------------------------------------------------------------------- #
# RAKIT ARRAY
# --------------------------------------------------------------------------- #
def _assemble(records, split_map, label_to_idx, src_dir):
    """Muat .npy per record → {split: (X, y)} + records_by_split (utk ringkasan)."""
    buckets = defaultdict(lambda: ([], []))
    by_split = defaultdict(list)
    expected = (config.SEQUENCE_LENGTH, config.FEATURE_DIM)

    for rec in records:
        sp = split_map.get(rec["video_id"])
        if sp is None:
            continue  # tak masuk split mana pun (mode signer: signer di luar daftar)
        arr = np.load(src_dir / rec["npy_path"])
        if arr.shape != expected:
            raise RuntimeError(
                f"Bentuk .npy tak sesuai {expected}: {rec['npy_path']} -> {arr.shape}"
            )
        Xs, ys = buckets[sp]
        Xs.append(arr.astype(np.float32))
        ys.append(label_to_idx[rec["label"]])
        by_split[sp].append(rec)

    arrays = {
        sp: (np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.int64))
        for sp, (Xs, ys) in buckets.items()
    }
    return arrays, by_split


# --------------------------------------------------------------------------- #
# RINGKASAN + ANTI-LEAKAGE
# --------------------------------------------------------------------------- #
def _print_summary(by_split, labels):
    """Cetak jumlah/ kelas/ signer per split + verifikasi anti-leakage."""
    splits = [sp for sp in _ALL_SPLITS if sp in by_split]

    # anti-leakage (irisan video_id antar split = kosong)
    vids = {sp: {r["video_id"] for r in by_split[sp]} for sp in splits}
    leak = set()
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            leak |= vids[a] & vids[b]

    print("=" * 72)
    print("LEAKAGE CHECK (video lintas split):", "PASS" if not leak else f"FAIL ({len(leak)})")
    for v in sorted(leak):
        print(f"   [BOCOR] {v}")

    # tabel jumlah per kelas per split
    print("-" * 72)
    header = f"{'kelas':<22}" + "".join(f"{sp:>10}" for sp in splits)
    print(header)
    print("-" * 72)
    cnt = {sp: Counter(r["label"] for r in by_split[sp]) for sp in splits}
    missing = []
    for name in labels:
        row = f"{name:<22}" + "".join(f"{cnt[sp][name]:>10}" for sp in splits)
        print(row)
        for sp in splits:
            if cnt[sp][name] == 0:
                missing.append((name, sp))
    print("-" * 72)
    total = f"{'TOTAL':<22}" + "".join(f"{sum(cnt[sp].values()):>10}" for sp in splits)
    print(total)

    # komposisi signer per split
    print("-" * 72)
    for sp in splits:
        sig = Counter((r["signer"] or "-") for r in by_split[sp])
        comp = ", ".join(f"{s}={n}" for s, n in sorted(sig.items()))
        print(f"signer[{sp}] : {comp}")

    # kelas hilang di suatu split
    if missing:
        print("-" * 72)
        for name, sp in missing:
            print(f"WARNING: kelas '{name}' tidak ada di split '{sp}'")
    print("=" * 72)


# --------------------------------------------------------------------------- #
# SIMPAN
# --------------------------------------------------------------------------- #
def _save(arrays, labels, out_dir, assume_yes):
    """Tulis X_<split>.npy/y_<split>.npy + label_map.json (konfirmasi bila menimpa)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = _split_paths(out_dir)
    label_map_path = out_dir / config.PROCESSED_LABEL_MAP_PATH.name

    # cek file yang akan tertimpa
    targets = []
    for sp in arrays:
        targets += [paths[sp][0], paths[sp][1]]
    targets.append(label_map_path)
    existing = [p for p in targets if p.exists()]
    if existing and not assume_yes:
        print(f"\n{len(existing)} file akan DITIMPA di {out_dir}:")
        for p in existing:
            print(f"   - {p.name}")
        if not _confirm("Lanjut menimpa?"):
            print("Dibatalkan. Tidak ada file yang ditulis.")
            return False

    for sp, (X, y) in arrays.items():
        xp, yp = paths[sp]
        np.save(xp, X)
        np.save(yp, y)
        logger.info("  %-5s : X=%s y=%s -> %s", sp, X.shape, y.shape, xp.name)

    idx_to_label = {i: name for i, name in enumerate(labels)}
    label_map_path.write_text(
        json.dumps(idx_to_label, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("label_map : %s", label_map_path)
    return True


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Susun landmark per-sample -> X_/y_ dataset (mode auto / signer / custom).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=("auto", "signer", "custom"), default="auto",
                        help="Skenario pembagian (default: auto). 'custom' = porsi "
                             "train:val:test per signer (interaktif / --config).")
    parser.add_argument("--src-dir", default=None,
                        help="Folder hasil Skrip 1. Default config.PROCESSED_DIR/'landmarks_persample'.")
    parser.add_argument("--out-dir", default=None,
                        help="Folder output X_/y_. Default config.PROCESSED_DIR.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed shuffle. Default config.RANDOM_SEED.")
    parser.add_argument("--yes", action="store_true", help="Lewati konfirmasi (timpa).")
    # mode auto
    parser.add_argument("--split", default=None, help="[auto] train,val,test (kombinasi bebas).")
    parser.add_argument("--ratio", default=None, help="[auto] rasio ':' sesuai --split, jumlah=100.")
    # mode signer
    parser.add_argument("--trainval-signers", default=None,
                        help="[signer] signer untuk train(+val), pisah koma.")
    parser.add_argument("--test-signers", default=None,
                        help="[signer] signer untuk test, pisah koma.")
    parser.add_argument("--val-ratio", type=float, default=0.0,
                        help="[signer] pecah val dari trainval (0..1, default 0).")
    # mode custom
    parser.add_argument("--config", default=None,
                        help="[custom] file JSON {signer:[train,val,test]} untuk "
                             "melewati dialog interaktif (reproduksi).")
    args = parser.parse_args()

    src_dir = Path(args.src_dir) if args.src_dir else (config.PROCESSED_DIR / "landmarks_persample")
    out_dir = Path(args.out_dir) if args.out_dir else config.PROCESSED_DIR
    seed = config.RANDOM_SEED if args.seed is None else args.seed

    # --- muat sumber ---
    records = _load_records(src_dir)
    labels = _load_labels(src_dir)
    if labels is None:
        labels = sorted({r["label"] for r in records})
        logger.warning("label_map.json tak ada di src-dir; urutan label dari manifest (terurut).")
    label_to_idx = {name: i for i, name in enumerate(labels)}
    # cek label asing
    unknown = {r["label"] for r in records} - set(label_to_idx)
    if unknown:
        parser.error(f"Label di manifest tak ada di label_map: {sorted(unknown)}")

    # --- bangun split_map sesuai mode ---
    if args.mode == "auto":
        if not args.split:
            parser.error("--mode auto butuh --split (mis. train,val,test).")
        try:
            splits = _parse_splits(args.split)
            if len(splits) > 1:
                if not args.ratio:
                    parser.error("--ratio wajib bila --split > 1 (mis. 70:15:15).")
                ratios = _parse_ratios(args.ratio, len(splits))
            else:
                ratios = [1.0]
        except ValueError as e:
            parser.error(str(e))
        split_map = _split_map_auto(records, splits, ratios, seed)
        print(f"Mode auto | split={splits} ratio={[round(r,3) for r in ratios]} seed={seed}")
    elif args.mode == "signer":
        if not args.trainval_signers or not args.test_signers:
            parser.error("--mode signer butuh --trainval-signers DAN --test-signers.")
        tv_set, te_set = _norm_set(args.trainval_signers), _norm_set(args.test_signers)
        overlap = tv_set & te_set
        if overlap:
            parser.error(f"Signer tumpang-tindih trainval & test: {sorted(overlap)} (harus disjoin).")
        if not (0.0 <= args.val_ratio < 1.0):
            parser.error("--val-ratio harus di rentang [0, 1).")
        present = {r["signer"].lower() for r in records}
        for s in (tv_set | te_set):
            if s not in present:
                logger.warning("Signer '%s' tak ditemukan di manifest.", s)
        split_map, ignored = _split_map_signer(
            records, args.trainval_signers, args.test_signers, args.val_ratio, seed
        )
        print(f"Mode signer | trainval={sorted(tv_set)} test={sorted(te_set)} "
              f"val_ratio={args.val_ratio} seed={seed} | diabaikan={ignored} video")

    else:  # custom — porsi train:val:test PER SIGNER
        signers_counts = _detect_signers(records)
        names = ", ".join(f"{_signer_label(s)} ({n} video)" for s, n in signers_counts)
        print(f"Terdeteksi {len(signers_counts)} signer: {names}")

        if args.config:
            try:
                ratios_by_signer = _load_custom_config(
                    args.config, [s for s, _ in signers_counts]
                )
            except (ValueError, OSError, json.JSONDecodeError) as e:
                parser.error(f"Gagal baca --config: {e}")
        else:
            ratios_by_signer = _ask_custom_ratios([s for s, _ in signers_counts])

        # tiap signer terdeteksi WAJIB punya porsi
        missing = [s for s, _ in signers_counts if s not in ratios_by_signer]
        if missing:
            parser.error(f"Porsi belum diisi utk signer: {[_signer_label(s) for s in missing]}")

        # tawarkan simpan config (hanya jalur interaktif, bukan saat --config/--yes)
        if not args.config and not args.yes:
            sp = input("\nSimpan konfigurasi rasio ke file? (kosongkan = tidak) path = ").strip()
            if sp:
                _save_custom_config(sp, ratios_by_signer)
                print(f"Konfigurasi disimpan: {sp}")

        _print_custom_plan(signers_counts, ratios_by_signer)
        _print_signer_dependent_notes(ratios_by_signer)
        print(f"seed={seed}")

        if not args.yes and not _confirm("Lanjut memproses?"):
            print("Dibatalkan.")
            return

        split_map = _split_map_custom(records, ratios_by_signer, seed)

    # --- rakit + ringkasan + simpan ---
    arrays, by_split = _assemble(records, split_map, label_to_idx, src_dir)
    if not arrays:
        logger.error("Tak ada sample yang masuk split mana pun. Periksa argumen/signer.")
        return
    _print_summary(by_split, labels)

    print(f"\nOutput -> {out_dir}")
    for sp, (X, _) in arrays.items():
        print(f"  {sp:<5} : {X.shape}")
    if _save(arrays, labels, out_dir, args.yes):
        logger.info("Selesai. Dataset tersimpan di %s", out_dir)


if __name__ == "__main__":
    main()
