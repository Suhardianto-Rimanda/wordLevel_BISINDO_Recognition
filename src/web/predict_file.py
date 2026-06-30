"""
predict_file.py — Service pengujian via UPLOAD VIDEO (fitur cadangan sidang).

MULTI-KATA via SEGMENTASI per-JEDA. Tiap isyarat dalam video dipisah berdasar
JEDA tangan (frame tanpa tangan), lalu TIAP segmen diproses PERSIS seperti data
training (``build_sequences.build_dataset`` per-video) agar konsisten dgn model:

    read_frames -> extract_sequence (T,126) -> segmentasi per-jeda tangan
      untuk tiap segmen:  normalize_sequence -> resample_sequence (30,126)
                          -> Predictor.predict -> threshold
      kumpulkan kata -> SentenceBuilder (dedup + NLP) -> kalimat

KENAPA segmentasi, bukan sliding-window 30 frame mentah: model dilatih atas SATU
isyarat penuh yang DIRESAMPLE ke 30 frame. Jendela 30 frame mentah dari video
panjang hanya FRAGMEN isyarat → prediksi salah-percaya-diri. Resample per-segmen
mengembalikan distribusi input ke kondisi training. Segmentasi per-jeda = analog
file dari idle-reset real-time (perbedaan tak terhindarkan: butuh jeda antar kata).

REUSE penuh modul yang ada (normalize/resample/predictor/SentenceBuilder). TANPA
YOLO ROI (konsisten jalur file). Tidak menyentuh pipeline real-time.

Model LSTM berat → ``Predictor`` dimuat sekali (singleton lazy, dijaga lock).
Extractor MediaPipe dibuat per-request (pola build_dataset; tanpa state bersama).
"""

import threading

import numpy as np

import config
from src.preprocessing.extract_frames import read_frames
from src.preprocessing.extract_landmarks import (
    HolisticLandmarkExtractor,
    extract_sequence,
)
from src.preprocessing.build_sequences import normalize_sequence, resample_sequence
from src.nlp.sentence_builder import SentenceBuilder
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VideoProcessingError(Exception):
    """Error terkontrol saat memproses video upload (pesan aman utk user).

    ``status`` = kode HTTP yang sebaiknya dibalas route (default 422).
    """

    def __init__(self, message, status=422):
        super().__init__(message)
        self.message = message
        self.status = status


# --------------------------------------------------------------------------- #
# Predictor singleton (lazy) — model berat, muat sekali, prediksi di-serialize.
# --------------------------------------------------------------------------- #
_predictor = None
_predictor_lock = threading.Lock()


def _get_predictor():
    """Kembalikan singleton ``Predictor`` (muat model + label_map sekali).

    Import ``Predictor`` lokal: TensorFlow berat, hanya dibutuhkan saat fitur ini
    benar dipakai (tak memengaruhi import app / pipeline real-time).
    """
    global _predictor
    if _predictor is None:
        with _predictor_lock:
            if _predictor is None:
                from src.inference.predictor import Predictor
                try:
                    _predictor = Predictor()  # default → config.WEIGHTS_PATH + LABEL_MAP_PATH
                except FileNotFoundError as e:
                    logger.warning("Bobot model tak ada untuk uji file: %s", e)
                    raise VideoProcessingError(
                        "Model belum tersedia (bobot LSTM belum ada di folder models/).",
                        status=503,
                    ) from e
    return _predictor


def _hand_present_mask(raw):
    """Mask (T,) True bila frame punya tangan (vektor tak semua nol).

    Dasar sama dgn deteksi 'tangan ada' di normalize_sequence (frame all-zero =
    tangan tak terdeteksi). Dipakai memisah isyarat lewat jeda.
    """
    return np.any(raw != 0, axis=1)


def _segment_signs(present, min_gap, min_len):
    """Pisah deret jadi segmen isyarat berdasar JEDA (run frame tanpa tangan).

    - Celah tanpa-tangan PENDEK (< ``min_gap``) di tengah isyarat diisi (tangan
      sempat tak terdeteksi sesaat → tetap satu isyarat).
    - Segmen aktif valid bila panjang >= ``min_len`` (buang kedipan noise).

    Returns: list (start, end) indeks frame, end eksklusif.
    """
    T = len(present)
    filled = present.copy()

    # isi gap pendek yang DIAPIT frame aktif (bukan di tepi video)
    i = 0
    while i < T:
        if not filled[i]:
            j = i
            while j < T and not filled[j]:
                j += 1
            if i > 0 and j < T and (j - i) < min_gap:
                filled[i:j] = True
            i = j
        else:
            i += 1

    # ambil run aktif sebagai segmen
    segs = []
    i = 0
    while i < T:
        if filled[i]:
            j = i
            while j < T and filled[j]:
                j += 1
            if (j - i) >= min_len:
                segs.append((i, j))
            i = j
        else:
            i += 1
    return segs


def predict_video_file(video_path, k=3):
    """Proses satu file video → MULTI-KATA (segmentasi per-jeda) + kalimat NLP.

    Returns:
        dict superset: {"ok","words","sentence","frames_processed","segments",
        "words_detail","topk","word","score","frames","model_loaded","note"}.

    Raises:
        VideoProcessingError: video tak terbaca / tak ada tangan / model tak ada.
    """
    predictor = _get_predictor()  # bisa raise VideoProcessingError(503) bila model absen

    # 1) baca seluruh frame
    frames = read_frames(video_path)
    if frames is None:
        raise VideoProcessingError(
            "Video tak terbaca atau kosong. Pastikan file video valid (.mp4/.avi/.mov/.mkv)."
        )
    n_frames = len(frames)

    # 2) ekstraksi landmark mentah (T,126) SEKALI — Holistic seperti build_dataset
    #    (static_image_mode=False, tanpa YOLO). Per-request → tanpa state bersama.
    with HolisticLandmarkExtractor(static_image_mode=False) as extractor:
        raw = extract_sequence(frames, extractor)   # (T, 126) BELUM dinormalisasi

    # 3) tak ada tangan sama sekali → tolak (sama build_dataset)
    if not np.any(raw):
        raise VideoProcessingError(
            "Tak ada tangan terdeteksi di video. Pastikan tangan terlihat jelas & pencahayaan cukup."
        )

    # 4) segmentasi per-jeda tangan (analog idle-reset real-time)
    present = _hand_present_mask(raw)
    segs = _segment_signs(
        present, config.SEGMENT_MIN_GAP_FRAMES, config.SEGMENT_MIN_LEN_FRAMES
    )

    builder = SentenceBuilder()      # dedup beruntun + MAX_WORDS (NLP yang SAMA)
    words_detail = []
    note = None

    # Semua akses model di-serialize (1 user sidang) → satu lock utk seluruh request.
    with _predictor_lock:
        # 5a) top-k whole-video (kompat tampilan satu-kata): normalize → resample 30 → predict_topk.
        seq30 = resample_sequence(normalize_sequence(raw), config.SEQUENCE_LENGTH)
        topk = predictor.predict_topk(seq30, k=k)

        # 5b) TIAP segmen = 1 isyarat → normalize → resample 30 (PERSIS training) → predict.
        for (a, b) in segs:
            seq = resample_sequence(
                normalize_sequence(raw[a:b]), config.SEQUENCE_LENGTH
            )
            word, prob = predictor.predict(seq)
            if prob >= config.PROBABILITY_THRESHOLD:   # threshold yang SAMA dgn real-time
                builder.add_word(word)                 # dedup beruntun (NLP yang SAMA)
                words_detail.append({
                    "word": word,
                    "score": round(float(prob), 4),
                    "frame_start": int(a),
                    "frame_end": int(b),
                })

    words = list(builder.words)
    sentence = builder.get_sentence()

    # catatan informatif (200, bukan error)
    if not segs:
        note = "Tak ada segmen isyarat terdeteksi (tangan tak cukup jelas/stabil)."
    elif not words:
        note = "Tak ada kata melewati ambang keyakinan (coba peragakan lebih jelas/stabil)."
    elif len(segs) == 1:
        note = "Hanya 1 isyarat terdeteksi (tak ada jeda antar isyarat di video)."

    topk_list = [{"word": w, "score": round(float(s), 4)} for w, s in topk]

    # field back-compat "word"/"score": kata pertama terdeteksi, atau top-1 whole-video.
    if words:
        best_word = words[0]
        best_score = words_detail[0]["score"] if words_detail else round(float(topk[0][1]), 4)
    else:
        best_word = topk[0][0]
        best_score = round(float(topk[0][1]), 4)

    logger.info(
        "Uji file: %d frame, %d segmen → words=%s | sentence=%r",
        n_frames, len(segs), words, sentence,
    )
    return {
        "ok": True,
        "words": words,                  # daftar kata berurutan (fitur baru)
        "sentence": sentence,            # kalimat hasil NLP rule-based
        "frames_processed": n_frames,
        "segments": len(segs),           # jumlah segmen isyarat terdeteksi
        "words_detail": words_detail,    # [{word, score, frame_start, frame_end}]
        "topk": topk_list,               # kandidat whole-video (kompat single-word)
        "word": best_word,               # back-compat
        "score": best_score,             # back-compat
        "frames": n_frames,              # back-compat (alias frames_processed)
        "model_loaded": True,
        "note": note,                    # null bila tak ada catatan khusus
    }
