"""
inference_engine.py — Mesin inference BISINDO real-time.

Dua lapisan terpisah:
  * ``InferenceEngine`` (PURE)  — frame masuk → (kata, skor, results) keluar.
    Tanpa webcam, tanpa jendela. Dipakai ulang oleh runner laptop & modul web.
  * ``annotate_frame`` + ``run_webcam`` (TAMPILAN) — buka webcam, gambar
    landmark/label/FPS, tampilkan jendela OpenCV.

Pipeline per frame:
    Holistic landmark → FrameBuffer(30) → [penuh] normalize → LSTM predict
    → threshold (config.PROBABILITY_THRESHOLD): di bawahnya = noise (word None).

Normalisasi memakai ``normalize_sequence`` yang SAMA dengan training
(wrist-relative + scale) agar distribusi fitur konsisten.
"""

import time
from dataclasses import dataclass

import config
from src.preprocessing.extract_landmarks import make_extractor
from src.preprocessing.build_sequences import normalize_sequence
from src.inference.frame_buffer import FrameBuffer
from src.inference.predictor import Predictor


@dataclass
class InferenceResult:
    """Hasil satu frame inference."""
    word: str | None       # kata terprediksi (None bila < threshold / buffer belum penuh)
    score: float           # probabilitas argmax (0.0 bila belum prediksi)
    results: object        # results MediaPipe Holistic (untuk gambar landmark)
    ready: bool            # True bila buffer penuh & prediksi dilakukan
    roi_bbox: tuple | None = None  # (x1,y1,x2,y2) ROI YOLO di frame penuh (None=baseline)


class InferenceEngine:
    """Logika inference murni (tanpa tampilan). Reusable laptop + web."""

    def __init__(self, model_path=None, label_map_path=None, threshold=None):
        self.threshold = (
            threshold if threshold is not None else config.PROBABILITY_THRESHOLD
        )
        self.predictor = Predictor(model_path, label_map_path)
        # Ekstraktor sesuai config.FEATURE_EXTRACTOR ("holistic" default).
        self.extractor = make_extractor(static_image_mode=False)
        self.buffer = FrameBuffer()
        self.last_word = None
        self.last_score = 0.0

        # YOLO ROI (opsional, di belakang toggle). Import lazy: baseline (OFF)
        # tak butuh ultralytics/torch ter-install.
        self.yolo = None
        if config.USE_YOLO_ROI:
            from src.inference.yolo_roi import HandROIDetector
            self.yolo = HandROIDetector()

    def process_frame(self, frame_bgr):
        """Proses satu frame BGR → InferenceResult.

        Selalu ekstrak landmark & isi buffer. Prediksi hanya saat buffer penuh.
        """
        # YOLO ROI (bila aktif): crop tangan → MediaPipe pada crop. Tak ada tangan
        # terdeteksi → fallback frame penuh (perilaku baseline, tak crash).
        target, roi_bbox = frame_bgr, None
        if self.yolo is not None:
            crop, roi_bbox = self.yolo.get_roi(frame_bgr)
            if crop is not None:
                target = crop

        vec, results = self.extractor.extract_with_results(target)
        self.buffer.append(vec)

        word, score, ready = None, 0.0, self.buffer.is_ready()
        if ready:
            seq = normalize_sequence(self.buffer.as_array())   # (30,126) sama spt training
            pred_word, prob = self.predictor.predict(seq)
            score = prob
            word = pred_word if prob >= self.threshold else None  # threshold = filter noise
            self.last_word, self.last_score = word, score

        return InferenceResult(word=word, score=score, results=results, ready=ready,
                               roi_bbox=roi_bbox)

    def reset(self):
        """Kosongkan buffer (mis. mulai sesi baru)."""
        self.buffer.clear()
        self.last_word, self.last_score = None, 0.0

    def close(self):
        """Lepas resource MediaPipe."""
        self.extractor.close()


# --------------------------------------------------------------------------- #
# LAPISAN TAMPILAN (terpisah dari engine)
# --------------------------------------------------------------------------- #
def annotate_frame(frame, result: InferenceResult, fps=None):
    """Gambar landmark + kata/score (+ FPS) ke frame. Kembalikan frame BGR.

    Reusable: laptop (imshow) maupun web (encode MJPEG).
    """
    from src.utils import visualization as viz

    if result.roi_bbox is not None:
        # YOLO aktif: gambar kotak ROI + landmark pada slice view crop.
        import cv2
        x1, y1, x2, y2 = result.roi_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        # results MediaPipe ber-koordinat relatif crop → gambar pada view
        # frame[y1:y2, x1:x2] (numpy/cv2 ROI) agar posisinya benar di frame penuh.
        sub = frame[y1:y2, x1:x2]
        if sub.size:
            viz.draw_landmarks(sub, result.results)
    else:
        viz.draw_landmarks(frame, result.results)
    viz.draw_prediction(frame, result.word, result.score)
    if fps is not None:
        viz.draw_fps(frame, fps)
    return frame


def run_webcam(source=0, model_path=None, label_map_path=None):
    """Jalankan inference real-time dari webcam (entry point laptop).

    Tekan 'q' untuk keluar. Menampilkan landmark + kata + FPS aktual.

    Catatan: frame TIDAK di-mirror sebelum ekstraksi agar konsisten dengan
    data training (video dataset tidak dicerminkan).
    """
    import cv2

    engine = InferenceEngine(model_path, label_map_path)
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        engine.close()
        raise RuntimeError(f"Webcam tak bisa dibuka (source={source}).")

    fps = 0.0
    prev = time.perf_counter()
    alpha = 0.9  # smoothing EMA FPS
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame gagal dibaca, stop.")
                break

            result = engine.process_frame(frame)

            now = time.perf_counter()
            dt = now - prev
            prev = now
            if dt > 0:
                inst = 1.0 / dt
                fps = inst if fps == 0.0 else alpha * fps + (1 - alpha) * inst

            annotate_frame(frame, result, fps)
            cv2.imshow("BISINDO Inference (tekan 'q' untuk keluar)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        engine.close()


if __name__ == "__main__":
    run_webcam()
