"""
yolo_roi.py — Deteksi tangan + ROI cropping (Tahap B) SEBELUM MediaPipe.

YOLO di sini HANYA deteksi + crop, BUKAN klasifikasi (klasifikasi tetap LSTM).
Modul terpisah & opsional: hanya dipakai bila ``config.USE_YOLO_ROI`` True.

PENTING (parity LSTM): crop yang diumpan ke MediaPipe dibuat ber-ASPEK SAMA dengan
frame penuh (``FRAME_WIDTH:FRAME_HEIGHT``). MediaPipe menormalkan x oleh lebar & y
oleh tinggi gambar secara terpisah; training memakai frame penuh 4:3. Crop ber-aspek
beda akan menyuntik distorsi anisotropik x/y yang TAK bisa dibatalkan normalisasi
wrist-relative+scale → prediksi melenceng. Maka union bbox semua tangan diperluas ke
aspek frame + margin, lalu di-clamp.

Import ultralytics/torch sengaja LAZY (di __init__) supaya baseline (toggle OFF)
tetap jalan tanpa kedua dependency itu ter-install.
"""

from pathlib import Path

import numpy as np

import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _resolve_device(requested):
    """Resolusi device: hormati ``requested`` tapi auto-fallback ke 'cpu' bila CUDA
    tak tersedia. TAK pernah hardcode 'cuda' → project tetap jalan tanpa GPU.
    """
    if str(requested).lower().startswith("cpu"):
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return requested
        logger.warning("YOLO_DEVICE='%s' diminta tapi CUDA tak tersedia → fallback 'cpu'.",
                       requested)
    except Exception as e:  # torch belum ter-install / error import
        logger.warning("torch tak bisa cek CUDA (%s) → fallback 'cpu'.", e)
    return "cpu"


class HandROIDetector:
    """Deteksi tangan via YOLOv8 → kembalikan crop ROI (aspek frame) + bbox.

    Pakai:
        det = HandROIDetector()
        crop, bbox = det.get_roi(frame_bgr)   # (None, None) bila tak ada tangan
    """

    def __init__(self, model_path=None, device=None, conf=None, imgsz=None,
                 margin=None):
        model_path = Path(model_path or config.YOLO_MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Bobot YOLO tak ditemukan: {model_path}\n"
                "File ini hasil Tahap A (training YOLOv8 deteksi tangan di Colab/GPU).\n"
                "Ekspor best.pt dari training lalu salin ke models/yolo_hand.pt.\n"
                "Atau set config.USE_YOLO_ROI = False untuk pakai baseline MediaPipe-only."
            )

        self.conf = config.YOLO_CONF if conf is None else conf
        self.imgsz = config.YOLO_IMGSZ if imgsz is None else imgsz
        self.margin = config.YOLO_ROI_MARGIN if margin is None else margin
        # aspek target crop = aspek frame penuh (jaga parity normalisasi LSTM)
        self.aspect = config.FRAME_WIDTH / config.FRAME_HEIGHT

        self.device = _resolve_device(device or config.YOLO_DEVICE)

        # import lazy: ultralytics berat & opsional (baseline tak butuh).
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics belum ter-install (dibutuhkan saat USE_YOLO_ROI=True).\n"
                "Install (jaga numpy<2 demi TF/MediaPipe): pip install ultralytics\n"
                "torch GPU opsional: pip install torch torchvision "
                "--index-url https://download.pytorch.org/whl/cu121"
            ) from e

        self.model = YOLO(str(model_path))
        logger.info("YOLO ROI siap | model=%s device=%s conf=%.2f imgsz=%d",
                    model_path.name, self.device, self.conf, self.imgsz)

    def _expand_to_aspect(self, x1, y1, x2, y2, W, H):
        """Perluas bbox (+margin) ke aspek frame, lalu clamp ke (W,H).

        Kembalikan int (x1,y1,x2,y2) valid di dalam frame.
        """
        bw, bh = x2 - x1, y2 - y1
        cx, cy = x1 + bw / 2.0, y1 + bh / 2.0

        # tambah margin di kedua sisi
        bw *= (1.0 + 2.0 * self.margin)
        bh *= (1.0 + 2.0 * self.margin)

        # paksa ke aspek frame (W/H): perlebar dimensi yang kurang
        if bw / bh < self.aspect:
            bw = bh * self.aspect
        else:
            bh = bw / self.aspect

        nx1 = int(round(cx - bw / 2.0))
        ny1 = int(round(cy - bh / 2.0))
        nx2 = int(round(cx + bw / 2.0))
        ny2 = int(round(cy + bh / 2.0))

        # clamp ke batas frame (aspek bisa sedikit bergeser di tepi — diterima,
        # jauh lebih kecil distorsinya dibanding crop ketat sembarang aspek)
        nx1 = max(0, min(nx1, W - 1))
        ny1 = max(0, min(ny1, H - 1))
        nx2 = max(nx1 + 1, min(nx2, W))
        ny2 = max(ny1 + 1, min(ny2, H))
        return nx1, ny1, nx2, ny2

    def get_roi(self, frame_bgr):
        """Frame BGR → (crop_bgr, bbox) atau (None, None) bila tak ada tangan.

        bbox = (x1,y1,x2,y2) di koordinat frame penuh (untuk overlay visualisasi).
        Fallback aman: tak ada deteksi → (None, None); pemanggil pakai frame penuh
        sehingga tak pernah lebih buruk dari baseline & tak crash.
        """
        H, W = frame_bgr.shape[:2]
        try:
            res = self.model.predict(
                frame_bgr, conf=self.conf, imgsz=self.imgsz,
                device=self.device, verbose=False,
            )
        except Exception as e:
            logger.warning("YOLO predict gagal (%s) → fallback frame penuh.", e)
            return None, None

        if not res:
            return None, None
        boxes = res[0].boxes
        if boxes is None or len(boxes) == 0:
            return None, None

        # union bbox semua tangan (BISINDO 2 tangan → ROI memuat keduanya supaya
        # Holistic tetap bisa pisah kiri/kanan).
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
        x1 = float(xyxy[:, 0].min())
        y1 = float(xyxy[:, 1].min())
        x2 = float(xyxy[:, 2].max())
        y2 = float(xyxy[:, 3].max())

        bx1, by1, bx2, by2 = self._expand_to_aspect(x1, y1, x2, y2, W, H)
        crop = frame_bgr[by1:by2, bx1:bx2]
        if crop.size == 0:
            return None, None
        return crop, (bx1, by1, bx2, by2)
