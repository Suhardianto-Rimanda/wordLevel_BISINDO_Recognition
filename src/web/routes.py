"""
routes.py — Blueprint Flask: halaman utama, stream video, state, audio, reset.

Endpoint:
  GET  /            → render index.html
  GET  /video_feed  → stream MJPEG (multipart) dari camera.gen_frames
  GET  /state       → kata mentah + skor + kalimat (JSON) untuk polling UI
  POST /speak       → gTTS kalimat → mp3 → {url}
  POST /reset       → kosongkan buffer kata
"""

import os
import tempfile
from pathlib import Path

from flask import (
    Blueprint, render_template, Response, jsonify, request, url_for,
)
from werkzeug.utils import secure_filename

import config
from src.web import camera as cam_mod
from src.web.tts import text_to_speech
from src.web.predict_file import predict_video_file, VideoProcessingError
from src.utils.logger import get_logger

logger = get_logger(__name__)
bp = Blueprint("main", __name__)

_IDLE_STATE = {
    "raw_word": None, "raw_score": 0.0, "fps": 0.0, "sentence": "",
    "final_sentence": "", "final_id": 0, "model_loaded": False,
}


@bp.route("/")
def index():
    """Halaman utama: video + panel teks + kontrol audio."""
    return render_template("index.html")


@bp.route("/video_feed")
def video_feed():
    """Stream MJPEG dari webcam + pipeline (dipicu tombol Mulai Kamera)."""
    camera = cam_mod.get_camera()
    if not camera.opened:
        return jsonify({"error": "Webcam tak bisa dibuka (cek perangkat kamera)."}), 503
    return Response(
        cam_mod.gen_frames(camera),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@bp.route("/state")
def state():
    """Kata mentah LSTM + skor + kalimat NLP terkini (JSON)."""
    camera = cam_mod.current_camera()
    if camera is None:
        return jsonify(_IDLE_STATE)
    return jsonify(camera.get_state())


@bp.route("/speak", methods=["POST"])
def speak():
    """Konversi kalimat → audio gTTS → URL mp3. Non-blok (thread request)."""
    data = request.get_json(silent=True) or {}
    text = (data.get("sentence") or "").strip()
    if not text:  # fallback: ambil dari state kamera
        camera = cam_mod.current_camera()
        if camera is not None:
            text = (camera.get_state().get("sentence") or "").strip()
    if not text:
        return jsonify({"error": "Kalimat kosong."}), 400

    try:
        out_path = text_to_speech(text)
    except Exception as e:  # gTTS butuh internet
        logger.warning("gTTS gagal: %s", e)
        return jsonify({"error": "Gagal generate audio (butuh internet?)."}), 503

    url = url_for("static", filename=f"audio/{out_path.name}")
    return jsonify({"url": url, "text": text})


@bp.route("/reset", methods=["POST"])
def reset():
    """Kosongkan buffer kata / kalimat (tombol Reset Model)."""
    camera = cam_mod.current_camera()
    if camera is not None:
        camera.reset()
    return jsonify({"ok": True})


@bp.route("/predict-file", methods=["POST"])
def predict_file():
    """Uji satu video upload lewat pipeline training → JSON prediksi (top-k).

    ADITIF & terpisah dari real-time (route lama tak tersentuh). Validasi tipe &
    ukuran, simpan sementara, proses, lalu HAPUS file temp. Semua error dibalas
    JSON yang bisa ditampilkan ke user — server TIDAK crash.
    """
    # --- ambil file ---
    file = request.files.get("video")
    if file is None or not file.filename:
        return jsonify({"ok": False, "error": "Tidak ada file video yang diunggah."}), 400

    # --- validasi ekstensi (allowlist; nama mentah user TIDAK dipercaya) ---
    ext = Path(secure_filename(file.filename)).suffix.lower()
    if ext not in config.UPLOAD_ALLOWED_EXTS:
        allowed = ", ".join(sorted(config.UPLOAD_ALLOWED_EXTS))
        return jsonify({
            "ok": False,
            "error": f"Tipe file tak didukung ({ext or 'tanpa ekstensi'}). Gunakan: {allowed}.",
        }), 415

    # --- validasi ukuran (cek cepat; MAX_CONTENT_LENGTH = backstop keras Werkzeug) ---
    if request.content_length and request.content_length > config.UPLOAD_MAX_BYTES:
        mb = config.UPLOAD_MAX_BYTES // (1024 * 1024)
        return jsonify({"ok": False, "error": f"File terlalu besar (maks {mb} MB)."}), 413

    # --- simpan sementara (suffix tervalidasi, BUKAN nama user) lalu proses ---
    tmp_path = None
    try:
        config.UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(suffix=ext, dir=str(config.UPLOAD_TEMP_DIR))
        os.close(fd)
        file.save(tmp_path)

        result = predict_video_file(tmp_path)
        return jsonify(result)
    except VideoProcessingError as e:
        return jsonify({"ok": False, "error": e.message}), e.status
    except Exception as e:  # jaga-jaga: jangan sampai meng-crash server
        logger.exception("Gagal memproses video upload: %s", e)
        return jsonify({
            "ok": False,
            "error": "Gagal memproses video (kesalahan tak terduga).",
        }), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)   # selalu bersihkan temp (tak menumpuk)
            except OSError:
                logger.warning("Gagal hapus file temp: %s", tmp_path)


@bp.app_errorhandler(413)
def request_too_large(_e):
    """Balas JSON (bukan halaman HTML) saat upload melewati MAX_CONTENT_LENGTH."""
    mb = config.UPLOAD_MAX_BYTES // (1024 * 1024)
    return jsonify({"ok": False, "error": f"File terlalu besar (maks {mb} MB)."}), 413
