"""
routes.py — Blueprint Flask: halaman utama, stream video, state, audio, reset.

Endpoint:
  GET  /            → render index.html
  GET  /video_feed  → stream MJPEG (multipart) dari camera.gen_frames
  GET  /state       → kata mentah + skor + kalimat (JSON) untuk polling UI
  POST /speak       → gTTS kalimat → mp3 → {url}
  POST /reset       → kosongkan buffer kata
"""

from flask import (
    Blueprint, render_template, Response, jsonify, request, url_for,
)

from src.web import camera as cam_mod
from src.web.tts import text_to_speech
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
