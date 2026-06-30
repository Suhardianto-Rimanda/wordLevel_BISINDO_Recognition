"""
app.py — Entry point Flask aplikasi penerjemah BISINDO real-time.

Registrasi blueprint web + jalankan server threaded (gTTS di /speak berjalan di
thread request terpisah → tak memblokir thread streaming video).
"""

import atexit

from flask import Flask

import config
from src.web.routes import bp
from src.web.camera import release_camera


def create_app():
    """Application factory Flask."""
    app = Flask(
        __name__,
        template_folder="src/web/templates",
        static_folder="src/web/static",
    )
    # pastikan folder output audio gTTS ada
    config.TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # folder temp upload video (fitur pengujian file) + batas ukuran request
    config.UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    app.config["MAX_CONTENT_LENGTH"] = config.UPLOAD_MAX_BYTES

    app.register_blueprint(bp)

    # lepas webcam saat proses berhenti
    atexit.register(release_camera)
    return app


if __name__ == "__main__":
    app = create_app()
    # threaded=True: /speak (gTTS) tak blokir stream.
    # use_reloader=False: cegah webcam dibuka dua kali oleh reloader.
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.DEBUG,
        threaded=True,
        use_reloader=False,
    )
