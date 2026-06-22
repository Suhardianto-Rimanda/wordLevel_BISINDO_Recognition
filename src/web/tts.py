"""
tts.py — Wrapper gTTS: teks kalimat → file mp3 (bahasa Indonesia).

Hasilkan mp3 di ``config.TTS_OUTPUT_DIR`` (di bawah static/ agar bisa diserve).
Dipanggil dari endpoint /speak yang berjalan di thread request terpisah →
tidak memblokir thread streaming video.
"""

import time

from gtts import gTTS

import config


def _cleanup_old(audio_dir, keep=3):
    """Sisakan ``keep`` mp3 terbaru, hapus sisanya (cegah folder membengkak)."""
    files = sorted(audio_dir.glob("*.mp3"), key=lambda p: p.stat().st_mtime)
    for old in files[:-keep] if len(files) > keep else []:
        try:
            old.unlink()
        except OSError:
            pass


def text_to_speech(text, lang=None):
    """Konversi ``text`` → file mp3. Kembalikan ``Path`` file.

    gTTS butuh koneksi internet; exception jaringan dilempar ke pemanggil.
    """
    audio_dir = config.TTS_OUTPUT_DIR
    audio_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_old(audio_dir)

    filename = f"tts_{int(time.time() * 1000)}.mp3"
    out_path = audio_dir / filename

    tts = gTTS(text=text, lang=lang or config.TTS_LANG)
    tts.save(str(out_path))
    return out_path
