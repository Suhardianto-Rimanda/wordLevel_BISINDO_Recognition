"""
smoother.py — Smoothing & dedup stream prediksi kata (rule-based).

Stream prediksi per-frame masih ber-flicker (kata berganti-ganti karena noise).
Dua alat:
  * ``smoothing`` — kata dianggap valid hanya bila muncul konsisten N kali
    berturut-turut (redam flicker).
  * ``deduping``  — hapus duplikat beruntun ("makan makan" -> "makan").
  * ``PredictionSmoother`` — versi stateful realtime (per-frame) dari aturan
    smoothing, untuk dipakai pipeline live (inference/web).

Semua rule-based sederhana — TANPA model bahasa.
"""

import config


def smoothing(predictions, n=None):
    """Stabilkan urutan prediksi: commit kata bila muncul ``n`` kali berturut-turut.

    Args:
        predictions: list kata per-frame (str). Entri falsy (None/"") = noise.
        n: panjang run konsisten minimum (default ``config.SMOOTHING_WINDOW``).

    Returns:
        list[str] kata ter-commit secara berurutan. Commit baru hanya ditambahkan
        bila berbeda dari commit sebelumnya (anti pengulangan langsung).
    """
    n = n or config.SMOOTHING_WINDOW
    committed_list = []
    run_word, run_len, last_committed = None, 0, None

    for word in predictions:
        if not word:                      # noise -> putus run, tak commit
            run_word, run_len = None, 0
            continue
        if word == run_word:
            run_len += 1
        else:
            run_word, run_len = word, 1
        if run_len >= n and word != last_committed:
            last_committed = word
            committed_list.append(word)

    return committed_list


def deduping(words):
    """Hapus duplikat BERUNTUN. Duplikat tak-beruntun dipertahankan.

    ``["makan","makan","minum","makan"] -> ["makan","minum","makan"]``
    Entri falsy dibuang.
    """
    out = []
    for word in words:
        if not word:
            continue
        if not out or out[-1] != word:
            out.append(word)
    return out


class PredictionSmoother:
    """Versi stateful realtime dari ``smoothing`` (dipanggil per frame).

    Pakai aturan run-konsisten yang sama: ``update`` mengembalikan kata HANYA
    pada saat ia baru ter-commit (muncul ``window`` kali berturut & beda dari
    commit terakhir), selain itu None.
    """

    def __init__(self, window=None):
        self.window = window or config.SMOOTHING_WINDOW
        self._run_word = None
        self._run_len = 0
        self._committed = None

    def update(self, word, prob=None):
        """Terima prediksi satu frame → kata ter-commit (str) atau None.

        ``prob`` diterima untuk kompatibilitas API engine (threshold sudah
        diterapkan di hulu); tak dipakai di sini.
        """
        if not word:                      # noise -> putus run
            self._run_word, self._run_len = None, 0
            return None
        if word == self._run_word:
            self._run_len += 1
        else:
            self._run_word, self._run_len = word, 1
        if self._run_len >= self.window and word != self._committed:
            self._committed = word
            return word
        return None

    def reset(self):
        """Reset state (mis. mulai kalimat baru)."""
        self._run_word, self._run_len, self._committed = None, 0, None
