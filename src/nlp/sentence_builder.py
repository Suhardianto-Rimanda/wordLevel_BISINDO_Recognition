"""
sentence_builder.py — Rakit stream kata stabil menjadi kalimat (rule-based).

Kumpulkan kata yang sudah lolos smoothing → susun jadi string kalimat.
Output dipakai untuk teks UI & input gTTS.

CATATAN JUJUR: ini BUKAN koreksi tata bahasa. Hanya penggabungan + kapitalisasi
+ spasi. Tidak ada parsing S-P-O / urutan kata bahasa Indonesia yang benar.
"""

import config


def sentence_assembly(words):
    """Gabung list kata → string kalimat.

    Aturan (sederhana): buang kosong → join spasi → kapitalisasi huruf pertama.

    # TODO: ini BUKAN koreksi grammar S-P-O. Tanpa anotasi tata bahasa, hasil
    #       hanya rangkaian kata; untuk kalimat gramatikal butuh model/rule
    #       linguistik yang lebih kaya (di luar lingkup TA ini).
    """
    clean = [w for w in words if w]
    if not clean:
        return ""
    sentence = " ".join(clean)
    return sentence[0].upper() + sentence[1:]


class SentenceBuilder:
    """Penampung & perakit kata menjadi kalimat berjalan (stateful realtime)."""

    def __init__(self):
        self.words = []

    def add_word(self, word):
        """Tambah satu kata; lewati kosong, dedup beruntun, & hormati MAX_WORDS."""
        if not word:
            return
        if config.ENFORCE_MAX_WORDS and len(self.words) >= config.MAX_WORDS:
            return  # kalimat sudah penuh -> abaikan sampai reset/finalisasi
        if not self.words or self.words[-1] != word:
            self.words.append(word)

    def get_sentence(self):
        """Kembalikan kalimat saat ini (str)."""
        return sentence_assembly(self.words)

    def clear(self):
        """Reset kalimat."""
        self.words = []
