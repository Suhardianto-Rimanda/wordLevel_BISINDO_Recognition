"""
frame_buffer.py — Sliding buffer 30 frame landmark untuk inference real-time.

Tampung vektor fitur per-frame (126-dim). Saat penuh (``config.SEQUENCE_LENGTH``)
sediakan array (SEQUENCE_LENGTH, FEATURE_DIM) siap diprediksi. Buffer geser:
frame terlama otomatis terbuang saat penuh (deque maxlen).
"""

from collections import deque

import numpy as np

import config


class FrameBuffer:
    """Buffer geser fitur landmark sepanjang SEQUENCE_LENGTH frame."""

    def __init__(self, maxlen=None):
        self.maxlen = maxlen or config.SEQUENCE_LENGTH
        self.buffer = deque(maxlen=self.maxlen)

    def append(self, features):
        """Tambah satu vektor fitur frame (np.ndarray shape (FEATURE_DIM,))."""
        self.buffer.append(np.asarray(features, dtype=np.float32))

    def is_ready(self):
        """True bila buffer sudah penuh (siap prediksi)."""
        return len(self.buffer) == self.maxlen

    def as_array(self):
        """Kembalikan np.ndarray (SEQUENCE_LENGTH, FEATURE_DIM)."""
        return np.array(self.buffer, dtype=np.float32)

    def clear(self):
        """Kosongkan buffer."""
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)
