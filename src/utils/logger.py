"""
logger.py — Setup logging terpusat.

STUB — implementasi minimal boleh diisi nanti.
"""

import logging


def get_logger(name="bisindo", level=logging.INFO):
    """Kembalikan logger terkonfigurasi sederhana."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
