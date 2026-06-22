"""
test_nlp.py — Unit test modul NLP (smoothing, deduping, sentence_assembly,
PredictionSmoother, SentenceBuilder). Stdlib unittest, zero dependency.

Jalankan: python -m unittest tests.test_nlp -v
"""

import sys
import unittest
from pathlib import Path

# pastikan root proyek di sys.path (untuk import config & src.*)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nlp.smoother import smoothing, deduping, PredictionSmoother
from src.nlp.sentence_builder import sentence_assembly, SentenceBuilder


class TestSmoothing(unittest.TestCase):
    def test_commit_saat_capai_n(self):
        # 'a' muncul 3x berturut -> commit; 'b' 3x -> commit
        self.assertEqual(smoothing(["a", "a", "a", "b", "b", "b"], n=3), ["a", "b"])

    def test_kurang_dari_n_tak_commit(self):
        # tak ada kata yang capai 3x berturut
        self.assertEqual(smoothing(["a", "a", "b", "b", "a", "a"], n=3), [])

    def test_none_reset_flicker(self):
        # noise (None) memutus run -> 'a' tak pernah capai 3 berturut
        self.assertEqual(smoothing(["a", "a", None, "a", "b", "b", "b"], n=3), ["b"])

    def test_multi_commit_dengan_run_panjang(self):
        # run panjang tak commit ulang; kata sama setelah jeda boleh commit lagi
        seq = ["x", "x", "x", "x", "y", "y", "y", "x", "x", "x"]
        self.assertEqual(smoothing(seq, n=3), ["x", "y", "x"])

    def test_default_n_dari_config(self):
        # default window=5: 4x belum cukup, 5x cukup
        self.assertEqual(smoothing(["a"] * 4), [])
        self.assertEqual(smoothing(["a"] * 5), ["a"])


class TestDeduping(unittest.TestCase):
    def test_kolaps_beruntun(self):
        self.assertEqual(
            deduping(["makan", "makan", "makan"]), ["makan"]
        )

    def test_non_beruntun_dipertahankan(self):
        self.assertEqual(
            deduping(["makan", "makan", "minum", "makan"]),
            ["makan", "minum", "makan"],
        )

    def test_buang_kosong(self):
        self.assertEqual(
            deduping(["", "halo", None, "halo", "halo", ""]), ["halo"]
        )

    def test_list_kosong(self):
        self.assertEqual(deduping([]), [])


class TestSentenceAssembly(unittest.TestCase):
    def test_join_kapital(self):
        self.assertEqual(
            sentence_assembly(["saya", "makan", "nasi"]), "Saya makan nasi"
        )

    def test_list_kosong(self):
        self.assertEqual(sentence_assembly([]), "")

    def test_filter_none(self):
        self.assertEqual(sentence_assembly([None, "halo", "", "dunia"]), "Halo dunia")

    def test_satu_kata(self):
        self.assertEqual(sentence_assembly(["halo"]), "Halo")


class TestPredictionSmoother(unittest.TestCase):
    def test_commit_konsisten_dengan_smoothing(self):
        seq = ["a", "a", "a", "b", "b", "b", "a", "a", "a"]
        sm = PredictionSmoother(window=3)
        commits = [w for w in (sm.update(x) for x in seq) if w is not None]
        self.assertEqual(commits, smoothing(seq, n=3))

    def test_update_balikan_none_sebelum_capai_window(self):
        sm = PredictionSmoother(window=3)
        self.assertIsNone(sm.update("a"))
        self.assertIsNone(sm.update("a"))
        self.assertEqual(sm.update("a"), "a")     # capai 3 -> commit
        self.assertIsNone(sm.update("a"))         # sudah ter-commit

    def test_noise_memutus_run(self):
        sm = PredictionSmoother(window=3)
        sm.update("a"); sm.update("a")
        self.assertIsNone(sm.update(None))        # reset
        self.assertIsNone(sm.update("a"))         # mulai run baru

    def test_reset(self):
        sm = PredictionSmoother(window=2)
        sm.update("a"); sm.update("a")
        sm.reset()
        self.assertIsNone(sm.update("a"))
        self.assertEqual(sm.update("a"), "a")


class TestSentenceBuilder(unittest.TestCase):
    def test_add_word_dedup_beruntun(self):
        sb = SentenceBuilder()
        for w in ["saya", "saya", "makan", "makan", "nasi"]:
            sb.add_word(w)
        self.assertEqual(sb.get_sentence(), "Saya makan nasi")

    def test_skip_kosong(self):
        sb = SentenceBuilder()
        for w in [None, "halo", "", "dunia"]:
            sb.add_word(w)
        self.assertEqual(sb.get_sentence(), "Halo dunia")

    def test_clear(self):
        sb = SentenceBuilder()
        sb.add_word("halo")
        sb.clear()
        self.assertEqual(sb.get_sentence(), "")

    def test_max_words_cap(self):
        # tak menumpuk melewati config.MAX_WORDS (kata unik agar lolos dedup)
        import config
        sb = SentenceBuilder()
        for i in range(config.MAX_WORDS + 10):
            sb.add_word(f"kata{i}")
        self.assertEqual(len(sb.words), config.MAX_WORDS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
