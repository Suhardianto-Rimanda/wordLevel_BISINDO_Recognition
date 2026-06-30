// upload.js — Panel "Pengujian via Upload Video". MURNI frontend untuk fitur uji
// file, TERPISAH dari pipeline real-time (app.js/speech.js tak disentuh). Kirim
// video ke POST /predict-file lalu tampilkan: daftar kata berurutan + kalimat NLP
// (+ top-k saat hanya 1 kata).

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("upload-form");
  if (!form) return; // panel tak ada → tak melakukan apa pun

  const fileInput = document.getElementById("video-file");
  const fileLabel = document.getElementById("file-label");
  const btn = document.getElementById("btn-upload-test");
  const statusEl = document.getElementById("upload-status");
  const resultEl = document.getElementById("upload-result");
  const wordsEl = document.getElementById("detected-words");
  const sentenceEl = document.getElementById("detected-sentence");
  const metaEl = document.getElementById("ur-meta");
  const topkSection = document.getElementById("topk-section");
  const topkEl = document.getElementById("upload-topk");

  const ALLOWED = [".mp4", ".avi", ".mov", ".mkv"];
  const MAX_BYTES = 50 * 1024 * 1024;

  function setStatus(text, cls) {
    statusEl.textContent = text || "";
    statusEl.classList.remove("is-listening", "is-error");
    if (cls) statusEl.classList.add(cls);
  }

  function pct(score) {
    return (Number(score || 0) * 100).toFixed(1) + "%";
  }

  function extOf(name) {
    const i = name.lastIndexOf(".");
    return i >= 0 ? name.slice(i).toLowerCase() : "";
  }

  // Nama file terpilih → label (validasi ringan di klien).
  fileInput.addEventListener("change", () => {
    const f = fileInput.files && fileInput.files[0];
    fileLabel.textContent = f ? f.name : "Pilih file video…";
    resultEl.hidden = true;
    setStatus("");
  });

  // Rentetan kata terdeteksi → chip dengan panah antar kata.
  function renderWords(words) {
    wordsEl.innerHTML = "";
    if (!words || words.length === 0) {
      const span = document.createElement("span");
      span.className = "words-empty";
      span.textContent = "— tidak ada kata terdeteksi —";
      wordsEl.appendChild(span);
      return;
    }
    words.forEach((w, i) => {
      if (i > 0) {
        const arrow = document.createElement("span");
        arrow.className = "word-arrow";
        arrow.textContent = "→";
        wordsEl.appendChild(arrow);
      }
      const chip = document.createElement("span");
      chip.className = "word-chip";
      const num = document.createElement("span");
      num.className = "word-num";
      num.textContent = i + 1;
      chip.appendChild(num);
      chip.appendChild(document.createTextNode(w));
      wordsEl.appendChild(chip);
    });
  }

  function renderTopk(list) {
    topkEl.innerHTML = "";
    (list || []).forEach((c, i) => {
      const li = document.createElement("li");
      li.className = "topk-item";
      const rank = document.createElement("span");
      rank.className = "topk-rank";
      rank.textContent = "#" + (i + 1);
      const word = document.createElement("span");
      word.className = "topk-word";
      word.textContent = c.word;
      const bar = document.createElement("span");
      bar.className = "topk-bar";
      const fill = document.createElement("i");
      fill.style.width = Math.max(2, Number(c.score || 0) * 100) + "%";
      bar.appendChild(fill);
      const sc = document.createElement("span");
      sc.className = "topk-score";
      sc.textContent = pct(c.score);
      li.append(rank, word, bar, sc);
      topkEl.appendChild(li);
    });
  }

  function renderResult(data) {
    const words = data.words || [];
    renderWords(words);
    sentenceEl.textContent = data.sentence ? data.sentence : "—";

    // meta: jumlah frame + catatan server (video pendek / tak ada kata, dll).
    const parts = [];
    if (data.frames_processed != null) parts.push(`${data.frames_processed} frame diproses`);
    if (data.words_detail && data.words_detail.length) parts.push(`${data.words_detail.length} kata`);
    metaEl.textContent = parts.join(" · ");
    if (data.note) {
      const noteEl = document.createElement("div");
      noteEl.className = "ur-note";
      noteEl.textContent = "ℹ " + data.note;
      metaEl.appendChild(noteEl);
    }

    // top-k hanya relevan untuk kasus 1 kata (kandidat whole-video).
    if (words.length <= 1 && data.topk && data.topk.length) {
      renderTopk(data.topk);
      topkSection.hidden = false;
    } else {
      topkSection.hidden = true;
    }

    resultEl.hidden = false;
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = fileInput.files && fileInput.files[0];

    // --- validasi klien (server tetap validasi ulang) ---
    if (!f) {
      setStatus("Pilih file video dulu.", "is-error");
      return;
    }
    if (!ALLOWED.includes(extOf(f.name))) {
      setStatus("Tipe tak didukung. Gunakan: " + ALLOWED.join(", ") + ".", "is-error");
      return;
    }
    if (f.size > MAX_BYTES) {
      setStatus("File terlalu besar (maks 50 MB).", "is-error");
      return;
    }

    const fd = new FormData();
    fd.append("video", f);

    btn.disabled = true;
    fileInput.disabled = true;
    resultEl.hidden = true;
    setStatus("⏳ Memproses video… (sliding window + inferensi LSTM, bisa beberapa detik)", "is-listening");

    try {
      const res = await fetch("/predict-file", { method: "POST", body: fd });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data.ok) {
        setStatus("⚠ " + (data.error || "Gagal memproses video."), "is-error");
        return;
      }
      setStatus("✓ Selesai.", "is-listening");
      renderResult(data);
    } catch (err) {
      setStatus("⚠ Gagal menghubungi server. Coba lagi.", "is-error");
    } finally {
      btn.disabled = false;
      fileInput.disabled = false;
    }
  });
});
