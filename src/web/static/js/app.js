// app.js — interaksi UI: mulai kamera, polling state, audio gTTS, reset.

document.addEventListener("DOMContentLoaded", () => {
  const video = document.getElementById("video");
  const placeholder = document.getElementById("video-placeholder");
  const rawWord = document.getElementById("raw-word");
  const rawScore = document.getElementById("raw-score");
  const sentenceEl = document.getElementById("sentence");
  const autoplay = document.getElementById("autoplay");
  const player = document.getElementById("player");
  const audioStatus = document.getElementById("audio-status");
  const btnStart = document.getElementById("btn-start");
  const btnReset = document.getElementById("btn-reset");
  const btnReplay = document.getElementById("btn-replay");

  let polling = null;
  let lastFinalId = 0;
  let speaking = false;

  // ---- Mulai Kamera ----
  btnStart.addEventListener("click", () => {
    video.src = "/video_feed";            // memicu pembukaan webcam server-side
    placeholder.style.display = "none";
    btnStart.disabled = true;
    btnStart.textContent = "● Kamera Aktif";
    if (!polling) polling = setInterval(pollState, 400);
  });

  video.addEventListener("error", () => {
    placeholder.style.display = "flex";
    placeholder.innerHTML = "Gagal membuka webcam.<br/>Cek perangkat kamera.";
    btnStart.disabled = false;
    btnStart.textContent = "▶ Mulai Kamera";
  });

  // ---- Polling state ----
  async function pollState() {
    try {
      const res = await fetch("/state");
      const s = await res.json();
      rawWord.textContent = s.raw_word ? s.raw_word : "—";
      rawScore.textContent = Number(s.raw_score || 0).toFixed(2);
      sentenceEl.textContent = s.sentence ? s.sentence : "—";

      // Autoplay HANYA saat kalimat DIFINALISASI (final_id naik), bukan tiap kata.
      // Cegah gTTS membaca ulang kalimat yang masih tumbuh.
      if (s.final_id && s.final_id !== lastFinalId) {
        lastFinalId = s.final_id;
        if (autoplay.checked && s.final_sentence) speakText(s.final_sentence);
      }
    } catch (e) {
      /* abaikan error transien polling */
    }
  }

  // ---- Reset ----
  btnReset.addEventListener("click", async () => {
    await fetch("/reset", { method: "POST" });
    rawWord.textContent = "—";
    rawScore.textContent = "0.00";
    sentenceEl.textContent = "—";
    audioStatus.textContent = "";
  });

  // ---- Audio gTTS ----
  // Putar ulang: bunyikan kalimat yang sedang tampil (live atau final terakhir).
  btnReplay.addEventListener("click", () => speakText(sentenceEl.textContent.trim()));

  async function speakText(text) {
    if (speaking) return;
    if (!text || text === "—") {
      audioStatus.textContent = "Belum ada kalimat.";
      return;
    }
    speaking = true;
    btnReplay.disabled = true;
    audioStatus.textContent = "Membuat audio…";
    try {
      const res = await fetch("/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentence: text }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        audioStatus.textContent = err.error || "Gagal membuat audio.";
        return;
      }
      const data = await res.json();
      player.src = data.url;
      await player.play();
      audioStatus.textContent = "▶ Memutar: " + data.text;
    } catch (e) {
      audioStatus.textContent = "Error audio.";
    } finally {
      speaking = false;
      btnReplay.disabled = false;
    }
  }
});
