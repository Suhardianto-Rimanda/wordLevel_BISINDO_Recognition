// speech.js — Panel "Suara → Teks" (Web Speech API). MURNI frontend, independen
// dari app.js (pipeline isyarat). Mengubah SUARA menjadi TEKS untuk dibaca pengguna
// tunarungu — BUKAN menerjemahkan suara ke isyarat. Tak menyentuh backend/route apa pun.

document.addEventListener("DOMContentLoaded", () => {
  const panel = document.getElementById("speech-panel");
  const toggle = document.getElementById("speech-toggle");
  const clearBtn = document.getElementById("speech-clear");
  const statusEl = document.getElementById("speech-status");
  const chatEl = document.getElementById("speech-chat"); // container gelembung chat
  const interimEl = document.getElementById("speech-interim");

  // Tambah 1 hasil final sebagai gelembung chat (display saja).
  function addBubble(text) {
    const t = (text || "").trim();
    if (!t) return;
    const div = document.createElement("div");
    div.className = "speech-bubble";
    div.textContent = t;
    chatEl.appendChild(div);
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;

  function setStatus(text, cls) {
    statusEl.textContent = text;
    statusEl.classList.remove("is-listening", "is-error");
    if (cls) statusEl.classList.add(cls);
  }

  // ---- Browser tak mendukung Web Speech API → pesan jelas + disable, jangan crash ----
  if (!SR) {
    panel.classList.add("unsupported");
    toggle.disabled = true;
    clearBtn.disabled = true;
    setStatus("Fitur ini memerlukan browser Chrome (Web Speech API tak tersedia).", "is-error");
    return;
  }

  // Pengaman anti restart-loop: di mode continuous, recognition bisa berhenti sendiri
  // (onend) terutama saat error beruntun. Tanpa guard, restart langsung bisa jadi loop
  // cepat. Maka: jeda sebelum restart + batas restart beruntun tanpa hasil.
  const MAX_RESTARTS = 5;
  const RESTART_DELAY_MS = 400;

  let recognition = null;
  let listening = false; // niat pengguna: true = ingin mendengarkan
  let restartCount = 0; // restart beruntun sejak aktivitas suara terakhir
  let restartTimer = null;

  function syncToggle() {
    if (listening) {
      toggle.textContent = "■ Berhenti";
      toggle.classList.remove("btn-primary");
      toggle.classList.add("btn-secondary");
    } else {
      toggle.textContent = "🎤 Mulai Bicara";
      toggle.classList.remove("btn-secondary");
      toggle.classList.add("btn-primary");
    }
  }

  function safeStart() {
    try {
      recognition.start();
    } catch (e) {
      // start() ganda saat sudah berjalan → InvalidStateError; aman diabaikan.
    }
  }

  function buildRecognition() {
    const r = new SR();
    r.lang = "id-ID";
    r.continuous = true;
    r.interimResults = true;

    r.onstart = () => setStatus("Mendengarkan…", "is-listening");

    r.onresult = (event) => {
      restartCount = 0; // ada hasil → sehat, reset guard anti-loop
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const res = event.results[i];
        const txt = res[0].transcript;
        if (res.isFinal) {
          addBubble(txt); // tiap hasil final → gelembung chat berurutan
        } else {
          interim += txt;
        }
      }
      interimEl.textContent = interim;
    };

    r.onerror = (event) => {
      const err = event.error;
      if (err === "not-allowed" || err === "service-not-allowed") {
        // fatal: izin ditolak → hentikan total, jangan auto-restart.
        listening = false;
        syncToggle();
        setStatus("Akses mikrofon ditolak. Izinkan mikrofon di browser lalu coba lagi.", "is-error");
      } else if (err === "audio-capture") {
        listening = false;
        syncToggle();
        setStatus("Mikrofon tak ditemukan. Cek perangkat audio.", "is-error");
      } else if (err === "no-speech") {
        setStatus("Tak ada suara terdeteksi…", "is-error");
      } else if (err === "network") {
        setStatus("Masalah jaringan — fitur ini butuh internet.", "is-error");
      } else if (err === "aborted") {
        // umumnya akibat stop() manual; tak perlu pesan.
      } else {
        setStatus("Kesalahan pengenalan suara: " + err, "is-error");
      }
      // Keputusan restart ditangani di onend (dengan guard), bukan di sini.
    };

    r.onend = () => {
      interimEl.textContent = "";
      if (!listening) {
        // pengguna menekan "Berhenti" → jangan restart.
        setStatus("Berhenti.");
        return;
      }
      // continuous putus sendiri → auto-restart, tapi dijaga agar tak loop cepat.
      if (restartCount >= MAX_RESTARTS) {
        listening = false;
        syncToggle();
        setStatus(
          "Pengenalan berhenti otomatis (gangguan berulang). Tekan “Mulai Bicara” untuk mencoba lagi.",
          "is-error"
        );
        return;
      }
      restartCount++;
      clearTimeout(restartTimer);
      restartTimer = setTimeout(() => {
        if (listening) safeStart();
      }, RESTART_DELAY_MS);
    };

    return r;
  }

  toggle.addEventListener("click", () => {
    if (listening) {
      listening = false;
      clearTimeout(restartTimer);
      if (recognition) recognition.stop();
      syncToggle();
      setStatus("Berhenti.");
    } else {
      if (!recognition) recognition = buildRecognition();
      listening = true;
      restartCount = 0;
      syncToggle();
      setStatus("Memulai…", "is-listening");
      safeStart();
    }
  });

  clearBtn.addEventListener("click", () => {
    chatEl.innerHTML = "";
    interimEl.textContent = "";
  });

  syncToggle();
});
