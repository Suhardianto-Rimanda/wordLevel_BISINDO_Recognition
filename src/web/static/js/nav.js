// nav.js — Navigasi single-page: ganti section (show/hide) lewat navbar / tombol
// [data-goto]. MURNI tampilan — tidak menyentuh kamera, polling, atau Web Speech.
// Semua elemen pipeline tetap ada di DOM (hanya section-nya yang disembunyikan).

document.addEventListener("DOMContentLoaded", () => {
  const pages = Array.from(document.querySelectorAll(".page")); // #page-landing/translator/about
  const links = Array.from(document.querySelectorAll(".nav-link"));
  const VALID = ["landing", "translator", "about"];

  function showPage(name) {
    const target = VALID.includes(name) ? name : "landing";
    pages.forEach((p) => {
      p.hidden = p.dataset.page !== target;
    });
    links.forEach((a) => {
      a.classList.toggle("is-active", a.dataset.goto === target);
    });
    if (location.hash !== "#" + target) {
      history.replaceState(null, "", "#" + target);
    }
    window.scrollTo({ top: 0, behavior: "auto" });
  }

  // Klik apa pun dengan [data-goto] (link navbar, brand, tombol CTA).
  document.querySelectorAll("[data-goto]").forEach((el) => {
    el.addEventListener("click", (e) => {
      e.preventDefault();
      showPage(el.dataset.goto);
    });
  });

  // Dukung tombol back/forward browser (perubahan hash).
  window.addEventListener("hashchange", () => {
    showPage(location.hash.replace("#", ""));
  });

  // Section awal dari hash (default: landing).
  showPage(location.hash.replace("#", "") || "landing");
});
