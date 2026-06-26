// Playground + docs interactions. ~vanilla JS, no framework.

// The server can't know the external reverse-proxy prefix, so it emits a
// placeholder and we fill the real absolute base from document.baseURI (set via
// the page's <base href>, which reflects the URL the browser actually used).
const APP_BASE = document.baseURI.replace(/\/+$/, "");
const BASE_TOKEN = "__APP_BASE__";

function fillBase(text) {
  return (text || "").split(BASE_TOKEN).join(APP_BASE);
}

// Substitute the placeholder anywhere it was rendered: endpoint URLs + the
// copy-paste curl snippets (both visible text and the toggle's stored variants).
document.querySelectorAll(".url").forEach((el) => {
  el.textContent = fillBase(el.textContent);
});
document.querySelectorAll("pre.curl").forEach((pre) => {
  if (pre.dataset.workload) pre.dataset.workload = fillBase(pre.dataset.workload);
  if (pre.dataset.offplatform) pre.dataset.offplatform = fillBase(pre.dataset.offplatform);
  pre.textContent = fillBase(pre.textContent);
});

// --- "Calling from" toggle: swap the curl snippets in place ---------------
const whereToggle = document.getElementById("where-toggle");
if (whereToggle) {
  whereToggle.addEventListener("change", () => {
    const key = whereToggle.value === "offplatform" ? "offplatform" : "workload";
    document.querySelectorAll("pre.curl").forEach((pre) => {
      pre.textContent = pre.dataset[key];
    });
  });
}

// --- Mode toggle: show the sync OR the async endpoint cards ---------------
const modeToggle = document.getElementById("mode-toggle");
if (modeToggle) {
  const applyMode = () => {
    const mode = modeToggle.value; // "sync" | "async"
    document.querySelectorAll("[data-ep-group]").forEach((el) => {
      el.hidden = el.dataset.epGroup !== mode;
    });
  };
  modeToggle.addEventListener("change", applyMode);
  applyMode();
}

// --- Copy to clipboard ----------------------------------------------------
function showToast(msg, bad) {
  let t = document.getElementById("toast");
  if (!t) {
    t = document.createElement("div");
    t.id = "toast";
    document.body.appendChild(t);
  }
  t.textContent = msg;
  t.className = "toast show" + (bad ? " bad" : "");
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => (t.className = "toast"), 2000);
}

async function copyText(text) {
  // Prefer the async Clipboard API; fall back to a hidden-textarea + execCommand
  // for non-secure contexts (e.g. plain-HTTP dev) where it's unavailable.
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch (_) { /* fall through */ }
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(ta);
    return ok;
  } catch (_) {
    return false;
  }
}

document.querySelectorAll(".copy-btn").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const wrap = btn.closest(".copy-wrap");
    const src = wrap && wrap.querySelector(".copy-src");
    if (!src) return;
    // textContent reflects the live value (post base-substitution + curl toggle).
    const ok = await copyText(src.textContent.trim());
    if (ok) {
      btn.classList.add("copied");
      showToast("Copied to clipboard");
      setTimeout(() => btn.classList.remove("copied"), 1400);
    } else {
      showToast("Copy failed", true);
    }
  });
});

// --- Playground -----------------------------------------------------------
const form = document.getElementById("pg-form");

function readFileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      // Strip the "data:...;base64," prefix → raw base64 string.
      const res = String(reader.result);
      resolve(res.includes(",") ? res.split(",")[1] : res);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function collectRecord() {
  // Passthrough models: a single raw-JSON textarea is the record.
  if (form.dataset.raw === "1") {
    const raw = document.getElementById("pg-raw").value;
    return JSON.parse(raw); // throws on bad JSON → surfaced as an error
  }
  const record = {};
  const inputs = form.querySelectorAll("[data-field]");
  for (const el of inputs) {
    const name = el.dataset.field;
    if (el.dataset.image === "1") {
      if (el.files && el.files[0]) record[name] = await readFileAsBase64(el.files[0]);
    } else {
      const v = el.value.trim();
      if (v !== "") record[name] = v;
    }
  }
  return record;
}

function setStatus(text, kind) {
  const s = document.getElementById("pg-status");
  s.textContent = text;
  s.className = "pg-status" + (kind ? " " + kind : "");
}

function showResult(obj) {
  document.getElementById("pg-result").textContent =
    typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
}

async function runSync(base, slug, record) {
  const t0 = performance.now();
  const resp = await fetch(`${base}/models/${slug}/latest/model`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data: record }),
    credentials: "same-origin",
  });
  const ms = Math.round(performance.now() - t0);
  const body = await resp.json().catch(() => ({}));
  setStatus(`HTTP ${resp.status} · ${ms} ms`, resp.ok ? "ok" : "bad");
  showResult(body);
}

async function runAsync(base, slug, record) {
  const abase = `${base}/api/modelApis/async/v1/${slug}`;
  setStatus("submitting…");
  const sub = await fetch(abase, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ parameters: record }),
    credentials: "same-origin",
  });
  const subBody = await sub.json().catch(() => ({}));
  if (!sub.ok) {
    setStatus(`HTTP ${sub.status}`, "bad");
    showResult(subBody);
    return;
  }
  const id = subBody.asyncPredictionId;
  showResult(subBody);

  const terminal = ["succeeded", "failed", "cancelled", "expired"];
  const t0 = performance.now();
  for (let i = 0; i < 120; i++) {
    await new Promise((r) => setTimeout(r, 1000));
    const poll = await fetch(`${abase}/${id}`, { credentials: "same-origin" });
    const pb = await poll.json().catch(() => ({}));
    const ms = Math.round(performance.now() - t0);
    setStatus(`status: ${pb.status} · ${ms} ms elapsed`, pb.status === "succeeded" ? "ok" : (pb.status === "failed" ? "bad" : ""));
    showResult(pb);
    if (terminal.includes(pb.status)) return;
  }
  setStatus("gave up polling after 120s", "bad");
}

if (form) {
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const base = APP_BASE;  // absolute app root from document.baseURI
    const slug = form.dataset.slug;
    const isAsync = document.getElementById("pg-async").checked;
    try {
      const record = await collectRecord();
      if (isAsync) await runAsync(base, slug, record);
      else await runSync(base, slug, record);
    } catch (err) {
      setStatus("error", "bad");
      showResult(String(err));
    }
  });
}
