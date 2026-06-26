// Owner Settings interactions: registry picker, custom-function form, clear,
// whoami diagnostic. All writes are JSON fetch calls re-checked server-side.

function toast(msg, bad) {
  const t = document.getElementById("toast");
  if (!t) return;
  t.textContent = msg;
  t.className = "toast show" + (bad ? " bad" : "");
  setTimeout(() => (t.className = "toast"), 2600);
}

// Persistent, copy-pasteable error shown inline under a save button. Used
// instead of a toast when the form is still on screen after a failed click, so
// the full message (e.g. a model load traceback) sticks around to copy.
function showInlineError(id, msg) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = msg;
  el.classList.add("show");
}

function clearInlineError(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = "";
  el.classList.remove("show");
}

async function postJSON(url, body) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
    credentials: "same-origin",
  });
  const data = await resp.json().catch(() => ({}));
  return { ok: resp.ok, status: resp.status, data };
}

// --- Registry picker ------------------------------------------------------
let REG_MODELS = [];
const regModelSel = document.getElementById("reg-model");
const regVersionSel = document.getElementById("reg-version");

async function loadRegistry() {
  const statusEl = document.getElementById("registry-status");
  try {
    const resp = await fetch("settings/models", { credentials: "same-origin" });
    const data = await resp.json();
    if (!data.available) {
      statusEl.textContent = data.error || "Registry unavailable.";
      return;
    }
    REG_MODELS = data.models || [];
    if (!REG_MODELS.length) {
      statusEl.textContent = "No registered models found in this project.";
      return;
    }
    statusEl.style.display = "none";
    regModelSel.innerHTML = "";
    REG_MODELS.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m.name;
      opt.textContent = m.name;
      regModelSel.appendChild(opt);
    });
    syncVersions();
  } catch (err) {
    statusEl.textContent = "Could not reach the registry: " + err;
  }
}

function syncVersions() {
  const m = REG_MODELS.find((x) => x.name === regModelSel.value);
  regVersionSel.innerHTML = "";
  (m ? m.versions : []).forEach((v) => {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = "v" + v;
    regVersionSel.appendChild(opt);
  });
}

if (regModelSel) {
  regModelSel.addEventListener("change", syncVersions);
  loadRegistry();
}

const btnSaveReg = document.getElementById("btn-save-registry");
if (btnSaveReg) {
  btnSaveReg.addEventListener("click", async () => {
    clearInlineError("reg-error");
    btnSaveReg.disabled = true;
    const r = await postJSON("settings/select", {
      source_type: "registry",
      model_name: regModelSel.value,
      version: regVersionSel.value,
      display_name: document.getElementById("reg-display").value,
    });
    btnSaveReg.disabled = false;
    if (r.ok) {
      toast("Now hosting " + r.data.display_name);
      setTimeout(() => (window.location = document.baseURI), 900);
    } else {
      showInlineError("reg-error", r.data.detail || "Failed to load model");
    }
  });
}

// --- Custom function ------------------------------------------------------
const btnSaveFn = document.getElementById("btn-save-function");
if (btnSaveFn) {
  btnSaveFn.addEventListener("click", async () => {
    clearInlineError("fn-error");
    btnSaveFn.disabled = true;
    const r = await postJSON("settings/select", {
      source_type: "custom_function",
      file_path: document.getElementById("fn-path").value,
      func_name: document.getElementById("fn-name").value,
      display_name: document.getElementById("fn-display").value,
    });
    btnSaveFn.disabled = false;
    if (r.ok) {
      toast("Now hosting " + r.data.display_name);
      setTimeout(() => (window.location = document.baseURI), 900);
    } else {
      showInlineError("fn-error", r.data.detail || "Failed to load function");
    }
  });
}

// --- Clear ----------------------------------------------------------------
const btnClear = document.getElementById("btn-clear");
if (btnClear) {
  btnClear.addEventListener("click", async () => {
    if (!confirm("Clear the hosted model? The endpoints will go offline.")) return;
    const r = await postJSON("settings/clear", {});
    if (r.ok) {
      toast("Cleared.");
      setTimeout(() => window.location.reload(), 700);
    } else toast("Failed to clear", true);
  });
}

// --- Whoami ---------------------------------------------------------------
const btnWho = document.getElementById("btn-whoami");
if (btnWho) {
  btnWho.addEventListener("click", async () => {
    const resp = await fetch("settings/whoami", { credentials: "same-origin" });
    const data = await resp.json();
    const out = document.getElementById("whoami-out");
    out.style.display = "block";
    out.textContent = JSON.stringify(data, null, 2);
  });
}
