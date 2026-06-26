// Settings page — unified model hosting card with form / status panels.

const INIT = window.INIT_STATE || {};

// --- Shared helpers ----------------------------------------------------------

function toast(msg, bad) {
  const t = document.getElementById("toast");
  if (!t) return;
  t.textContent = msg;
  t.className = "toast show" + (bad ? " bad" : "");
  setTimeout(() => (t.className = "toast"), 2600);
}

function escHtml(s) {
  return String(s || "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function showInlineError(id, msg) {
  const el = document.getElementById(id);
  if (el) { el.textContent = msg; el.classList.add("show"); }
}

function clearInlineError(id) {
  const el = document.getElementById(id);
  if (el) { el.textContent = ""; el.classList.remove("show"); }
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

// --- Panel switching ---------------------------------------------------------

function showPanel(which) {
  document.getElementById("status-panel").style.display = which === "status" ? "" : "none";
  document.getElementById("config-panel").style.display = which === "config" ? "" : "none";
}

// --- Status panel ------------------------------------------------------------

function renderStatusPanel(st) {
  const row = document.getElementById("status-row");
  const sourceLabel = st.sourceType === "registry" ? "registered model" : "custom function";
  const dotCls = st.ready ? "ok" : "bad";
  row.innerHTML =
    `<span class="status-dot ${dotCls}"></span>` +
    `<span class="status-name">${escHtml(st.displayName)}</span>` +
    `<span class="status-meta">${sourceLabel}&thinsp;&middot;&thinsp;slug&nbsp;<code>${escHtml(st.slug)}</code></span>` +
    `<span class="status-badge ${dotCls}">${st.ready ? "Live" : "Error"}</span>`;
  const errEl = document.getElementById("status-error");
  if (st.error) { errEl.textContent = st.error; errEl.style.display = ""; }
  else { errEl.style.display = "none"; }
}

// --- Source tab (segmented control) ------------------------------------------

let activeTab = "registry";

function setTab(tab) {
  activeTab = tab;
  document.getElementById("registry-fields").style.display = tab === "registry" ? "" : "none";
  document.getElementById("function-fields").style.display = tab === "function" ? "" : "none";
  document.querySelectorAll(".seg-btn[data-tab]").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.tab === tab);
  });
}

document.querySelectorAll(".seg-btn[data-tab]").forEach(btn => {
  btn.addEventListener("click", () => setTab(btn.dataset.tab));
});

// --- Registry picker ---------------------------------------------------------

let REG_MODELS = [];
let pendingRegModel = null;
let pendingRegVersion = null;

const regModelSel = document.getElementById("reg-model");
const regVersionSel = document.getElementById("reg-version");

function syncVersions() {
  const m = REG_MODELS.find(x => x.name === regModelSel.value);
  regVersionSel.innerHTML = "";
  (m ? m.versions : []).forEach(v => {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = "v" + v;
    regVersionSel.appendChild(opt);
  });
  if (pendingRegVersion) {
    regVersionSel.value = pendingRegVersion;
    pendingRegVersion = null;
  }
}

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
    REG_MODELS.forEach(m => {
      const opt = document.createElement("option");
      opt.value = m.name;
      opt.textContent = m.name;
      regModelSel.appendChild(opt);
    });
    if (pendingRegModel) {
      regModelSel.value = pendingRegModel;
      pendingRegModel = null;
    }
    syncVersions();
  } catch (err) {
    statusEl.textContent = "Could not reach the registry: " + err;
  }
}

if (regModelSel) {
  regModelSel.addEventListener("change", syncVersions);
  loadRegistry();
}

// --- Pre-fill form from existing config (for Edit mode) ----------------------

function prefillForm(initState) {
  const params = initState.params || {};
  if (initState.sourceType === "registry") {
    setTab("registry");
    pendingRegModel = params.model_name || null;
    pendingRegVersion = params.version || null;
    // If registry already loaded, apply immediately
    if (REG_MODELS.length && pendingRegModel) {
      regModelSel.value = pendingRegModel;
      pendingRegModel = null;
      syncVersions();
    }
    document.getElementById("reg-display").value = initState.displayName || "";
  } else if (initState.sourceType === "custom_function") {
    setTab("function");
    document.getElementById("fn-path").value = params.file_path || "";
    document.getElementById("fn-name").value = params.func_name || "predict";
    document.getElementById("fn-display").value = initState.displayName || "";
  }
}

// --- Deploy button -----------------------------------------------------------

const btnDeploy = document.getElementById("btn-deploy");
if (btnDeploy) {
  btnDeploy.addEventListener("click", async () => {
    clearInlineError("deploy-error");

    let body;
    if (activeTab === "registry") {
      if (!regModelSel || !regModelSel.value) {
        showInlineError("deploy-error", "Select a model first.");
        return;
      }
      body = {
        source_type: "registry",
        model_name: regModelSel.value,
        version: regVersionSel.value,
        display_name: document.getElementById("reg-display").value,
      };
    } else {
      const filePath = (document.getElementById("fn-path").value || "").trim();
      if (!filePath) {
        showInlineError("deploy-error", "File path is required.");
        return;
      }
      body = {
        source_type: "custom_function",
        file_path: filePath,
        func_name: document.getElementById("fn-name").value || "predict",
        display_name: document.getElementById("fn-display").value,
      };
    }

    btnDeploy.disabled = true;
    btnDeploy.textContent = "Deploying…";
    btnDeploy.classList.add("btn-loading");

    const r = await postJSON("settings/select", body);

    btnDeploy.disabled = false;
    btnDeploy.textContent = "Deploy model";
    btnDeploy.classList.remove("btn-loading");

    if (r.ok) {
      toast("Now hosting " + r.data.display_name);
      renderStatusPanel({
        ready: r.data.ready,
        displayName: r.data.display_name,
        slug: r.data.slug,
        sourceType: body.source_type,
        error: "",
      });
      showPanel("status");
    } else {
      showInlineError("deploy-error", r.data.detail || "Failed to deploy model.");
    }
  });
}

// --- Stop button -------------------------------------------------------------

const btnStop = document.getElementById("btn-stop");
if (btnStop) {
  btnStop.addEventListener("click", async () => {
    if (!confirm("Stop hosting this model? The endpoints will go offline.")) return;
    btnStop.disabled = true;
    const r = await postJSON("settings/clear", {});
    btnStop.disabled = false;
    if (r.ok) {
      toast("Model stopped.");
      clearInlineError("deploy-error");
      setTab("registry");
      showPanel("config");
    } else {
      toast("Failed to stop model.", true);
    }
  });
}

// --- Edit button -------------------------------------------------------------

const btnEdit = document.getElementById("btn-edit");
if (btnEdit) {
  btnEdit.addEventListener("click", () => {
    prefillForm(INIT);
    showPanel("config");
  });
}

// --- Whoami ------------------------------------------------------------------

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

// --- Initialize --------------------------------------------------------------

if (INIT.configured) {
  renderStatusPanel(INIT);
  showPanel("status");
} else {
  showPanel("config");
}
