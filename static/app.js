// Playground + docs interactions. ~vanilla JS, no framework.

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
    const base = form.dataset.base;
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
