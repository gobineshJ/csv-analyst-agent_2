// The page. Plain JavaScript modules, no build step.
// Rule: text from the agent or the data is only ever inserted with textContent / renderMarkdown,
// never innerHTML, because it can contain whatever was in a CSV cell.
import { renderMarkdown } from "./markdown.js";

const $ = (id) => document.getElementById(id);
const state = { datasets: [], session: null, busy: false, controller: null };
const SUGGESTIONS = [
  "What are total sales by region?",
  "Make a bar chart of sales by category",
  "Which category has the highest average units per order?",
  "What percentage of sales came from Electronics?",
];

// --- small helpers --------------------------------------------------------------------

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function toast(message) {
  const t = $("toast");
  t.textContent = message;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (t.hidden = true), 4000);
}

// Charts live in runtime/outputs and are served by /api/files/<name>. Any other image is dropped.
function chartUrl(path) {
  const name = String(path).split(/[\\/]/).pop();
  return /^[\w.-]+\.png$/.test(name) ? `/api/files/${encodeURIComponent(name)}?t=${Date.now()}` : null;
}

function markdown(text) {
  const box = el("div", "md");
  box.append(renderMarkdown(text, { image: chartUrl }));
  return box;
}

function pretty(value) {
  if (typeof value !== "string") return JSON.stringify(value, null, 2);
  try { return JSON.stringify(JSON.parse(value), null, 2); } catch { return value; }
}

function parseJson(text) {
  try { return JSON.parse(text); } catch { return null; }
}

async function api(path, options = {}) {
  const resp = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({}));
    throw new Error(typeof detail.detail === "string" ? detail.detail : resp.statusText);
  }
  return resp.json();
}

// POST and read a Server-Sent Events stream (EventSource only supports GET).
async function stream(path, body, onEvent, signal) {
  const resp = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" },
                                   body: JSON.stringify(body), signal });
  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({}));
    throw new Error(typeof detail.detail === "string" ? detail.detail : resp.statusText);
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop();
    for (const part of parts) if (part.startsWith("data: ")) onEvent(JSON.parse(part.slice(6)));
  }
}

// --- datasets ---------------------------------------------------------------------------

async function loadDatasets(select) {
  state.datasets = await api("/api/datasets");
  const box = $("dataset");
  const keep = select || box.value || "sample_sales.csv";
  box.replaceChildren(...state.datasets.map((d) => el("option", "", d.uploaded ? `${d.name} (uploaded)` : d.name)));
  [...box.options].forEach((o, i) => (o.value = state.datasets[i].name));
  if (state.datasets.some((d) => d.name === keep)) box.value = keep;
  $("reportDataset").textContent = `Dataset: ${box.value}`;
}

async function showPreview() {
  const name = $("dataset").value;
  const data = await api(`/api/datasets/${encodeURIComponent(name)}`);
  $("previewTitle").textContent = `${name} · ${data.rows} rows`;
  const body = el("div", "preview");
  const types = el("div", "types");
  for (const [col, type] of Object.entries(data.columns)) types.append(el("span", "", `${col}: ${type}`));
  const wrap = el("div", "table-wrap");
  const table = el("table");
  const head = el("tr");
  Object.keys(data.columns).forEach((c) => head.append(el("th", "", c)));
  table.append(el("thead"), el("tbody"));
  table.tHead.append(head);
  for (const row of data.sample) {
    const tr = el("tr");
    Object.keys(data.columns).forEach((c) => tr.append(el("td", "", row[c] ?? "")));
    table.tBodies[0].append(tr);
  }
  wrap.append(table);
  body.append(types, wrap, el("p", "muted", `First ${data.sample.length} of ${data.rows} rows.`));
  $("previewBody").replaceChildren(body);
  $("previewDialog").showModal();
}

async function upload(file) {
  if (file.size > 5_000_000) return toast("That file is over 5 MB.");
  try {
    const saved = await api("/api/datasets", { method: "POST",
                                               body: JSON.stringify({ name: file.name, content: await file.text() }) });
    await loadDatasets(saved.name);
    toast(`Uploaded ${saved.name}: ${saved.rows} rows, ${saved.columns} columns.`);
  } catch (err) {
    toast(`Upload failed: ${err.message}`);
  }
}

// --- conversations list -----------------------------------------------------------------

async function loadSessions() {
  const list = await api("/api/sessions");
  const nav = $("sessions");
  if (!list.length) return nav.replaceChildren(el("div", "empty-list", "No conversations yet."));
  nav.replaceChildren(...list.map((s) => {
    const row = el("div", "session" + (s.id === state.session?.id ? " active" : ""));
    const open = el("button", "open");
    open.append(el("span", "name", s.title), el("small", "", `${s.csv} · ${new Date(s.created * 1000).toLocaleDateString()}`));
    open.onclick = () => { openSession(s.id); closeMenu(); };
    const del = el("button", "del", "✕");
    del.title = "Delete conversation";
    del.setAttribute("aria-label", `Delete ${s.title}`);
    del.onclick = async () => {
      if (!confirm(`Delete "${s.title}"?`)) return;
      await api(`/api/sessions/${s.id}`, { method: "DELETE" });
      if (state.session?.id === s.id) newChatScreen();
      loadSessions();
    };
    row.append(open, del);
    return row;
  }));
}

// --- one question and everything the agent did for it ---------------------------------

class Turn {
  constructor(question) {
    this.root = el("div", "turn");
    if (question !== undefined) {
      const u = el("div", "user-msg");
      u.append(el("div", "bubble", question));
      this.root.append(u);
    }
    this.activity = el("details", "activity");
    this.summary = el("summary");
    this.steps = el("div", "steps");
    this.activity.append(this.summary, this.steps);
    this.activity.hidden = true;
    this.body = el("div");
    this.root.append(this.activity, this.body);
    this.tools = 0;
    $("thread").append(this.root);
  }

  start() {
    this.started = performance.now();
    this.activity.hidden = false;
    const tick = () => {
      const s = ((performance.now() - this.started) / 1000).toFixed(1);
      this.summary.replaceChildren(el("span", "spinner"), document.createTextNode(
        `Working… ${s}s${this.tools ? ` · ${this.tools} tool call${this.tools > 1 ? "s" : ""}` : ""}`));
    };
    tick();
    this.timer = setInterval(tick, 100);
  }

  stop(extra = "") {
    clearInterval(this.timer);
    const secs = this.started ? ` · ${((performance.now() - this.started) / 1000).toFixed(1)}s` : "";
    this.summary.textContent = `${this.tools ? `Used ${this.tools} tool call${this.tools > 1 ? "s" : ""}` : "No tools used"}${secs}${extra}`;
    this.activity.hidden = this.tools === 0;
  }

  call(name, args) {
    this.tools += 1;
    this.activity.hidden = false;
    const step = el("div", "step");
    const label = el("div", "label");
    label.append(el("span", "kind", "→ call "), document.createTextNode(name));
    step.append(label, el("pre", "", name === "run_pandas" && args.code ? args.code : pretty(args)));
    this.steps.append(step);
    scrollDown();
  }

  result(name, content) {
    const step = el("div", "step");
    const label = el("div", "label");
    label.append(el("span", "kind", "← result "), document.createTextNode(name));
    step.append(label, el("pre", "", pretty(content)));
    this.steps.append(step);
    const data = parseJson(content);
    if (name === "make_chart" && data?.path && chartUrl(data.path)) {
      const fig = el("figure", "chart");
      const img = el("img");
      img.src = chartUrl(data.path);
      img.alt = "Chart made by the agent";
      img.onload = scrollDown;
      fig.append(img);
      this.body.append(fig);
    }
    scrollDown();
  }

  answer(text) {
    const box = el("div", "answer");
    box.append(markdown(text));
    const copy = el("button", "ghost copy", "Copy");
    copy.onclick = async () => { await navigator.clipboard.writeText(text); copy.textContent = "Copied"; };
    box.append(copy);
    this.body.append(box);
    this.root.style.marginBottom = "44px";
    scrollDown();
  }

  error(message) {
    this.body.append(el("div", "callout error", message));
    scrollDown();
  }

  usage(u) {
    this.usageText = ` · ${u.input_tokens + u.output_tokens} tokens · ~$${u.cost_usd.toFixed(4)}`;
  }

  approval(request) {
    const card = el("div", "approval");
    const code = el("textarea");
    code.value = request.args.code ?? JSON.stringify(request.args, null, 2);
    code.setAttribute("aria-label", "Code the agent wants to run");
    const reason = el("input");
    reason.type = "text";
    reason.placeholder = "Reason (sent to the agent if you reject)";
    const approve = el("button", "primary", "Approve");
    const reject = el("button", "danger", "Reject");
    const row = el("div", "row");
    row.append(approve, reject, reason);
    card.append(el("h3", "", `Allow ${request.tool}?`),
                el("p", "", "The agent wrote this code and wants to run it on your data. You can edit it first."),
                code, row);
    this.body.append(card);
    setBusy(false, true);
    code.focus();
    const original = code.value;
    approve.onclick = () => {
      const edited = code.value !== original;
      card.replaceWith(el("div", "decision", edited ? "✓ You edited and approved the code." : "✓ You approved the code."));
      send("decide", edited ? { action: "edit", code: code.value } : { action: "approve" }, this);
    };
    reject.onclick = () => {
      card.replaceWith(el("div", "decision", `✕ You rejected the code${reason.value ? `: ${reason.value}` : "."}`));
      send("decide", { action: "reject", reason: reason.value || "not approved" }, this);
    };
    scrollDown();
  }

  handle(e) {
    if (e.type === "call") this.call(e.name, e.args);
    else if (e.type === "result") this.result(e.name, e.content);
    else if (e.type === "answer") this.answer(e.text);
    else if (e.type === "approval") this.approval(e.request);
    else if (e.type === "error") this.error(e.message);
    else if (e.type === "usage") this.usage(e);
  }
}

function scrollDown() {
  const t = $("thread");
  t.scrollTop = t.scrollHeight;
}

// --- chat -------------------------------------------------------------------------------

function setBusy(busy, waitingForApproval = false) {
  state.busy = busy;
  $("question").disabled = waitingForApproval;
  $("question").placeholder = waitingForApproval ? "Approve or reject the code above to continue…" : "Ask about your data…";
  $("sendBtn").textContent = busy ? "Stop" : "Send";
  $("sendBtn").classList.toggle("primary", !busy);
  $("sendBtn").disabled = waitingForApproval;
}

async function send(endpoint, body, turn) {
  state.controller = new AbortController();
  setBusy(true);
  turn.start();
  let waiting = false;
  try {
    await stream(`/api/sessions/${state.session.id}/${endpoint}`, body, (e) => {
      if (e.type === "approval") waiting = true;
      turn.handle(e);
    }, state.controller.signal);
  } catch (err) {
    if (err.name === "AbortError") turn.error("Stopped.");
    else turn.error(err.message);
  } finally {
    turn.stop(turn.usageText || "");
    turn.usageText = "";
    setBusy(false, waiting);
    if (!waiting) $("question").focus();
    loadSessions();
  }
}

async function ask(question) {
  if (!state.session) {
    state.session = await api("/api/sessions", { method: "POST", body: JSON.stringify({ csv: $("dataset").value }) });
    location.hash = state.session.id;
    $("thread").replaceChildren();
  }
  $("title").textContent = question;
  $("thread").querySelector(".welcome")?.remove();
  await send("ask", { question }, new Turn(question));
}

function welcome() {
  const box = el("div", "welcome");
  box.append(el("h1", "", "Ask your data a question"),
             el("p", "", `Using ${$("dataset").value}. The agent picks tools, runs them, and shows its work.`));
  const chips = el("div", "chips");
  for (const s of SUGGESTIONS) {
    const b = el("button", "", s);
    b.onclick = () => ask(s);
    chips.append(b);
  }
  box.append(chips);
  return box;
}

function newChatScreen() {
  state.session = null;
  history.replaceState(null, "", location.pathname);
  $("title").textContent = "New chat";
  $("thread").replaceChildren(welcome());
  setBusy(false);
  loadSessions();
  $("question").focus();
}

async function openSession(id) {
  let s;
  try { s = await api(`/api/sessions/${id}`); } catch { return newChatScreen(); }
  state.session = s;
  location.hash = id;
  $("title").textContent = s.title;
  await loadDatasets(s.csv);
  $("thread").replaceChildren();
  let turn = null;
  for (const m of s.messages) {
    if (m.role === "user") { turn?.stop(); turn = new Turn(m.text); }
    else {
      turn ??= new Turn();
      if (m.role === "call") turn.call(m.name, m.args);
      else if (m.role === "result") turn.result(m.name, m.content);
      else if (m.role === "assistant") turn.answer(m.text);
    }
  }
  turn?.stop();
  if (!s.messages.length) $("thread").append(welcome());
  setBusy(false, s.pending.length > 0);
  for (const p of s.pending) (turn ?? new Turn()).approval(p);   // a pause survives reloads and restarts
  loadSessions();
  showTab("chat");
}

// --- report -----------------------------------------------------------------------------

async function runReport() {
  const progress = $("reportProgress");
  const doc = $("reportDoc");
  doc.hidden = true;
  const card = el("div", "card");
  const plan = el("ol", "plan");
  const status = el("p", "muted", "Planning…");
  const log = el("div", "log");
  card.append(el("h2", "", "Progress"), status, plan, log);
  progress.replaceChildren(card);
  $("runReport").disabled = true;
  const rows = {};
  try {
    await stream("/api/reports", { csv: $("dataset").value, goal: $("goal").value }, (e) => {
      if (e.type === "plan") {
        status.textContent = `Plan approved automatically · ${e.tasks.length} tasks`;
        for (const t of e.tasks) {
          const li = el("li");
          li.append(el("span", "status", "○"), el("span", "worker", t.worker), el("span", "", t.task));
          rows[t.id] = li;
          plan.append(li);
        }
      } else if (e.type === "log") {
        log.append(el("div", "", e.message));
        log.scrollTop = log.scrollHeight;
        const done = e.message.match(/^(\S+) done/);
        if (done && rows[done[1]]) { rows[done[1]].classList.add("done"); rows[done[1]].firstChild.textContent = "✓"; }
        else status.textContent = e.message;
      } else if (e.type === "report") {
        status.textContent = `Status: ${e.status}`;
        if (e.markdown) showReport(e.markdown, e.totals);
      } else if (e.type === "error") {
        card.append(el("div", "callout error", e.message));
      }
    });
  } catch (err) {
    card.append(el("div", "callout error", err.message));
  } finally {
    $("runReport").disabled = false;
  }
}

function showReport(text, totals) {
  const doc = $("reportDoc");
  const bar = el("div", "toolbar");
  bar.append(el("span", "", totals?.llm_calls
    ? `${totals.llm_calls} model calls · ${totals.input_tokens + totals.output_tokens} tokens · ~$${totals.cost_usd.toFixed(4)}`
    : ""));
  const copy = el("button", "ghost", "Copy Markdown");
  copy.onclick = async () => { await navigator.clipboard.writeText(text); copy.textContent = "Copied"; };
  bar.append(copy);
  doc.replaceChildren(bar, markdown(text));
  doc.hidden = false;
  doc.scrollIntoView({ behavior: "smooth", block: "start" });
}

// --- tabs, menu, wiring -----------------------------------------------------------------

function showTab(name) {
  $("tabChat").setAttribute("aria-selected", name === "chat");
  $("tabReport").setAttribute("aria-selected", name === "report");
  $("chatView").hidden = name !== "chat";
  $("reportView").hidden = name !== "report";
  if (name === "report") $("title").textContent = "Autonomous report";
  else $("title").textContent = state.session?.title ?? "New chat";
}

const closeMenu = () => $("app").classList.remove("menu-open");

function wire() {
  $("tabChat").onclick = () => showTab("chat");
  $("tabReport").onclick = () => showTab("report");
  $("newChat").onclick = () => { showTab("chat"); newChatScreen(); closeMenu(); };
  $("previewBtn").onclick = () => showPreview().catch((e) => toast(e.message));
  $("closePreview").onclick = () => $("previewDialog").close();
  $("uploadBtn").onclick = () => $("fileInput").click();
  $("fileInput").onchange = (e) => { if (e.target.files[0]) upload(e.target.files[0]); e.target.value = ""; };
  $("openSidebar").onclick = () => $("app").classList.add("menu-open");
  $("closeSidebar").onclick = closeMenu;
  $("runReport").onclick = runReport;
  $("dataset").onchange = () => {
    $("reportDataset").textContent = `Dataset: ${$("dataset").value}`;
    if (!state.session) $("thread").replaceChildren(welcome());
  };

  const q = $("question");
  q.addEventListener("input", () => { q.style.height = "auto"; q.style.height = `${q.scrollHeight}px`; });
  q.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("composer").requestSubmit(); }
  });
  $("composer").onsubmit = (e) => {
    e.preventDefault();
    if (state.busy) return state.controller?.abort();
    const text = q.value.trim();
    if (!text) return;
    q.value = "";
    q.style.height = "auto";
    ask(text).catch((err) => toast(err.message));
  };
}

wire();
await loadDatasets();
if (location.hash.length > 1) await openSession(location.hash.slice(1));
else newChatScreen();
