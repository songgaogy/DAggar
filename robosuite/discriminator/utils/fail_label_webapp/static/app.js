"use strict";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  task: null,
  file: null,
  cameras: [],
  cam: null,
  demos: [],          // [{demo,length,labeled,label}]
  demoIndex: -1,      // index into filtered list
  demo: null,
  length: 0,
  cur: 0,             // current frame
  playing: false,
  speed: 1,
  fps: 20,
  cache: new Map(),   // "demo|cam|idx" -> HTMLImageElement
  lastMode: "drop",   // persist mode across unlabeled demos
};

const prefetch = {
  generation: 0,
  priorityDemo: null,
  running: false,
  restartAfterRun: false,
  concurrency: 8,
  jobs: [],
  jobCursor: 0,
  totalFrames: 0,
};

// ---------------------------------------------------------------------------
// DOM
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const el = {
  task: $("task-select"),
  file: $("file-select"),
  progress: $("progress"),
  exportBtn: $("export-btn"),
  demoFilter: $("demo-filter"),
  demoList: $("demo-list"),
  demoTitle: $("demo-title"),
  cam: $("cam-select"),
  viewport: $("viewport"),
  frameImg: $("frame-img"),
  loading: $("loading"),
  loadPct: $("load-pct"),
  segHighlight: $("seg-highlight"),
  slider: $("frame-slider"),
  prev: $("prev-btn"),
  play: $("play-btn"),
  next: $("next-btn"),
  curFrame: $("cur-frame"),
  maxFrame: $("max-frame"),
  speed: $("speed-select"),
  setStart: $("set-start-btn"),
  startInput: $("start-input"),
  endInput: $("end-input"),
  endLast: $("end-last-btn"),
  modeSelect: $("mode-select"),
  modeCustom: $("mode-custom"),
  save: $("save-btn"),
  clear: $("clear-btn"),
  prevDemo: $("prev-demo-btn"),
  nextDemo: $("next-demo-btn"),
  curLabel: $("cur-label"),
  toast: $("toast"),
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function frameUrl(demo, cam, idx) {
  const p = new URLSearchParams({
    task: state.task, file: state.file, demo, cam, idx: String(idx),
  });
  return `/api/frame?${p.toString()}`;
}

let toastTimer = null;
function toast(msg, kind = "") {
  el.toast.textContent = msg;
  el.toast.className = "toast " + kind;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.toast.classList.add("hidden"), 2600);
}

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
async function postJSON(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

function filteredDemos() {
  const q = el.demoFilter.value.trim().toLowerCase();
  if (!q) return state.demos;
  return state.demos.filter((d) => {
    const m = d.label && d.label.mode ? d.label.mode : "";
    return d.demo.toLowerCase().includes(q) || m.toLowerCase().includes(q);
  });
}

// ---------------------------------------------------------------------------
// Loaders
// ---------------------------------------------------------------------------
async function loadTasks() {
  const { tasks } = await getJSON("/api/tasks");
  el.task.innerHTML = "";
  tasks.forEach((t) => {
    const o = document.createElement("option");
    o.value = t; o.textContent = t; el.task.appendChild(o);
  });
  if (tasks.length) {
    state.task = tasks[0];
    el.task.value = state.task;
    await loadFiles();
  }
}

async function loadFiles() {
  const { files } = await getJSON(`/api/files?task=${encodeURIComponent(state.task)}`);
  el.file.innerHTML = "";
  files.forEach((f) => {
    const o = document.createElement("option");
    o.value = f; o.textContent = f; el.file.appendChild(o);
  });
  if (files.length) {
    state.file = files[0];
    el.file.value = state.file;
    await loadDemos();
  } else {
    state.file = null;
    el.demoList.innerHTML = "";
  }
}

async function loadDemos() {
  const url = `/api/demos?task=${encodeURIComponent(state.task)}&file=${encodeURIComponent(state.file)}`;
  const data = await getJSON(url);
  state.demos = data.demos;
  state.cameras = data.cameras;
  state.cam = data.default_camera || (data.cameras[0] || null);
  // camera select
  el.cam.innerHTML = "";
  state.cameras.forEach((c) => {
    const o = document.createElement("option");
    o.value = c; o.textContent = c; el.cam.appendChild(o);
  });
  if (state.cam) el.cam.value = state.cam;
  updateProgress(data.n_labeled, data.n_total);
  renderDemoList();
  startContinuousPrefetch();
  // auto-select first demo
  const list = filteredDemos();
  if (list.length) selectDemo(0);
}

function updateProgress(n, total) {
  el.progress.textContent = `Labeled ${n} / ${total}`;
}

function renderDemoList() {
  const list = filteredDemos();
  el.demoList.innerHTML = "";
  list.forEach((d, i) => {
    const li = document.createElement("li");
    li.className = (d.labeled ? "labeled " : "") + (i === state.demoIndex ? "active" : "");
    const name = document.createElement("span");
    name.className = "mname";
    name.textContent = d.demo.replace("demo_", "#");
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = d.labeled
      ? (d.label.mode || "labeled") + ` [${d.label.start}-${d.label.end}]`
      : "unlabeled";
    li.appendChild(name);
    li.appendChild(tag);
    li.addEventListener("click", () => selectDemo(i));
    el.demoList.appendChild(li);
  });
}

async function selectDemo(filteredIdx) {
  const list = filteredDemos();
  if (filteredIdx < 0 || filteredIdx >= list.length) return;
  stopPlay();
  state.demoIndex = filteredIdx;
  const d = list[filteredIdx];
  state.demo = d.demo;
  state.length = d.length;
  state.cur = 0;
  el.demoTitle.textContent = `${d.demo}  (len=${d.length})`;
  el.slider.max = String(Math.max(0, d.length - 1));
  el.slider.value = "0";
  el.maxFrame.textContent = String(Math.max(0, d.length - 1));
  renderDemoList();
  loadLabelToPanel(d);
  showFrame(0);
  bumpPrefetchPriority(d.demo);
}

async function selectNextDemoInTask() {
  const cur = state.demos.findIndex((x) => x.demo === state.demo);
  if (cur < 0 || cur >= state.demos.length - 1) return;
  const nextDemo = state.demos[cur + 1].demo;
  const filteredIdx = filteredDemos().findIndex((x) => x.demo === nextDemo);
  if (filteredIdx >= 0) await selectDemo(filteredIdx);
}

function demoCacheKey(demo, cam, idx) {
  return `${demo}|${cam}|${idx}`;
}

function cacheKey(idx) {
  return demoCacheKey(state.demo, state.cam, idx);
}

function isFrameCached(demo, cam, idx) {
  const img = state.cache.get(demoCacheKey(demo, cam, idx));
  return !!(img && img.complete);
}

function countCachedFrames(cam = state.cam) {
  if (!cam) return 0;
  let n = 0;
  for (const d of state.demos) {
    for (let i = 0; i < d.length; i++) {
      if (isFrameCached(d.demo, cam, i)) n++;
    }
  }
  return n;
}

function orderedDemosForPrefetch() {
  const demos = state.demos;
  if (!prefetch.priorityDemo) return demos;
  const pri = demos.find((d) => d.demo === prefetch.priorityDemo);
  if (!pri) return demos;
  return [pri, ...demos.filter((d) => d.demo !== prefetch.priorityDemo)];
}

function rebuildPrefetchJobs() {
  const cam = state.cam;
  prefetch.jobs = [];
  prefetch.jobCursor = 0;
  prefetch.totalFrames = state.demos.reduce((s, d) => s + d.length, 0);
  if (!cam || !prefetch.totalFrames) return;
  for (const d of orderedDemosForPrefetch()) {
    for (let i = 0; i < d.length; i++) {
      if (!isFrameCached(d.demo, cam, i)) {
        prefetch.jobs.push({ demo: d.demo, idx: i });
      }
    }
  }
}

function updatePrefetchProgress() {
  const total = prefetch.totalFrames;
  if (!total) {
    el.loading.classList.add("hidden");
    return;
  }
  const done = countCachedFrames();
  el.loadPct.textContent = String(Math.round((done / total) * 100));
  if (done >= total) {
    setTimeout(() => el.loading.classList.add("hidden"), 300);
  } else {
    el.loading.classList.remove("hidden");
  }
}

function resetPrefetch(clearCache = false) {
  prefetch.generation++;
  prefetch.priorityDemo = null;
  prefetch.jobs = [];
  prefetch.jobCursor = 0;
  prefetch.totalFrames = 0;
  prefetch.restartAfterRun = false;
  if (clearCache) state.cache.clear();
}

function startContinuousPrefetch() {
  rebuildPrefetchJobs();
  updatePrefetchProgress();
  kickPrefetchWorkers();
}

function bumpPrefetchPriority(demo) {
  prefetch.priorityDemo = demo;
  prefetch.generation++;
  rebuildPrefetchJobs();
  updatePrefetchProgress();
  kickPrefetchWorkers();
}

async function kickPrefetchWorkers() {
  if (!state.cam || !prefetch.jobs.length) return;
  if (prefetch.running) {
    prefetch.restartAfterRun = true;
    return;
  }
  prefetch.running = true;
  prefetch.restartAfterRun = false;
  const gen = prefetch.generation;
  el.loading.classList.remove("hidden");

  async function worker() {
    while (gen === prefetch.generation) {
      const jobIdx = prefetch.jobCursor++;
      if (jobIdx >= prefetch.jobs.length) break;
      const job = prefetch.jobs[jobIdx];
      const cam = state.cam;
      const key = demoCacheKey(job.demo, cam, job.idx);
      if (isFrameCached(job.demo, cam, job.idx)) {
        if (jobIdx % 20 === 0) updatePrefetchProgress();
        continue;
      }
      await new Promise((resolve) => {
        const img = new Image();
        img.onload = img.onerror = () => resolve();
        img.src = frameUrl(job.demo, cam, job.idx);
        state.cache.set(key, img);
      });
      if (gen !== prefetch.generation) break;
      if (jobIdx % 10 === 0) updatePrefetchProgress();
    }
  }

  await Promise.all(Array.from({ length: prefetch.concurrency }, worker));
  prefetch.running = false;
  rebuildPrefetchJobs();
  updatePrefetchProgress();
  if (prefetch.restartAfterRun || prefetch.jobs.length) {
    prefetch.restartAfterRun = false;
    kickPrefetchWorkers();
  }
}

function showFrame(idx) {
  if (state.length <= 0) return;
  idx = Math.max(0, Math.min(state.length - 1, idx));
  state.cur = idx;
  const key = cacheKey(idx);
  const cached = state.cache.get(key);
  if (cached && cached.complete) {
    el.frameImg.src = cached.src;
  } else {
    el.frameImg.src = frameUrl(state.demo, state.cam, idx);
  }
  el.slider.value = String(idx);
  el.curFrame.textContent = String(idx);
  syncStartFromFrame();
}

function syncStartFromFrame() {
  const d = state.demos.find((x) => x.demo === state.demo);
  if (!d || d.labeled) return;
  el.startInput.value = String(state.cur);
  updateSegHighlight();
}

// ---------------------------------------------------------------------------
// Playback
// ---------------------------------------------------------------------------
let playTimer = null;
function startPlay() {
  if (state.playing || state.length <= 0) return;
  state.playing = true;
  el.play.textContent = "⏸ Pause";
  const tick = () => {
    if (!state.playing) return;
    if (state.cur >= state.length - 1) { stopPlay(); return; }
    showFrame(state.cur + 1);
    playTimer = setTimeout(tick, 1000 / (state.fps * state.speed));
  };
  playTimer = setTimeout(tick, 1000 / (state.fps * state.speed));
}
function stopPlay() {
  state.playing = false;
  el.play.textContent = "▶ Play";
  if (playTimer) { clearTimeout(playTimer); playTimer = null; }
}
function togglePlay() { state.playing ? stopPlay() : startPlay(); }

// ---------------------------------------------------------------------------
// Viewport scrub (maps horizontal position to frame slider)
// ---------------------------------------------------------------------------
let viewportScrubbing = false;

function frameFromViewportX(clientX) {
  if (state.length <= 0) return 0;
  const rect = el.viewport.getBoundingClientRect();
  const w = rect.width;
  if (w <= 0) return state.cur;
  const x = Math.max(0, Math.min(w, clientX - rect.left));
  const max = Math.max(0, state.length - 1);
  if (max === 0) return 0;
  return Math.round((x / w) * max);
}

function scrubViewportTo(clientX) {
  stopPlay();
  showFrame(frameFromViewportX(clientX));
}

function endViewportScrub() {
  if (!viewportScrubbing) return;
  viewportScrubbing = false;
  el.viewport.classList.remove("scrubbing");
}

el.viewport.addEventListener("pointerdown", (e) => {
  if (e.button !== 0 || state.length <= 0) return;
  e.preventDefault();
  viewportScrubbing = true;
  el.viewport.classList.add("scrubbing");
  el.viewport.setPointerCapture(e.pointerId);
  scrubViewportTo(e.clientX);
});

el.viewport.addEventListener("pointermove", (e) => {
  if (!viewportScrubbing) return;
  scrubViewportTo(e.clientX);
});

el.viewport.addEventListener("pointerup", endViewportScrub);
el.viewport.addEventListener("pointercancel", endViewportScrub);

// ---------------------------------------------------------------------------
// Labeling
// ---------------------------------------------------------------------------
function currentMode() {
  if (el.modeSelect.value === "__custom__") return el.modeCustom.value.trim();
  return el.modeSelect.value;
}

function rememberMode(mode) {
  const m = (mode || "").trim();
  if (m) state.lastMode = m;
}

function setModeUI(mode) {
  const known = ["drop", "grasp_failure", "wrong_place"];
  if (mode && !known.includes(mode)) {
    el.modeSelect.value = "__custom__";
    el.modeCustom.classList.remove("hidden");
    el.modeCustom.value = mode;
  } else {
    el.modeSelect.value = mode || state.lastMode || "drop";
    el.modeCustom.classList.add("hidden");
    el.modeCustom.value = "";
  }
}

function loadLabelToPanel(d) {
  if (d.label) {
    el.startInput.value = d.label.start;
    el.endInput.value = d.label.end;
    setModeUI(d.label.mode);
    rememberMode(d.label.mode);
    el.curLabel.textContent = `Labeled: start=${d.label.start}, end=${d.label.end}, mode=${d.label.mode || "(empty)"}`;
    el.curLabel.classList.add("has");
  } else {
    el.startInput.value = state.cur;
    el.endInput.value = Math.max(0, state.length - 1);
    setModeUI(state.lastMode);
    el.curLabel.textContent = "No label for this demo";
    el.curLabel.classList.remove("has");
  }
  updateSegHighlight();
}

function updateSegHighlight() {
  const s = parseInt(el.startInput.value, 10);
  const e = parseInt(el.endInput.value, 10);
  const n = state.length;
  if (!Number.isFinite(s) || !Number.isFinite(e) || n <= 1) {
    el.segHighlight.classList.add("hidden");
    return;
  }
  const lo = Math.max(0, Math.min(s, e));
  const hi = Math.min(n - 1, Math.max(s, e));
  const leftPct = (lo / (n - 1)) * 100;
  const widthPct = ((hi - lo) / (n - 1)) * 100;
  el.segHighlight.style.left = `calc(${leftPct}% )`;
  el.segHighlight.style.width = `calc(${widthPct}% )`;
  el.segHighlight.classList.remove("hidden");
}

async function saveLabel() {
  if (!state.demo) return;
  const start = parseInt(el.startInput.value, 10);
  const end = parseInt(el.endInput.value, 10);
  const mode = currentMode();
  try {
    const res = await postJSON("/api/label", {
      task: state.task, file: state.file, demo: state.demo,
      start, end, mode,
    });
    // update local state
    const d = state.demos.find((x) => x.demo === state.demo);
    if (d) { d.labeled = true; d.label = res.label; }
    rememberMode(mode);
    updateProgress(res.n_labeled, state.demos.length);
    renderDemoList();
    toast("Saved", "ok");
    await selectNextDemoInTask();
  } catch (err) {
    toast("Save failed: " + err.message, "err");
  }
}

async function clearLabel() {
  if (!state.demo) return;
  try {
    const res = await postJSON("/api/delete_label", {
      task: state.task, file: state.file, demo: state.demo,
    });
    const d = state.demos.find((x) => x.demo === state.demo);
    if (d) { d.labeled = false; d.label = null; }
    updateProgress(res.n_labeled, state.demos.length);
    loadLabelToPanel(d);
    renderDemoList();
    toast("Cleared", "ok");
  } catch (err) {
    toast("Clear failed: " + err.message, "err");
  }
}

async function doExport() {
  el.exportBtn.disabled = true;
  el.exportBtn.textContent = "Exporting...";
  try {
    const res = await postJSON("/api/export", { task: state.task, file: state.file });
    toast(`Exported ${res.n_written} demo(s) -> ${res.output_path}`, "ok");
  } catch (err) {
    toast("Export failed: " + err.message, "err");
  } finally {
    el.exportBtn.disabled = false;
    el.exportBtn.textContent = "Export labeled HDF5";
  }
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------
el.task.addEventListener("change", async () => {
  state.task = el.task.value;
  resetPrefetch(true);
  await loadFiles();
});
el.file.addEventListener("change", async () => {
  state.file = el.file.value;
  resetPrefetch(true);
  await loadDemos();
});
el.cam.addEventListener("change", () => {
  state.cam = el.cam.value;
  showFrame(state.cur);
  prefetch.generation++;
  startContinuousPrefetch();
});
el.demoFilter.addEventListener("input", () => renderDemoList());

el.slider.addEventListener("input", () => { stopPlay(); showFrame(parseInt(el.slider.value, 10)); });
el.prev.addEventListener("click", () => { stopPlay(); showFrame(state.cur - 1); });
el.next.addEventListener("click", () => { stopPlay(); showFrame(state.cur + 1); });
el.play.addEventListener("click", togglePlay);
el.speed.addEventListener("change", () => {
  state.speed = parseFloat(el.speed.value);
  if (state.playing) { stopPlay(); startPlay(); }
});

el.setStart.addEventListener("click", () => {
  el.startInput.value = String(state.cur);
  if (parseInt(el.endInput.value, 10) < state.cur) {
    el.endInput.value = String(Math.max(0, state.length - 1));
  }
  updateSegHighlight();
});
el.endLast.addEventListener("click", () => {
  el.endInput.value = String(Math.max(0, state.length - 1));
  updateSegHighlight();
});
el.startInput.addEventListener("input", updateSegHighlight);
el.endInput.addEventListener("input", updateSegHighlight);
el.modeSelect.addEventListener("change", () => {
  if (el.modeSelect.value === "__custom__") {
    el.modeCustom.classList.remove("hidden");
    el.modeCustom.focus();
  } else {
    el.modeCustom.classList.add("hidden");
    rememberMode(el.modeSelect.value);
  }
});
el.modeCustom.addEventListener("input", () => rememberMode(el.modeCustom.value));
el.save.addEventListener("click", saveLabel);
el.clear.addEventListener("click", clearLabel);
el.prevDemo.addEventListener("click", () => selectDemo(state.demoIndex - 1));
el.nextDemo.addEventListener("click", () => selectDemo(state.demoIndex + 1));
el.exportBtn.addEventListener("click", doExport);

// keyboard shortcuts
document.addEventListener("keydown", (e) => {
  const tag = (e.target.tagName || "").toLowerCase();
  const typing = tag === "input" || tag === "select" || tag === "textarea";
  if (typing && e.key !== "Escape") return;
  switch (e.key) {
    case " ": e.preventDefault(); togglePlay(); break;
    case "ArrowLeft": e.preventDefault(); stopPlay(); showFrame(state.cur - 1); break;
    case "ArrowRight": e.preventDefault(); stopPlay(); showFrame(state.cur + 1); break;
    case "f": case "F": el.setStart.click(); break;
    case "s": case "S": saveLabel(); break;
    case "n": case "N": selectDemo(state.demoIndex + 1); break;
    case "p": case "P": selectDemo(state.demoIndex - 1); break;
  }
});

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
loadTasks().catch((err) => toast("Init failed: " + err.message, "err"));
