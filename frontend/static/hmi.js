/* PaprikaVisionMDE HMI client
 *
 * Polls /result and /status and paints the rail. The camera feed carries its
 * own overlay from the server, so nothing here tries to position anything over
 * the video - see the note in frontend/web.py for why the boxes are burned
 * into the JPEG instead of layered in the DOM.
 */

const POLL_MS = 250;
const STATUS_MS = 3000;

let OVERLAY_RUNNING = true;

/* ------------------------------------------------------------------ utils */

const $ = (id) => document.getElementById(id);

function toast(message, isError = false) {
  const el = $("toast");
  el.textContent = message;
  el.classList.toggle("is-error", isError);
  el.hidden = false;
  clearTimeout(el._timer);
  el._timer = setTimeout(() => { el.hidden = true; }, 3500);
}

async function post(url, body) {
  const res = await fetch(url, {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  return res;
}

/* ------------------------------------------------------------- angle dial
 *
 * A number alone does not tell an operator whether 137 is right. The needle
 * next to the fruit does: they can look at the belt, look at the dial, and see
 * agreement or disagreement immediately, without knowing the convention.
 *
 * Screen convention, matching the engine: 0 points right, 90 points up,
 * counter-clockwise positive. Canvas y grows downward, hence the negated sine.
 */
function drawDial(angleDeg, placement) {
  const canvas = $("angleDial");
  const ctx = canvas.getContext("2d");
  const size = canvas.width;
  const cx = size / 2;
  const cy = size / 2;
  const radius = size / 2 - 12;

  const colors = {
    place: "#46be5a",
    reorient: "#ebb43c",
    reject: "#dc3c3c",
    unknown: "#93a1b3",
  };
  const color = colors[placement] || colors.unknown;

  ctx.clearRect(0, 0, size, size);

  ctx.strokeStyle = "#313b49";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.stroke();

  // Ticks every 30 degrees, longer at the cardinals, so the needle can be read
  // to roughly the nearest tick without squinting at the number.
  for (let deg = 0; deg < 360; deg += 30) {
    const rad = (deg * Math.PI) / 180;
    const isCardinal = deg % 90 === 0;
    const inner = radius - (isCardinal ? 11 : 6);
    ctx.strokeStyle = isCardinal ? "#546074" : "#313b49";
    ctx.lineWidth = isCardinal ? 2 : 1;
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(rad) * inner, cy - Math.sin(rad) * inner);
    ctx.lineTo(cx + Math.cos(rad) * radius, cy - Math.sin(rad) * radius);
    ctx.stroke();
  }

  if (angleDeg === null || angleDeg === undefined) {
    ctx.fillStyle = "#313b49";
    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, Math.PI * 2);
    ctx.fill();
    return;
  }

  const rad = (angleDeg * Math.PI) / 180;
  const tipX = cx + Math.cos(rad) * (radius - 16);
  const tipY = cy - Math.sin(rad) * (radius - 16);

  ctx.strokeStyle = color;
  ctx.lineWidth = 5;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(cx - Math.cos(rad) * 22, cy + Math.sin(rad) * 22);
  ctx.lineTo(tipX, tipY);
  ctx.stroke();

  // Arrowhead, so which end is the stem is unmistakable.
  const head = 11;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(tipX, tipY);
  ctx.lineTo(
    tipX - Math.cos(rad - 0.42) * head,
    tipY + Math.sin(rad - 0.42) * head
  );
  ctx.lineTo(
    tipX - Math.cos(rad + 0.42) * head,
    tipY + Math.sin(rad + 0.42) * head
  );
  ctx.closePath();
  ctx.fill();

  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(cx, cy, 5, 0, Math.PI * 2);
  ctx.fill();
}

/* ---------------------------------------------------------------- meters */

function setMeter(barId, valId, value) {
  const bar = $(barId);
  const val = $(valId);
  if (value === null || value === undefined) {
    bar.style.width = "0%";
    bar.className = "bar-fill";
    val.textContent = "—";
    return;
  }
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100);
  bar.style.width = pct + "%";
  bar.className = "bar-fill" + (pct < 40 ? " is-bad" : pct < 65 ? " is-low" : "");
  val.textContent = pct + "%";
}

/* ----------------------------------------------------------------- render */

function renderLive(live) {
  const card = $("angleCard");
  const primary = live && live.primary;

  $("factCount").textContent = live ? live.count : 0;
  $("factScan").textContent = live && live.scan_ms ? `${Math.round(live.scan_ms)} ms` : "— ms";

  card.classList.remove("is-place", "is-reorient", "is-reject");

  if (!primary || live.stale) {
    $("angleNumber").textContent = "—";
    $("placementText").textContent = live && live.stale ? "no recent result" : "waiting for fruit";
    $("poseText").textContent = "—";
    $("factSource").textContent = "—";
    setMeter("barAxis", "valAxis", null);
    setMeter("barFlip", "valFlip", null);
    drawDial(null, "unknown");
    return;
  }

  const orientation = primary.orientation || {};
  const placement = primary.placement || "unknown";

  card.classList.add("is-" + placement);

  const angle = primary.angle_plc;
  $("angleNumber").textContent = angle === null || angle === undefined
    ? "—"
    : Math.round(angle);

  const poseLabels = {
    upside_down: "upside down — cannot be picked",
    stem_not_found: "stem not found — nothing to measure",
    incomplete: "incomplete in frame",
    standing_stem_up: "standing, stem up",
    standing_stem_down: "standing, stem down",
  };
  $("placementText").textContent = poseLabels[orientation.pose] || placement;
  $("poseText").textContent = orientation.pose ? orientation.pose.replace(/_/g, " ") : "—";
  $("factSource").textContent = orientation.source || "—";

  setMeter("barAxis", "valAxis", orientation.confidence);
  setMeter("barFlip", "valFlip", orientation.flip_confidence);

  // The dial shows the vision-frame angle, not the PLC-frame one: it is drawn
  // alongside the video, so it has to agree with the arrow burned into the
  // video, which is also in the vision frame.
  drawDial(orientation.angle_deg, placement);
}

function renderCounters(counters) {
  if (!counters) return;
  $("cntOk").textContent = counters.ok ?? 0;
  $("cntReorient").textContent = counters.reorient ?? 0;
  $("cntUpside").textContent = counters.upside_down ?? 0;
  $("cntIncomplete").textContent = counters.incomplete ?? 0;
  $("cntTotal").textContent = counters.total ?? 0;
}

/* ---------------------------------------------------------------- polling */

async function pollResult() {
  try {
    const res = await fetch("/result");
    const data = await res.json();
    renderLive(data.live);
    renderCounters(data.counters);
    $("cameraDown").hidden = true;
  } catch (err) {
    $("cameraDown").hidden = false;
  } finally {
    setTimeout(pollResult, POLL_MS);
  }
}

async function pollStatus() {
  try {
    const res = await fetch("/status", { credentials: "same-origin" });
    const data = await res.json();

    $("machineId").textContent = data.machine_id || "—";
    refreshSource();
    $("factBackend").textContent = (data.engine && data.engine.backend) || "—";

    if (data.overlay) {
      OVERLAY_RUNNING = !!data.overlay.running;
      $("overlayBtn").textContent = OVERLAY_RUNNING ? "Pause overlay" : "Resume overlay";
    }

    if (data.engine && data.engine.ready === false) {
      $("placementText").textContent = "detector not ready";
    }
  } catch (err) {
    /* transient - the next poll will pick it up */
  } finally {
    setTimeout(pollStatus, STATUS_MS);
  }
}

/* ---------------------------------------------------------------- actions */

$("overlayBtn").addEventListener("click", async () => {
  const res = await post("/overlay/toggle");
  if (!res.ok) return toast("Could not toggle overlay.", true);
  const data = await res.json();
  OVERLAY_RUNNING = data.running;
  $("overlayBtn").textContent = OVERLAY_RUNNING ? "Pause overlay" : "Resume overlay";
});

$("rotateBtn").addEventListener("click", async () => {
  const res = await post("/camera_rotation");
  if (!res.ok) toast("Could not rotate camera.", true);
});

$("resetBtn").addEventListener("click", async () => {
  const res = await post("/reset_counters");
  if (res.ok) toast("Counters reset.");
  else toast("Could not reset counters.", true);
});

/* ----------------------------------------------------------- file source
 *
 * The buttons only appear when the backend is playing a folder. With a real
 * camera the endpoint does not exist and the bar stays empty, so there is never
 * a button on screen that does nothing.
 */

let IS_FOLDER = false;

async function refreshSource() {
  try {
    const data = await (await fetch("/source")).json();
    if (data.source !== "folder") return;
    IS_FOLDER = true;
    $("sourceControls").hidden = false;
    $("factImage").textContent = `${data.index + 1}/${data.count}`;
    $("factImage").title = data.name || "";
    $("holdBtn").textContent = data.hold ? "Resume" : "Hold";
  } catch (err) {
    /* not a file source */
  }
}

async function stepSource(url) {
  const res = await post(url);
  if (!res.ok) return toast("Could not switch image.", true);
  const data = await res.json();
  $("factImage").textContent = `${data.index + 1}/${data.count}`;
  $("factImage").title = data.name || "";
  $("holdBtn").textContent = data.hold ? "Resume" : "Hold";
}

$("nextBtn").addEventListener("click", () => stepSource("/source/next"));
$("prevBtn").addEventListener("click", () => stepSource("/source/previous"));
$("holdBtn").addEventListener("click", () => stepSource("/source/hold"));

// Arrow keys: faster than clicking when stepping through 160 images.
document.addEventListener("keydown", (event) => {
  if (!IS_FOLDER) return;
  if (event.target.tagName === "INPUT") return;
  if (event.key === "ArrowRight") stepSource("/source/next");
  if (event.key === "ArrowLeft") stepSource("/source/previous");
});

/* ------------------------------------------------------------------- boot */

drawDial(null, "unknown");
refreshSource();
pollStatus();
pollResult();
