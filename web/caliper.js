// Caliper — in-browser facial geometry + skin, in plain language.
// Mirrors the Python core (same math, same interpretation). Everything runs locally;
// the only network calls are this page, the WASM runtime, and the Apache-2.0 model.
import { FaceLandmarker, FilesetResolver }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs";

const IRIS_MM = 11.7, ERR_PCT = 4.3;
const LEFT_IRIS = 468, LEFT_RING = [469, 470, 471, 472];
const RIGHT_IRIS = 473, RIGHT_RING = [474, 475, 476, 477];
const IDX = {
  menton: 152, forehead_top: 10, glabella: 9, nasion: 168, subnasale: 2, nose_tip: 1,
  r_canthus_out: 33, r_canthus_in: 133, l_canthus_in: 362, l_canthus_out: 263,
  alare_r: 48, alare_l: 278, zygo_r: 234, zygo_l: 454, cheilion_r: 61, cheilion_l: 291,
  brow_r: 105, brow_l: 334, lip_top: 0,
};
const SKIN_REGIONS = { forehead: 10, glabella: 9, right_cheek: 50, left_cheek: 280, nose: 195, chin: 200 };
const ORDER = ["canthal_tilt", "intercanthal_width", "fifth_intercanthal", "eye_fissure_length",
  "nasal_width", "nasal_height", "nasal_index", "mouth_width", "bizygomatic_width",
  "facial_height", "facial_index", "third_middle", "third_lower", "third_upper", "fwhr", "ipd"];
const PIGMENT_FIRST = new Set(["east_asian", "south_asian", "african", "african_american", "middle_eastern"]);
const ITA_CEIL = { african: 25, african_american: 30, south_asian: 45, east_asian: 50, middle_eastern: 50, european: 66 };
const PHOTOTYPE = {
  east_asian: "III–IV (light-medium)", south_asian: "IV–V", african: "V–VI",
  african_american: "IV–VI", middle_eastern: "III–V", european: "I–III (varies widely)",
};

let landmarker = null, NORMS = null, GEN = null, IVS = null;
const $ = (id) => document.getElementById(id);
const status = (m) => { $("status").innerHTML = m; };

// --- math ----------------------------------------------------------------
const hypot = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const erf = (x) => {
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
    - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return x >= 0 ? y : -y;
};
const pctile = (z) => 50 * (1 + erf(z / Math.SQRT2));
const ordinal = (n) => {
  const s = ["th", "st", "nd", "rd"], v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
};
const median = (a) => { const b = [...a].sort((p, q) => p - q); return b[b.length >> 1]; };
const medRGB = (px) => [0, 1, 2].map((c) => median(px.map((p) => p[c])));

function levelRoll(P) {
  const r = P[RIGHT_IRIS], l = P[LEFT_IRIS];
  let ang = Math.atan2(l[1] - r[1], l[0] - r[0]);
  if (ang > Math.PI / 2) ang -= Math.PI;        // fold to nearest horizontal — never flip
  else if (ang < -Math.PI / 2) ang += Math.PI;
  const c = Math.cos(-ang), s = Math.sin(-ang);
  const cx = (r[0] + l[0]) / 2, cy = (r[1] + l[1]) / 2;
  return P.map(([x, y]) => {
    const dx = x - cx, dy = y - cy;
    return [cx + dx * c - dy * s, cy + dx * s + dy * c];
  });
}
function irisDiameter(P, center, ring) {
  const c = P[center];
  return 2 * ring.reduce((s, i) => s + hypot(P[i], c), 0) / ring.length;
}
function calibrate(P) {
  const iris = (irisDiameter(P, LEFT_IRIS, LEFT_RING) + irisDiameter(P, RIGHT_IRIS, RIGHT_RING)) / 2;
  const mm = IRIS_MM / iris;
  const impliedIpd = hypot(P[LEFT_IRIS], P[RIGHT_IRIS]) * mm;
  return { mm, impliedIpd, plausible: impliedIpd >= 54 && impliedIpd <= 72 };
}
const tilt = (inner, outer) => Math.atan2(inner[1] - outer[1], Math.abs(outer[0] - inner[0])) * 180 / Math.PI;

function metrics(Praw, mm) {
  const P = levelRoll(Praw);
  const d = (a, b) => hypot(P[IDX[a]], P[IDX[b]]);
  const M = {};
  const add = (id, value) => { M[id] = value; };
  add("canthal_tilt", (tilt(P[IDX.r_canthus_in], P[IDX.r_canthus_out]) + tilt(P[IDX.l_canthus_in], P[IDX.l_canthus_out])) / 2);
  add("intercanthal_width", d("l_canthus_in", "r_canthus_in") * mm);
  add("eye_fissure_length", (d("r_canthus_out", "r_canthus_in") + d("l_canthus_out", "l_canthus_in")) / 2 * mm);
  const nw = d("alare_l", "alare_r") * mm, nh = d("nasion", "subnasale") * mm;
  add("nasal_width", nw); add("nasal_height", nh); add("nasal_index", nw / nh * 100);
  add("mouth_width", d("cheilion_l", "cheilion_r") * mm);
  const bz = d("zygo_l", "zygo_r") * mm, fh = d("nasion", "menton") * mm;
  add("bizygomatic_width", bz); add("facial_height", fh); add("facial_index", fh / bz * 100);
  add("ipd", hypot(P[LEFT_IRIS], P[RIGHT_IRIS]) * mm);
  const upY = P[IDX.glabella][1] - P[IDX.forehead_top][1];
  const midY = P[IDX.subnasale][1] - P[IDX.glabella][1];
  const lowY = P[IDX.menton][1] - P[IDX.subnasale][1];
  const tot = upY + midY + lowY;
  add("third_upper", 100 * upY / tot); add("third_middle", 100 * midY / tot); add("third_lower", 100 * lowY / tot);
  const fw = Math.abs(P[IDX.zygo_l][0] - P[IDX.zygo_r][0]);
  const unit = fw / 5;
  add("fifth_intercanthal", Math.abs(P[IDX.l_canthus_in][0] - P[IDX.r_canthus_in][0]) / unit);
  add("fwhr", fw / (P[IDX.lip_top][1] - Math.min(P[IDX.brow_r][1], P[IDX.brow_l][1])));
  return M;
}

// Evidence-graded mean shift for ages outside the 18-45 band (mirrors norms.py _age_shift).
function ageShift(id, age, sex) {
  const ae = NORMS.age_effects;
  if (!ae || age == null) return null;
  const eff = (ae.metrics || {})[id];
  if (!eff) return null;
  const ref = ae._ref_age != null ? ae._ref_age : 31;
  let shift = eff.per_year_mm * (age - ref);
  const m = ae._menopause || {};
  if (m.sex && sex === m.sex && age > (m.onset_age != null ? m.onset_age : Infinity))
    shift += eff.per_year_mm * ((m.accel_factor || 1) - 1) * (age - m.onset_age);
  return shift;
}
function normEval(id, value, ancestry, sex, age) {
  const amap = NORMS.ancestry_map[ancestry];
  if (!amap) return { status: "no_population" };
  const pop = amap.population, popLabel = NORMS.populations[pop].label;
  const cell = ((((NORMS.norms[pop] || {})[sex] || {})["18-45"]) || {})[id];
  if (!cell) return { status: "no_metric", popLabel };
  if (cell.sd == null) return { status: "no_sd", popLabel, mean: cell.mean };
  let mean = cell.mean, ageAdjusted = false;
  if (age != null && !(age >= 18 && age <= 45)) {
    const shift = ageShift(id, age, sex);
    if (shift != null) { mean = cell.mean + shift; ageAdjusted = true; }
  }
  const z = (value - mean) / cell.sd;
  return { status: "ok", popLabel, percentile: pctile(z), ageAdjusted };
}

// --- skin (sclera-normalized tone) ---------------------------------------
function srgbToLab([r, g, b]) {
  const f = (c) => { c /= 255; return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
  const [R, G, B] = [f(r), f(g), f(b)];
  const X = (0.4124 * R + 0.3576 * G + 0.1805 * B) * 100;
  const Y = (0.2126 * R + 0.7152 * G + 0.0722 * B) * 100;
  const Z = (0.0193 * R + 0.1192 * G + 0.9505 * B) * 100;
  const g3 = (t) => t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116;
  const fx = g3(X / 95.047), fy = g3(Y / 100), fz = g3(Z / 108.883);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}
const ita = (lab) => Math.atan2(lab[0] - 50, lab[2]) * 180 / Math.PI;
const melaninIndex = ([r]) => 100 * Math.log10(255 / (r + 1));   // relative; lower red -> higher melanin
function itaCategory(v) {
  return v > 55 ? "very light" : v > 41 ? "light" : v > 28 ? "intermediate"
    : v > 10 ? "tan" : v > -30 ? "brown" : "dark";
}
// ITA->Monk via published Del Bino/Monk bin boundaries (npj Digital Medicine 2025), not a linear interp.
const ITA_MONK_BINS = [[55, 1], [48, 2], [41, 3], [34, 4], [28, 5], [18, 6], [10, 7], [-10, 8], [-30, 9]];
function itaToMonk(v) { for (const [thr, m] of ITA_MONK_BINS) if (v > thr) return m; return 10; }
function boxPixels(data, W, H, x0, x1, y0, y1) {
  const px = [];
  for (let y = Math.max(0, y0); y < Math.min(H, y1); y++)
    for (let x = Math.max(0, x0); x < Math.min(W, x1); x++) {
      const i = (y * W + x) * 4; px.push([data[i], data[i + 1], data[i + 2]]);
    }
  return px;
}
function estimateIlluminant(data, W, H, P) {
  const eyes = [[33, 133, RIGHT_IRIS], [362, 263, LEFT_IRIS]];
  let cand = [];
  for (const [o, i, ir] of eyes) {
    const x0 = Math.min(P[o][0], P[i][0]), x1 = Math.max(P[o][0], P[i][0]);
    if (x1 - x0 < 6) continue;
    const half = Math.max(3, 0.3 * (x1 - x0)), cy = P[ir][1];
    const px = boxPixels(data, W, H, x0 | 0, x1 | 0, (cy - half) | 0, (cy + half) | 0);
    const neutral = px.filter((p) => { const mx = Math.max(...p), mn = Math.min(...p); return (mx - mn) / (mx + 1e-6) < 0.22; });
    if (neutral.length >= 12) cand = cand.concat(neutral);
  }
  if (cand.length < 12) return null;
  cand.sort((a, b) => (a[0] + a[1] + a[2]) - (b[0] + b[1] + b[2]));
  const bright = cand.slice(Math.floor(cand.length * 0.75));   // the actual eye-white
  return medRGB(bright);
}
function samplePatch(data, W, H, x, y, half = 9) {
  const px = boxPixels(data, W, H, x - half, x + half + 1, y - half, y + half + 1);
  if (px.length < 4) return [0, 0, 0];
  const lum = px.map((p) => (p[0] + p[1] + p[2]) / 3).sort((a, b) => a - b);
  const lo = lum[Math.floor(lum.length * 0.2)], hi = lum[Math.floor(lum.length * 0.7)];
  const keep = px.filter((p) => { const l = (p[0] + p[1] + p[2]) / 3; return l >= lo && l <= hi; });
  return medRGB(keep.length >= 3 ? keep : px);
}
function analyzeSkin(data, W, H, P, ancestry) {
  const illum = estimateIlluminant(data, W, H, P);
  let gain = [1, 1, 1], wb = "none — no clear eye-white found", normalized = false;
  if (illum) {
    const mean = (illum[0] + illum[1] + illum[2]) / 3;
    gain = illum.map((c) => Math.min(2, Math.max(0.5, mean / Math.max(c, 1))));
    wb = "sclera (eye-white) colour-cast correction"; normalized = true;
  }
  const regions = {}, mel = {};
  for (const [name, idx] of Object.entries(SKIN_REGIONS)) {
    const raw = samplePatch(data, W, H, Math.round(P[idx][0]), Math.round(P[idx][1]));
    const corr = raw.map((c, k) => Math.min(255, c * gain[k]));
    regions[name] = ita(srgbToLab(corr));
    mel[name] = melaninIndex(corr);
  }
  const allIta = Object.values(regions);
  const cheeks = [regions.right_cheek, regions.left_cheek].filter((v) => v != null);
  const overall = cheeks.length ? median(cheeks) : median(allIta);
  const cat = itaCategory(overall);
  const mean = allIta.reduce((s, v) => s + v, 0) / allIta.length;
  const evenness = Math.sqrt(allIta.reduce((s, v) => s + (v - mean) ** 2, 0) / allIta.length);
  // melasma-/PIH-sensitive pigment signals (melanin index; mirrors skin.py)
  const rc = mel.right_cheek, lc = mel.left_cheek, fh = mel.forehead;
  const cheekAsym = (rc != null && lc != null) ? Math.abs(rc - lc) : 0;
  const cheekMels = [rc, lc].filter((v) => v != null);
  const malarDelta = (cheekMels.length && fh != null)
    ? cheekMels.reduce((s, v) => s + v, 0) / cheekMels.length - fh : 0;
  const DEEP = new Set(["intermediate", "tan", "brown", "dark"]);
  const bucket = PIGMENT_FIRST.has(ancestry) ? "deep"
    : ancestry === "european" ? "fair" : (DEEP.has(cat) ? "deep" : "fair");
  const ceil = ITA_CEIL[ancestry];
  const toneReliable = !(ceil != null && overall > ceil + 6);
  return { overall, cat, monk: itaToMonk(overall), evenness, cheekAsym, malarDelta,
    bucket, normalized, wb, toneReliable, expected: PHOTOTYPE[ancestry] || "" };
}

// --- interpretation (plain language) -------------------------------------
function bandLabel(value, bands) {
  for (const b of bands) if (value <= b.max) return b.label;
  return bands[bands.length - 1].label;
}
function interpretItem(id, value, nr) {
  const info = GEN.metrics[id]; if (!info) return null;
  const valueStr = value.toFixed(1) + (info.unit || "");
  let standing = "", notable = false;
  if (info.unreliable) standing = "not reliable from a photo";
  else if (nr.status === "ok") {
    const p = Math.round(nr.percentile), typical = p >= 15 && p <= 85;
    const ageAdj = nr.ageAdjusted ? " (age-adjusted)" : "";
    standing = `${ordinal(p)} percentile for ${nr.popLabel}${ageAdj}` + (typical ? "" : " — toward the edge, still normal");
    notable = !typical;
  } else if (info.bands) standing = bandLabel(value, info.bands) + " (general range)";
  else if (nr.status === "no_sd") standing = `near the ${nr.popLabel} average`;
  else standing = "your baseline — for tracking change";
  return { plainName: info.plain_name, valueStr, standing, meaning: info.meaning || "", notable };
}
function summarize(items, cohort) {
  const notable = items.filter((it) => it.notable);
  const a = notable.length === 0
    ? `Your facial proportions all sit within the typical range for ${cohort}. Nothing here is unusual or needs attention.`
    : `Almost everything is typical for ${cohort}. A few sit toward the edges of the normal range (still normal variation): ${notable.map((it) => it.plainName.toLowerCase()).join(", ")}.`;
  return [a, "Facial structure is essentially fixed in adults, so read this as your personal baseline for tracking change over time — not a list of things to “fix”."];
}
function skinGuidance(ancestry, skin) {
  const key = skin.bucket === "deep" ? "pigment_first" : "photoaging_first";
  const block = GEN.skin_by_ancestry[key];
  const moves = block.interventions.map((k) => IVS.interventions[k]).filter(Boolean)
    .map((iv) => `${iv.label} (grade ${iv.grade}) — helps with ${(iv.affects[0] || "skin health")}.`);
  const toneLine = skin.toneReliable
    ? `Detected skin tone: ${skin.cat} (Monk ${skin.monk}; lighting cast-corrected from your eye-whites).`
    : `Skin tone: uncertain from this photo (it's bright) — by ancestry, roughly Fitzpatrick ${skin.expected}.`;
  // malar read (P2) + ancestry-resolved notes (P3) + PIH prevention (P5)
  const extra = [];
  if (key === "pigment_first" && (skin.malarDelta || skin.cheekAsym)) {
    const md = (skin.malarDelta >= 0 ? "+" : "") + skin.malarDelta.toFixed(1);
    extra.push(`Pigment pattern: malar delta (cheek vs forehead) ${md}, left/right cheek asymmetry `
      + `${skin.cheekAsym.toFixed(1)}. A higher symmetric malar delta points to melasma; higher `
      + "asymmetry points to sun spots or one-sided marks. Track these, not just overall evenness.");
  }
  extra.push(...((GEN.skin_by_ancestry.ancestry_notes || {})[ancestry] || []));
  if (block.prevent) extra.push(block.prevent);
  return { summary: block.summary, watch: block.watch, moves, toneLine, evenness: skin.evenness, extra };
}

// --- pipeline ------------------------------------------------------------
async function ensure() {
  if (landmarker) return;
  status("Loading on-device model (once)…");
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm");
  landmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/" +
        "face_landmarker/float16/1/face_landmarker.task",
    },
    numFaces: 1, runningMode: "IMAGE",
  });
  [NORMS, GEN, IVS] = await Promise.all([
    fetch("./norms/frontal_anthropometry.json").then((r) => r.json()),
    fetch("./norms/general_reference.json").then((r) => r.json()),
    fetch("./norms/interventions.json").then((r) => r.json()),
  ]);
}

async function run(img) {
  await ensure();
  status("Analyzing on your device…");
  const W = img.naturalWidth, H = img.naturalHeight;
  const res = landmarker.detect(img);
  if (!res.faceLandmarks || !res.faceLandmarks.length) {
    status("<span class='warn'>No face detected — try a clearer, front-facing photo.</span>"); return;
  }
  const lm = res.faceLandmarks[0];
  if (lm.length < 478) { status("<span class='warn'>Iris landmarks missing.</span>"); return; }
  const P = lm.map((p) => [p.x * W, p.y * H]);

  const cv = document.createElement("canvas"); cv.width = W; cv.height = H;
  const cx = cv.getContext("2d", { willReadFrequently: true }); cx.drawImage(img, 0, 0);
  const data = cx.getImageData(0, 0, W, H).data;

  const ancestry = $("ancestry").value, sex = $("sex").value;
  const ageRaw = parseInt($("age").value, 10);
  const age = Number.isFinite(ageRaw) ? ageRaw : null;
  const cal = calibrate(P);
  const M = metrics(P, cal.mm);
  const skin = analyzeSkin(data, W, H, P, ancestry);

  const items = ORDER.map((id) => interpretItem(id, M[id], normEval(id, M[id], ancestry, sex, age))).filter(Boolean);
  const cohort = (Object.values(M).length && normEval("intercanthal_width", M.intercanthal_width, ancestry, sex, age).popLabel) || `${ancestry} ${sex}`;
  render(summarize(items, cohort), skinGuidance(ancestry, skin), items, cal, img, P);
  status("Done — nothing was uploaded.");
}

function render(summary, sg, items, cal, img, P) {
  $("results").classList.remove("hidden");
  $("gist").innerHTML = summary.map((s) => `<p>${s}</p>`).join("");

  $("skin").innerHTML = `
    <p>${sg.summary}</p>
    <p><span class="tone">${sg.toneLine}</span></p>
    <p class="note">Tone evenness is ${sg.evenness.toFixed(1)} (lower = more even) — a robust number to track over time.</p>
    <p style="margin:6px 0 2px"><b>Watch first:</b> ${sg.watch.join("; ")}.</p>
    <p style="margin:6px 0 2px"><b>Highest-yield, evidence-graded moves:</b></p>
    <ul class="clean">${sg.moves.map((m) => `<li>${m}</li>`).join("")}</ul>
    ${(sg.extra || []).map((e) => `<p class="note">${e}</p>`).join("")}`;

  $("face").innerHTML = items.map((it) => {
    const edge = it.notable ? ' <span class="edge">(edge of normal)</span>' : "";
    const mean = it.meaning ? ` <span class="s">— ${it.meaning}</span>` : "";
    return `<div class="item"><b>${it.plainName}</b> <span class="v">${it.valueStr}</span> · ${it.standing}${edge}${mean}</div>`;
  }).join("");

  $("calib").innerHTML = `Ruler: your iris (≈11.7 mm) → ${cal.mm.toFixed(3)} mm/px, ±${ERR_PCT}% per measurement.`;

  const ov = $("overlay"); ov.width = img.naturalWidth; ov.height = img.naturalHeight;
  const g = ov.getContext("2d"); g.drawImage(img, 0, 0); g.fillStyle = "rgba(10,132,255,.9)";
  for (const i of Object.values(IDX)) { g.beginPath(); g.arc(P[i][0], P[i][1], Math.max(2, img.naturalWidth / 250), 0, 7); g.fill(); }
}

// --- input ---------------------------------------------------------------
function loadFile(file) {
  if (!file) { status("<span class='warn'>No image found in that drop. Click to choose, or drag from Finder.</span>"); return; }
  if (!file.type || !file.type.startsWith("image/")) { status("<span class='warn'>That file isn't an image.</span>"); return; }
  const img = new Image();
  img.onload = () => run(img);
  img.onerror = () => status("<span class='warn'>Could not read that image.</span>");
  img.src = URL.createObjectURL(file);
}

// Resolve a dropped image to a File. Finder drags land in dataTransfer.files;
// the macOS Photos app drags a "promised file" that only shows up in
// dataTransfer.items (getAsFile), or as a draggable image URL. Try all three.
// Note: items/getAsFile/getData must be read synchronously, before any await.
async function fileFromDrop(dt) {
  if (dt.files && dt.files.length) return dt.files[0];
  if (dt.items) {
    for (const it of dt.items) {
      if (it.kind === "file") { const f = it.getAsFile(); if (f) return f; }
    }
  }
  const uri = (dt.getData("text/uri-list") || dt.getData("text/plain") || "");
  const url = uri.split("\n").map((s) => s.trim()).find((s) => /^https?:\/\//.test(s));
  if (url) {
    try {
      const b = await (await fetch(url)).blob();
      if (b.type.startsWith("image/")) return new File([b], "dropped-image", { type: b.type });
    } catch (_) { /* cross-origin or unreachable — fall through to the hint */ }
  }
  return null;
}

const drop = $("drop"), file = $("file");
drop.addEventListener("click", () => file.click());
file.addEventListener("change", (e) => loadFile(e.target.files[0]));
["dragover", "dragenter"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("hot"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("hot"); }));
drop.addEventListener("drop", async (e) => {
  const f = await fileFromDrop(e.dataTransfer);
  if (f) { loadFile(f); return; }
  status("<span class='warn'>Couldn't read that drop — dragging straight from the Photos app often doesn't work in browsers. "
    + "Drag from Finder instead, or click to choose (the file picker can reach your Photos library).</span>");
});
