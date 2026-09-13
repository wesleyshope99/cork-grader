import * as db from './db.js';
import { drawNormalizedDisk, transformFromCalibration, CANVAS_SIZE, DISK_PX } from './crop.js';
import { predict, hasTrainedModel } from './model.js';

// --- tunables for the hands-free "disk is present and stable" heuristic ---
// These are isolated here on purpose: the spec calls out auto-detection as
// the least-proven part of the design, meant to be tuned from real testing
// with real disks and a real fixture, not decided up front.
const PRESENCE_VARIANCE_THRESHOLD = 12; // stddev of grayscale value inside the disk circle
const STABILITY_FRAMES = 5;             // consecutive matching predictions required to auto-commit
const CONFIDENCE_THRESHOLD = 0.5;
const LOOP_INTERVAL_MS = 150;

function $(id) { return document.getElementById(id); }

let sharedStream = null;
let calibVideoW = 0, calibVideoH = 0;
let calibState = null; // { cx, cy, radius } in native pixel coords, for the calibration screen only
let calibRafId = null;

let gradeVideoW = 0, gradeVideoH = 0;
let calibrationPixels = null; // reconstructed from stored fractional calibration
let loopTimer = null;
let predictionBuffer = [];
let awaitingRemoval = false;
let isGradeScreenVisible = false;

const offscreen = document.createElement('canvas');
offscreen.width = CANVAS_SIZE;
offscreen.height = CANVAS_SIZE;
const offscreenCtx = offscreen.getContext('2d');

async function getSharedStream() {
  if (sharedStream) return sharedStream;
  sharedStream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: 'environment' } },
    audio: false,
  });
  return sharedStream;
}

function coverParams(containerRect, nativeW, nativeH) {
  const scale = Math.max(containerRect.width / nativeW, containerRect.height / nativeH);
  return {
    scale,
    offX: (containerRect.width - nativeW * scale) / 2,
    offY: (containerRect.height - nativeH * scale) / 2,
  };
}
function nativeToCss(x, y, cover) { return { x: x * cover.scale + cover.offX, y: y * cover.scale + cover.offY }; }
function cssToNative(x, y, cover) { return { x: (x - cover.offX) / cover.scale, y: (y - cover.offY) / cover.scale }; }

// ============================= Calibration =============================

function drawCalibOverlay() {
  const canvas = $('calib-overlay');
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (!calibState || !calibVideoW) return;

  const cover = coverParams(rect, calibVideoW, calibVideoH);
  const center = nativeToCss(calibState.cx, calibState.cy, cover);
  const radiusCss = calibState.radius * cover.scale;

  ctx.strokeStyle = '#c49b6b';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(center.x, center.y, radiusCss, 0, Math.PI * 2);
  ctx.stroke();
}

function drawCalibPreview() {
  if (!calibState || !calibVideoW) return;
  const video = $('calib-video');
  const t = transformFromCalibration(calibState, calibVideoW, calibVideoH, { canvasSize: 300, diskPx: DISK_PX });
  drawNormalizedDisk($('calib-preview').getContext('2d'), video, t, { canvasSize: 300, diskPx: DISK_PX });
}

function calibLoop() {
  drawCalibOverlay();
  drawCalibPreview();
  calibRafId = requestAnimationFrame(calibLoop);
}

function wireCalibPointerEvents() {
  const canvas = $('calib-overlay');
  const pointers = new Map();
  let dragLast = null;
  let pinchStartDist = null;
  let pinchStartRadius = null;

  function currentPinchDist() {
    const pts = [...pointers.values()];
    return Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
  }

  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.size === 1) dragLast = { x: e.clientX, y: e.clientY };
    else if (pointers.size === 2) { pinchStartDist = currentPinchDist(); pinchStartRadius = calibState.radius; }
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!pointers.has(e.pointerId)) return;
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    const rect = canvas.getBoundingClientRect();
    const cover = coverParams(rect, calibVideoW, calibVideoH);

    if (pointers.size === 1 && dragLast) {
      const prevNative = cssToNative(dragLast.x - rect.left, dragLast.y - rect.top, cover);
      const curNative = cssToNative(e.clientX - rect.left, e.clientY - rect.top, cover);
      calibState.cx += curNative.x - prevNative.x;
      calibState.cy += curNative.y - prevNative.y;
      dragLast = { x: e.clientX, y: e.clientY };
    } else if (pointers.size === 2) {
      const dist = currentPinchDist();
      if (pinchStartDist) calibState.radius = Math.max(20, pinchStartRadius * (dist / pinchStartDist));
    }
  });
  function release(e) {
    pointers.delete(e.pointerId);
    if (pointers.size < 2) { pinchStartDist = null; pinchStartRadius = null; }
    dragLast = pointers.size === 1 ? [...pointers.values()][0] : null;
  }
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);
  canvas.addEventListener('pointerleave', release);
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    calibState.radius = Math.max(20, calibState.radius * (1 - e.deltaY * 0.001));
  }, { passive: false });
}

async function startCalibrationScreen() {
  $('grade-no-model').hidden = true;
  $('grade-calibrate-prompt').hidden = true;
  $('grading-screen').hidden = true;
  stopGradingLoop();
  $('calibration-screen').hidden = false;

  const stream = await getSharedStream();
  const video = $('calib-video');
  video.srcObject = stream;
  await video.play().catch(() => {});
  await new Promise((resolve) => {
    if (video.videoWidth) resolve();
    else video.addEventListener('loadedmetadata', resolve, { once: true });
  });
  calibVideoW = video.videoWidth;
  calibVideoH = video.videoHeight;

  const existing = await db.getSetting('calibration', null);
  if (existing) {
    calibState = {
      cx: existing.fx * calibVideoW,
      cy: existing.fy * calibVideoH,
      radius: existing.fr * calibVideoW,
    };
  } else {
    calibState = {
      cx: calibVideoW / 2,
      cy: calibVideoH / 2,
      radius: Math.min(calibVideoW, calibVideoH) * 0.3,
    };
  }

  calibLoop();
}

function stopCalibrationScreen() {
  if (calibRafId) cancelAnimationFrame(calibRafId);
  calibRafId = null;
  $('calibration-screen').hidden = true;
}

async function confirmCalibration() {
  await db.setSetting('calibration', {
    fx: calibState.cx / calibVideoW,
    fy: calibState.cy / calibVideoH,
    fr: calibState.radius / calibVideoW,
  });
  stopCalibrationScreen();
  await showGradingScreenIfReady();
}

// ============================= Grading loop =============================

function grayscaleVariance(canvas, diskPx) {
  const ctx = canvas.getContext('2d');
  const size = canvas.width;
  const r = diskPx / 2;
  const cx = size / 2, cy = size / 2;
  const { data } = ctx.getImageData(0, 0, size, size);
  let sum = 0, sumSq = 0, n = 0;
  const step = 4; // sample every 4th pixel for speed
  for (let y = 0; y < size; y += step) {
    for (let x = 0; x < size; x += step) {
      const dx = x - cx, dy = y - cy;
      if (dx * dx + dy * dy > r * r) continue;
      const i = (y * size + x) * 4;
      const gray = (data[i] + data[i + 1] + data[i + 2]) / 3;
      sum += gray; sumSq += gray * gray; n++;
    }
  }
  if (n === 0) return 0;
  const mean = sum / n;
  return Math.sqrt(Math.max(0, sumSq / n - mean * mean));
}

function setResultOverlay(grade) {
  const overlay = $('grade-result-overlay');
  const letter = $('grade-result-letter');
  overlay.className = 'result-overlay' + (grade ? ` grade-${grade}` : '');
  letter.textContent = grade || '';
  overlay.hidden = !grade;
}

async function commitGrade(gradeResult) {
  const diameterInput = $('session-diameter').value;
  const diameterMm = diameterInput ? parseFloat(diameterInput) : null;
  await db.addLogEntry({ grade: gradeResult.grade, confidence: gradeResult.confidence, diameterMm });
  setResultOverlay(gradeResult.grade);
  $('grade-status').textContent = `Graded: ${gradeResult.grade} (${Math.round(gradeResult.confidence * 100)}% confidence)`;
  $('last-graded').textContent = `${gradeResult.grade} at ${new Date().toLocaleTimeString()}`;
  if (navigator.vibrate) navigator.vibrate(80);
  awaitingRemoval = true;
  predictionBuffer = [];
}

async function loopTick() {
  const video = $('grade-video');
  if (!video.videoWidth || !calibrationPixels) return;

  drawNormalizedDisk(offscreenCtx, video, calibrationPixels.transform, { canvasSize: CANVAS_SIZE, diskPx: DISK_PX });
  const variance = grayscaleVariance(offscreen, DISK_PX);
  const present = variance > PRESENCE_VARIANCE_THRESHOLD;

  if (!present) {
    if (awaitingRemoval) {
      awaitingRemoval = false;
      setResultOverlay(null);
      $('grade-status').textContent = 'Show a disk to the camera…';
    }
    predictionBuffer = [];
    return;
  }

  if (awaitingRemoval) return; // disk still sitting there from the last grade; wait for it to be removed

  let result;
  try {
    result = await predict(offscreen);
  } catch (err) {
    $('grade-status').textContent = err.message;
    return;
  }

  predictionBuffer.push(result);
  if (predictionBuffer.length > STABILITY_FRAMES) predictionBuffer.shift();

  const allSameGrade = predictionBuffer.length === STABILITY_FRAMES &&
    predictionBuffer.every((r) => r.grade === predictionBuffer[0].grade);
  const avgConfidence = predictionBuffer.reduce((s, r) => s + r.confidence, 0) / (predictionBuffer.length || 1);

  if (allSameGrade && avgConfidence >= CONFIDENCE_THRESHOLD) {
    await commitGrade(predictionBuffer[predictionBuffer.length - 1]);
  } else {
    $('grade-status').textContent = 'Reading disk…';
  }
}

function startGradingLoop() {
  stopGradingLoop();
  loopTimer = setInterval(() => { if (isGradeScreenVisible) loopTick(); }, LOOP_INTERVAL_MS);
}
function stopGradingLoop() {
  if (loopTimer) clearInterval(loopTimer);
  loopTimer = null;
}

async function showGradingScreenIfReady() {
  const calibration = await db.getSetting('calibration', null);
  const trained = await hasTrainedModel();

  $('grade-no-model').hidden = trained;
  $('grade-calibrate-prompt').hidden = !trained || !!calibration;
  $('grading-screen').hidden = !trained || !calibration;
  $('calibration-screen').hidden = true;

  if (!trained || !calibration) return;

  const stream = await getSharedStream();
  const video = $('grade-video');
  video.srcObject = stream;
  await video.play().catch(() => {});
  await new Promise((resolve) => {
    if (video.videoWidth) resolve();
    else video.addEventListener('loadedmetadata', resolve, { once: true });
  });
  gradeVideoW = video.videoWidth;
  gradeVideoH = video.videoHeight;

  const calibPx = { cx: calibration.fx * gradeVideoW, cy: calibration.fy * gradeVideoH, radius: calibration.fr * gradeVideoW };
  calibrationPixels = { transform: transformFromCalibration(calibPx, gradeVideoW, gradeVideoH, { canvasSize: CANVAS_SIZE, diskPx: DISK_PX }) };

  predictionBuffer = [];
  awaitingRemoval = false;
  setResultOverlay(null);
  $('grade-status').textContent = 'Show a disk to the camera…';
  startGradingLoop();
}

async function handleManualGrade() {
  const video = $('grade-video');
  if (!video.videoWidth || !calibrationPixels) return;
  drawNormalizedDisk(offscreenCtx, video, calibrationPixels.transform, { canvasSize: CANVAS_SIZE, diskPx: DISK_PX });
  try {
    const result = await predict(offscreen);
    await commitGrade(result);
  } catch (err) {
    $('grade-status').textContent = err.message;
  }
}

export async function initGradeTab() {
  $('btn-start-calibration').addEventListener('click', startCalibrationScreen);
  $('btn-confirm-calibration').addEventListener('click', confirmCalibration);
  $('btn-cancel-calibration').addEventListener('click', async () => { stopCalibrationScreen(); await showGradingScreenIfReady(); });
  $('btn-recalibrate').addEventListener('click', startCalibrationScreen);
  $('btn-manual-grade').addEventListener('click', handleManualGrade);
  wireCalibPointerEvents();

  window.addEventListener('model-trained', showGradingScreenIfReady);
  window.addEventListener('tab-shown', (e) => {
    isGradeScreenVisible = e.detail.tab === 'grade';
    if (isGradeScreenVisible) showGradingScreenIfReady();
  });

  isGradeScreenVisible = true; // grade is the default active tab on load
  await showGradingScreenIfReady();
}
