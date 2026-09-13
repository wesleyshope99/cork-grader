import * as db from './db.js';
import { CropCanvas, drawNormalizedDisk } from './crop.js';
import { trainClassifier, GRADES } from './model.js';

let cropper = null;
let pendingImage = null;

function $(id) { return document.getElementById(id); }

function showCaptureStep(step) {
  $('capture-step-file').hidden = step !== 'file';
  $('capture-step-crop').hidden = step !== 'crop';
  $('capture-step-details').hidden = step !== 'details';
}

function openCaptureModal() {
  $('capture-file-input').value = '';
  pendingImage = null;
  showCaptureStep('file');
  $('capture-screen').hidden = false;
}

function closeCaptureModal() {
  $('capture-screen').hidden = true;
}

function loadImageFromFile(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}

async function refreshLibraryGrid() {
  const entries = await db.getAllLibraryEntries();
  const grid = $('library-grid');
  grid.innerHTML = '';
  for (const entry of entries.slice().reverse()) {
    const url = URL.createObjectURL(entry.imageBlob);
    const div = document.createElement('div');
    div.className = 'library-item';
    div.innerHTML = `
      <img src="${url}" alt="${entry.grade} disk" />
      <span class="grade-tag">${entry.grade}</span>
      <button class="btn-delete" data-id="${entry.id}" aria-label="Delete">×</button>
    `;
    grid.appendChild(div);
  }
  grid.querySelectorAll('.btn-delete').forEach((btn) => {
    btn.addEventListener('click', async () => {
      await db.deleteLibraryEntry(Number(btn.dataset.id));
      await refreshLibraryGrid();
      await refreshGradeCounts();
    });
  });
}

async function refreshGradeCounts() {
  const counts = await db.countLibraryByGrade();
  const el = $('grade-counts');
  el.innerHTML = GRADES.map((g) => `<span class="grade-count-pill">${g}: ${counts[g] || 0}</span>`).join('');
}

function setTrainStatus(text) { $('train-status').textContent = text; }

async function loadStoredTrainStatus() {
  const stats = await db.getSetting('lastTrainStats', null);
  if (stats) {
    const acc = stats.finalTrainAccuracy != null ? `${Math.round(stats.finalTrainAccuracy * 100)}% train acc` : '';
    setTrainStatus(`Last trained on ${stats.imagesUsed} images ${acc ? '(' + acc + ')' : ''} · ${new Date(stats.trainedAt).toLocaleString()}`);
  }
}

async function handleTrain() {
  const entries = await db.getAllLibraryEntries();
  const btn = $('btn-train');
  const progressWrap = $('train-progress-wrap');
  const progressBar = $('train-progress-bar');
  btn.disabled = true;
  progressWrap.hidden = false;
  progressBar.style.width = '0%';

  try {
    const stats = await trainClassifier(entries, {
      onProgress: ({ phase, current, total }) => {
        if (phase === 'embedding') {
          const pct = (current / total) * 40; // embedding = first 40% of the bar
          progressBar.style.width = `${pct}%`;
          setTrainStatus(`Computing embeddings ${current}/${total}…`);
        } else {
          const pct = 40 + (current / total) * 60;
          progressBar.style.width = `${pct}%`;
          setTrainStatus(`Training ${current}/${total} epochs…`);
        }
      },
    });
    stats.trainedAt = Date.now();
    await db.setSetting('lastTrainStats', stats);
    await loadStoredTrainStatus();
    window.dispatchEvent(new CustomEvent('model-trained'));
  } catch (err) {
    setTrainStatus(err.message);
  } finally {
    btn.disabled = false;
    progressWrap.hidden = true;
  }
}

export async function initLibraryTab() {
  $('btn-add-reference').addEventListener('click', openCaptureModal);
  $('btn-crop-cancel').addEventListener('click', closeCaptureModal);
  $('btn-details-back').addEventListener('click', () => showCaptureStep('crop'));

  $('capture-file-input').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    pendingImage = await loadImageFromFile(file);
    showCaptureStep('crop');
    if (!cropper) {
      cropper = new CropCanvas($('crop-canvas'));
    }
    cropper.setSource(pendingImage);
  });

  $('btn-crop-next').addEventListener('click', () => {
    const thumbCtx = $('details-thumb').getContext('2d');
    drawNormalizedDisk(thumbCtx, cropper.canvas, { offsetX: 0, offsetY: 0, scale: 150 / 300 }, { canvasSize: 150, diskPx: 130 });
    showCaptureStep('details');
  });

  $('btn-save-reference').addEventListener('click', async () => {
    const diameter = parseFloat($('input-diameter').value);
    const grade = $('input-grade').value;
    const note = $('input-note').value.trim();
    if (!diameter || diameter <= 0) {
      alert('Enter a valid diameter in mm.');
      return;
    }
    // Captured here, at save time, from the still-live crop canvas -- not
    // pre-computed at the "Next" step -- so there is no window where a fast
    // tap-through could save an entry before its blob has finished encoding.
    const blob = await cropper.toBlob();
    await db.addLibraryEntry({
      grade,
      diameterMm: diameter,
      note,
      timestamp: Date.now(),
      imageBlob: blob,
    });
    $('input-diameter').value = '';
    $('input-note').value = '';
    closeCaptureModal();
    await refreshLibraryGrid();
    await refreshGradeCounts();
  });

  $('btn-train').addEventListener('click', handleTrain);

  await refreshLibraryGrid();
  await refreshGradeCounts();
  await loadStoredTrainStatus();
}
