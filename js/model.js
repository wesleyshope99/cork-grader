// Transfer-learning model: a frozen MobileNetV2 feature extractor (vendored,
// cached offline by the service worker) plus a small trainable classifier
// head that is retrained in-browser from the reference library. Training
// happens entirely client-side -- with only dozens of images per grade,
// training the small head takes seconds, so there is no need for a separate
// Node/offline training pipeline.

const tf = window.tf;

export const GRADES = ['Flor', 'Extra', 'AAA', 'A', 'B', 'C'];
const FEATURE_MODEL_URL = 'vendor/mobilenet/model.json';
const CLASSIFIER_STORAGE_KEY = 'indexeddb://cork-grader-classifier';
const EMBEDDING_SIZE = 1280;
const INPUT_SIZE = 224;

let featureModelPromise = null;

export function loadFeatureExtractor() {
  if (!featureModelPromise) {
    featureModelPromise = tf.loadGraphModel(FEATURE_MODEL_URL);
  }
  return featureModelPromise;
}

/** Compute the 1280-d MobileNetV2 embedding for a 300x300 normalized disk canvas/image. */
export async function embed(source) {
  const model = await loadFeatureExtractor();
  const embedding = tf.tidy(() => {
    const img = tf.browser.fromPixels(source);
    const resized = tf.image.resizeBilinear(img, [INPUT_SIZE, INPUT_SIZE]);
    const normalized = resized.toFloat().div(127.5).sub(1).expandDims(0);
    const out = model.predict(normalized);
    return out.reshape([EMBEDDING_SIZE]);
  });
  const data = await embedding.data();
  embedding.dispose();
  return data; // Float32Array, length 1280
}

function buildClassifierHead() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [EMBEDDING_SIZE], units: 64, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.dense({ units: GRADES.length, activation: 'softmax' }));
  model.compile({ optimizer: tf.train.adam(0.001), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
  return model;
}

/**
 * Retrain the classifier head from scratch using every image currently in
 * the reference library. `entries` is the array from db.getAllLibraryEntries()
 * (each has .grade and .imageBlob). Returns training stats.
 */
export async function trainClassifier(entries, { onProgress } = {}) {
  if (entries.length < GRADES.length * 2) {
    throw new Error(`Need at least 2 images per grade to train (have ${entries.length} total).`);
  }

  const xsData = [];
  const ysData = [];
  for (let i = 0; i < entries.length; i++) {
    const entry = entries[i];
    const labelIdx = GRADES.indexOf(entry.grade);
    if (labelIdx === -1) continue;
    const img = await blobToImage(entry.imageBlob);
    const vec = await embed(img);
    xsData.push(vec);
    ysData.push(labelIdx);
    if (onProgress) onProgress({ phase: 'embedding', current: i + 1, total: entries.length });
  }

  const xs = tf.tensor2d(xsData, [xsData.length, EMBEDDING_SIZE]);
  const ys = tf.oneHot(tf.tensor1d(ysData, 'int32'), GRADES.length);

  const model = buildClassifierHead();
  const epochs = 40;
  const history = await model.fit(xs, ys, {
    epochs,
    batchSize: 16,
    shuffle: true,
    validationSplit: xsData.length > GRADES.length * 4 ? 0.15 : 0,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if (onProgress) onProgress({ phase: 'training', current: epoch + 1, total: epochs, logs });
      },
    },
  });

  await model.save(CLASSIFIER_STORAGE_KEY);
  xs.dispose();
  ys.dispose();

  const finalLogs = history.history;
  const lastAcc = finalLogs.acc ? finalLogs.acc[finalLogs.acc.length - 1] : null;
  return { imagesUsed: xsData.length, finalTrainAccuracy: lastAcc };
}

let classifierPromise = null;

export async function loadClassifier({ forceReload = false } = {}) {
  if (forceReload) classifierPromise = null;
  if (!classifierPromise) {
    classifierPromise = tf.loadLayersModel(CLASSIFIER_STORAGE_KEY).catch(() => null);
  }
  return classifierPromise;
}

export async function hasTrainedModel() {
  const models = await tf.io.listModels();
  return Object.prototype.hasOwnProperty.call(models, CLASSIFIER_STORAGE_KEY);
}

/** Run inference on a normalized 300x300 disk canvas. Returns { grade, confidence, scores }. */
export async function predict(source) {
  const classifier = await loadClassifier();
  if (!classifier) throw new Error('No trained model yet -- build the reference library and train first.');

  const vec = await embed(source);
  const scores = tf.tidy(() => {
    const input = tf.tensor2d([Array.from(vec)], [1, EMBEDDING_SIZE]);
    const out = classifier.predict(input);
    return out.dataSync();
  });

  let bestIdx = 0;
  for (let i = 1; i < scores.length; i++) if (scores[i] > scores[bestIdx]) bestIdx = i;

  return {
    grade: GRADES[bestIdx],
    confidence: scores[bestIdx],
    scores: GRADES.reduce((acc, g, i) => ({ ...acc, [g]: scores[i] }), {}),
  };
}

function blobToImage(blob) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(blob);
    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img);
    };
    img.onerror = reject;
    img.src = url;
  });
}
