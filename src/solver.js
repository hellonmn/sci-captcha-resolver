const fs = require('node:fs/promises');
const path = require('node:path');
const { performance } = require('node:perf_hooks');
const { Recognizer } = require('./predict');

const IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tif', '.tiff']);

/**
 * Parse and evaluate a simple arithmetic expression recognized by the OCR pipeline.
 * Supports integer literals and binary +, -, × (*), ÷ (/) expressions with optional trailing '='.
 * @param {string} text
 * @returns {number | null}
 */
function evaluateExpression(text) {
  if (!text || typeof text !== 'string') return null;
  const cleaned = text
    .replace(/\s+/g, '')
    .replace(/=+$/, '')
    .replace(/×/g, '*')
    .replace(/÷/g, '/');

  const match = cleaned.match(/^(-?\d+)(?:([+\-*/])(\d+))?$/);
  if (!match) return null;

  const left = Number(match[1]);
  const operator = match[2];
  if (!operator) {
    return Number.isFinite(left) ? Math.round(left) : null;
  }

  const right = Number(match[3]);
  let result;
  switch (operator) {
    case '+':
      result = left + right;
      break;
    case '-':
      result = left - right;
      break;
    case '*':
      result = left * right;
      break;
    case '/':
      if (right === 0) return null;
      result = left / right;
      break;
    default:
      return null;
  }

  return Number.isFinite(result) ? Math.round(result) : null;
}

/**
 * Recognize and solve an arithmetic expression from a single image buffer.
 * @param {Recognizer} recognizer
 * @param {Buffer} buffer
 * @param {{ threshold?: number }} [options]
 */
async function solveBuffer(recognizer, buffer, options = {}) {
  const threshold = Number.isFinite(options.threshold) ? options.threshold : 0;
  const prediction = await recognizer.predict(buffer);
  const confidences = prediction.predictions.map((item) => item.confidence);
  const minConfidence = confidences.length > 0 ? Math.min(...confidences) : 0;
  const rawAnswer = evaluateExpression(prediction.text);
  const belowThreshold = threshold > 0 && minConfidence < threshold;
  const status = rawAnswer === null || belowThreshold ? 'unsure' : 'solved';

  return {
    text: prediction.text,
    answer: status === 'solved' ? rawAnswer : null,
    status,
    minConfidence,
    predictions: prediction.predictions,
    preprocessingMs: prediction.preprocessingMs,
    inferenceMs: prediction.inferenceMs,
    processingMs: prediction.processingMs
  };
}

/**
 * Recursively collect supported image file paths under a directory in deterministic order.
 * @param {string} directory
 * @param {boolean} [recursive=true]
 * @returns {Promise<string[]>}
 */
async function collectImageFiles(directory, recursive = true) {
  const entries = await fs.readdir(directory, { withFileTypes: true });
  entries.sort((a, b) => a.name.localeCompare(b.name));

  const files = [];
  for (const entry of entries) {
    const fullPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      if (recursive) {
        files.push(...await collectImageFiles(fullPath, recursive));
      }
      continue;
    }
    if (entry.isFile() && IMAGE_EXTENSIONS.has(path.extname(entry.name).toLowerCase())) {
      files.push(fullPath);
    }
  }
  return files;
}

function percentile(values, ratio) {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.max(0, Math.min(sorted.length - 1, Math.ceil(ratio * sorted.length) - 1));
  return sorted[index];
}

/**
 * Run the arithmetic solver over all images in a folder.
 * @param {string} folderPath
 * @param {{
 *   threshold?: number,
 *   recursive?: boolean,
 *   modelPath?: string,
 *   digitModelPath?: string,
 *   recognizer?: Recognizer
 * }} [options]
 */
async function solveFolder(folderPath, options = {}) {
  if (!folderPath || typeof folderPath !== 'string') {
    throw new TypeError('Provide a path to a folder of images.');
  }
  const resolvedFolder = path.resolve(folderPath);
  const stats = await fs.stat(resolvedFolder).catch(() => null);
  if (!stats || !stats.isDirectory()) {
    throw new Error(`Image folder not found: ${resolvedFolder}`);
  }

  const recursive = options.recursive !== false;
  const files = await collectImageFiles(resolvedFolder, recursive);
  if (files.length === 0) {
    throw new Error(`No supported image files found in: ${resolvedFolder}`);
  }

  const recognizer = options.recognizer || new Recognizer({
    modelPath: options.modelPath,
    digitModelPath: options.digitModelPath
  });

  const loadStart = performance.now();
  await recognizer.load();
  const loadMs = performance.now() - loadStart;

  const threshold = Number.isFinite(options.threshold) ? options.threshold : 0;
  const results = [];
  const timings = [];

  for (const filePath of files) {
    const relativePath = path.relative(resolvedFolder, filePath).replace(/\\/g, '/');
    try {
      const buffer = await fs.readFile(filePath);
      const solved = await solveBuffer(recognizer, buffer, { threshold });
      timings.push(solved.processingMs);
      results.push({
        file: relativePath,
        status: solved.status,
        text: solved.text,
        answer: solved.answer,
        minConfidence: Number(solved.minConfidence.toFixed(6)),
        processingMs: Number(solved.processingMs.toFixed(3))
      });
    } catch (error) {
      results.push({
        file: relativePath,
        status: 'error',
        text: '',
        answer: null,
        minConfidence: 0,
        processingMs: 0,
        error: error.message
      });
    }
  }

  const solvedCount = results.filter((item) => item.status === 'solved').length;
  const unsureCount = results.filter((item) => item.status === 'unsure').length;
  const errorCount = results.filter((item) => item.status === 'error').length;
  const totalMs = timings.reduce((sum, value) => sum + value, 0);

  return {
    summary: {
      folder: resolvedFolder,
      total: results.length,
      solved: solvedCount,
      unsure: unsureCount,
      errors: errorCount,
      coverage: Number((solvedCount / results.length).toFixed(4)),
      threshold,
      loadMs: Number(loadMs.toFixed(3)),
      averageMs: timings.length ? Number((totalMs / timings.length).toFixed(3)) : 0,
      p95Ms: Number(percentile(timings, 0.95).toFixed(3))
    },
    results
  };
}

module.exports = {
  IMAGE_EXTENSIONS,
  evaluateExpression,
  solveBuffer,
  collectImageFiles,
  solveFolder
};
