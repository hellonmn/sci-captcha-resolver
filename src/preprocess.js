const sharp = require('sharp');

const SIDE = 32;
const MAX_SIDE = 1200;

async function preprocess(buffer) {
  if (!Buffer.isBuffer(buffer) || buffer.length === 0) throw new TypeError('Image buffer is empty');
  const { data, info } = await sharp(buffer, { limitInputPixels: 16_000_000 })
    .rotate().resize(SIDE, SIDE, { fit: 'contain', background: { r: 255, g: 255, b: 255 } })
    .greyscale().raw().toBuffer({ resolveWithObject: true });
  if (info.width !== SIDE || info.height !== SIDE || info.channels !== 1) throw new Error('Unexpected preprocessing shape');
  return normalize(data);
}

function normalize(data) {
  const tensor = new Float32Array(data.length);
  for (let i = 0; i < tensor.length; i++) tensor[i] = data[i] / 127.5 - 1;
  return tensor;
}

function toMnistTensor(tensor32) {
  if (!tensor32 || tensor32.length !== SIDE * SIDE) throw new TypeError('Expected a normalized 32x32 glyph');
  const tensor28 = new Float32Array(28 * 28);
  for (let y = 0; y < 28; y++) {
    const sourceY = Math.min(SIDE - 1, Math.floor((y + 0.5) * SIDE / 28));
    for (let x = 0; x < 28; x++) {
      const sourceX = Math.min(SIDE - 1, Math.floor((x + 0.5) * SIDE / 28));
      // The ONNX Model Zoo digit model expects white ink on a black canvas in [0, 1].
      tensor28[y * 28 + x] = (1 - tensor32[sourceY * SIDE + sourceX]) / 2;
    }
  }
  return tensor28;
}

function normalizeCrop(data, imageWidth, imageHeight, box) {
  const margin = Math.round(Math.max(box.right - box.left, box.bottom - box.top) * 0.16);
  const left = Math.max(0, box.left - margin), top = Math.max(0, box.top - margin);
  const right = Math.min(imageWidth, box.right + margin), bottom = Math.min(imageHeight, box.bottom + margin);
  const cropWidth = right - left, cropHeight = bottom - top;
  const scale = Math.min(SIDE / cropWidth, SIDE / cropHeight);
  const scaledWidth = Math.max(1, Math.round(cropWidth * scale));
  const scaledHeight = Math.max(1, Math.round(cropHeight * scale));
  const offsetX = Math.floor((SIDE - scaledWidth) / 2);
  const offsetY = Math.floor((SIDE - scaledHeight) / 2);
  const tensor = new Float32Array(SIDE * SIDE);
  tensor.fill(1); // white letterbox, matching the Sharp inference preprocessing

  for (let y = 0; y < scaledHeight; y++) {
    const sourceY = top + Math.max(0, Math.min(cropHeight - 1, Math.floor((y + 0.5) * cropHeight / scaledHeight)));
    for (let x = 0; x < scaledWidth; x++) {
      const sourceX = left + Math.max(0, Math.min(cropWidth - 1, Math.floor((x + 0.5) * cropWidth / scaledWidth)));
      const value = data[sourceY * imageWidth + sourceX];
      tensor[(offsetY + y) * SIDE + offsetX + x] = value / 127.5 - 1;
    }
  }
  return tensor;
}

function otsuThreshold(histogram, total) {
  let sum = 0;
  for (let i = 0; i < 256; i++) sum += i * histogram[i];
  let sumBackground = 0, weightBackground = 0, bestVariance = -1, best = 127;
  for (let t = 0; t < 256; t++) {
    weightBackground += histogram[t];
    if (weightBackground === 0) continue;
    const weightForeground = total - weightBackground;
    if (weightForeground === 0) break;
    sumBackground += t * histogram[t];
    const meanBackground = sumBackground / weightBackground;
    const meanForeground = (sum - sumBackground) / weightForeground;
    const variance = weightBackground * weightForeground * (meanBackground - meanForeground) ** 2;
    if (variance > bestVariance) { bestVariance = variance; best = t; }
  }
  return best;
}

function components(data, width, height) {
  const histogram = new Uint32Array(256);
  for (const value of data) histogram[value]++;
  const threshold = otsuThreshold(histogram, data.length);
  const border = [];
  for (let x = 0; x < width; x++) { border.push(data[x], data[(height - 1) * width + x]); }
  for (let y = 1; y < height - 1; y++) { border.push(data[y * width], data[y * width + width - 1]); }
  const background = border.reduce((sum, value) => sum + value, 0) / Math.max(border.length, 1);
  const lightBackground = background >= threshold;
  const isInk = (value) => lightBackground ? value < threshold : value > threshold;
  const visited = new Uint8Array(data.length);
  const queue = new Int32Array(data.length);
  const minimumArea = Math.max(3, Math.floor(data.length * 0.000005));
  const found = [];

  for (let start = 0; start < data.length; start++) {
    if (visited[start] || !isInk(data[start])) continue;
    let head = 0, tail = 0;
    queue[tail++] = start; visited[start] = 1;
    let minX = width, minY = height, maxX = -1, maxY = -1;
    while (head < tail) {
      const index = queue[head++], x = index % width, y = (index / width) | 0;
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
      for (let dy = -1; dy <= 1; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= height) continue;
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = x + dx;
          if (nx < 0 || nx >= width) continue;
          const next = ny * width + nx;
          if (!visited[next] && isInk(data[next])) { visited[next] = 1; queue[tail++] = next; }
        }
      }
    }
    if (tail >= minimumArea) found.push({ left: minX, top: minY, right: maxX + 1, bottom: maxY + 1, area: tail });
  }
  if (found.length < 2) return found;

  // Join detached strokes that belong to one glyph, such as the two bars of '='
  // and the dot/bar/dot parts of '÷'.
  const typicalHeight = found.map((box) => box.bottom - box.top).sort((a, b) => a - b)[Math.floor(found.length / 2)];
  let changed = true;
  while (changed) {
    changed = false;
    outer: for (let i = 0; i < found.length; i++) {
      for (let j = i + 1; j < found.length; j++) {
        const a = found[i], b = found[j];
        const overlap = Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left));
        const minWidth = Math.min(a.right - a.left, b.right - b.left);
        const verticalGap = Math.max(0, Math.max(a.top, b.top) - Math.min(a.bottom, b.bottom));
        if (overlap / minWidth >= 0.55 && verticalGap <= typicalHeight * 0.7) {
          found[i] = { left: Math.min(a.left, b.left), top: Math.min(a.top, b.top), right: Math.max(a.right, b.right), bottom: Math.max(a.bottom, b.bottom), area: a.area + b.area };
          found.splice(j, 1); changed = true; break outer;
        }
      }
    }
  }
  return found.sort((a, b) => a.left - b.left);
}

async function extractGlyphs(buffer, options = {}) {
  if (!Buffer.isBuffer(buffer) || buffer.length === 0) throw new TypeError('Image buffer is empty');
  const { data, info } = await sharp(buffer, { limitInputPixels: 16_000_000 })
    .rotate().resize({ width: MAX_SIDE, height: MAX_SIDE, fit: 'inside', withoutEnlargement: true })
    .greyscale().raw().toBuffer({ resolveWithObject: true });
  const boxes = components(data, info.width, info.height);
  if (boxes.length === 0) throw new Error('No characters detected in image');

  const glyphs = await Promise.all(boxes.map(async (box) => {
    const tensor = normalizeCrop(data, info.width, info.height, box);
    const glyph = {
      tensor,
      bounds: { x: box.left, y: box.top, width: box.right - box.left, height: box.bottom - box.top }
    };
    if (options.includeBuffers) {
      const pixels = Buffer.from(Uint8Array.from(tensor, (value) => Math.round((value + 1) * 127.5)));
      glyph.buffer = await sharp(pixels, { raw: { width: SIDE, height: SIDE, channels: 1 } }).png().toBuffer();
    }
    return glyph;
  }));
  return glyphs;
}

module.exports = { preprocess, extractGlyphs, components, toMnistTensor, SIDE };
