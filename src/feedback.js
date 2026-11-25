const fs = require('node:fs/promises');
const path = require('node:path');
const { randomUUID } = require('node:crypto');

const CLASSES = [...'0123456789+-×÷='];

function normalizeExpectedText(value) {
  if (typeof value !== 'string') throw new TypeError('expectedText must be a string containing the correct expression.');
  const text = value.normalize('NFC').replace(/\s+/gu, '');
  if (!text || [...text].some((character) => !CLASSES.includes(character))) {
    throw new TypeError('expectedText may contain only digits and +, -, ×, ÷, = characters.');
  }
  return text;
}

async function saveVerifiedGlyphs(glyphBuffers, expectedText, datasetDirectory) {
  const text = normalizeExpectedText(expectedText);
  const labels = [...text];
  if (!Array.isArray(glyphBuffers) || glyphBuffers.length !== labels.length) {
    const detectedCount = Array.isArray(glyphBuffers) ? glyphBuffers.length : 0;
    throw new RangeError(`expectedText has ${labels.length} characters, but the image was segmented into ${detectedCount} glyphs.`);
  }
  const paths = [];
  for (let index = 0; index < labels.length; index++) {
    const classIndex = CLASSES.indexOf(labels[index]);
    const directory = path.join(datasetDirectory, 'train', String(classIndex).padStart(2, '0'));
    await fs.mkdir(directory, { recursive: true });
    const filePath = path.join(directory, `feedback-${randomUUID()}.png`);
    await fs.writeFile(filePath, glyphBuffers[index], { flag: 'wx' });
    paths.push(filePath);
  }
  return paths;
}

module.exports = { CLASSES, normalizeExpectedText, saveVerifiedGlyphs };
