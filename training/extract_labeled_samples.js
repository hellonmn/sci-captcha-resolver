// Convert expression-level labels into the same 32x32 glyph tensors used at inference.
const fs = require('node:fs');
const path = require('node:path');
const { extractGlyphs } = require('../src/preprocess');

const CLASSES = [...'0123456789', '+', '-', '×', '÷', '='];

function parseCsv(text) {
  const rows = [];
  let row = [], field = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (ch === '"') quoted = false;
      else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ',') { row.push(field); field = ''; }
    else if (ch === '\n' || ch === '\r') {
      if (ch === '\r' && text[i + 1] === '\n') i++;
      row.push(field); field = '';
      if (row.some((value) => value !== '')) rows.push(row);
      row = [];
    } else field += ch;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  if (quoted) throw new Error('CSV has an unterminated quoted field');
  const headers = rows.shift();
  if (!headers) throw new Error('CSV is empty');
  return rows.map((values) => Object.fromEntries(headers.map((key, i) => [key, values[i] ?? ''])));
}

async function main() {
  const [, , csvArg, outputArg, seedArg = '1729'] = process.argv;
  if (!csvArg || !outputArg) throw new Error('Usage: node extract_labeled_samples.js <labels.csv> <output-dir> [seed]');
  const csvPath = path.resolve(csvArg), output = path.resolve(outputArg);
  const rows = parseCsv(fs.readFileSync(csvPath, 'utf8').replace(/^\uFEFF/, ''));
  for (const split of ['train', 'validation']) {
    for (let i = 0; i < CLASSES.length; i++) fs.mkdirSync(path.join(output, split, String(i).padStart(2, '0')), { recursive: true });
  }

  // Split by whole expression, stratified by operator, so glyphs from one image
  // never appear in both training and validation.
  const rng = (() => { let state = Number(seedArg) >>> 0; return () => ((state = (1664525 * state + 1013904223) >>> 0) / 2 ** 32); })();
  const groups = new Map();
  for (const row of rows) {
    if (!row.file || !row.expression) throw new Error('Each CSV row needs file and expression fields');
    const operators = [...row.expression].filter((char) => char === '+' || char === '-');
    if (operators.length !== 1 || [...row.expression].some((char) => !CLASSES.includes(char))) {
      throw new Error(`Unsupported expression in ${row.file}: ${row.expression}`);
    }
    const key = operators[0];
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  const validationRows = new Set();
  for (const group of groups.values()) {
    for (let i = group.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [group[i], group[j]] = [group[j], group[i]]; }
    const count = Math.max(1, Math.round(group.length * 0.2));
    for (const row of group.slice(0, count)) validationRows.add(row);
  }

  const classCounts = { train: Object.fromEntries(CLASSES.map((c) => [c, 0])), validation: Object.fromEntries(CLASSES.map((c) => [c, 0])) };
  let glyphTotal = 0;
  for (const [rowIndex, row] of rows.entries()) {
    const imagePath = path.resolve(path.dirname(csvPath), row.file);
    if (!imagePath.startsWith(`${path.dirname(csvPath)}${path.sep}`) || !fs.existsSync(imagePath)) {
      throw new Error(`Image is missing or outside the sample directory: ${row.file}`);
    }
    const glyphs = await extractGlyphs(fs.readFileSync(imagePath), { includeBuffers: true });
    const labels = [...row.expression];
    if (glyphs.length !== labels.length) throw new Error(`${row.file}: expected ${labels.length} glyphs from ${row.expression}, segmented ${glyphs.length}`);
    const split = validationRows.has(row) ? 'validation' : 'train';
    for (const [index, char] of labels.entries()) {
      const classIndex = CLASSES.indexOf(char);
      const target = path.join(output, split, String(classIndex).padStart(2, '0'), `${String(rowIndex).padStart(5, '0')}-${index}.png`);
      fs.writeFileSync(target, glyphs[index].buffer);
      classCounts[split][char]++;
      glyphTotal++;
    }
  }
  process.stdout.write(`${JSON.stringify({ expressions: rows.length, glyphs: glyphTotal, classCounts })}\n`);
}

main().catch((error) => { console.error(error.message); process.exitCode = 1; });
