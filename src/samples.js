const fs = require('node:fs/promises');
const path = require('node:path');
const { IMAGE_EXTENSIONS } = require('./solver');

/**
 * Normalize an expression string and split binary expressions into left, operator, right parts.
 * @param {string} rawText
 */
function parseExpressionParts(rawText) {
  const cleaned = String(rawText || '')
    .trim()
    .replace(/\s+/g, '')
    .replace(/=+$/, '')
    .replace(/\*/g, '×')
    .replace(/\//g, '÷');

  if (!cleaned) {
    return { expression: '', left: '', operator: '', right: '' };
  }

  const binary = cleaned.match(/^(-?\d+)([+\-×÷])(\d+)$/);
  if (binary) {
    return {
      expression: cleaned,
      left: binary[1],
      operator: binary[2],
      right: binary[3]
    };
  }

  const single = cleaned.match(/^(-?\d+)$/);
  if (single) {
    return {
      expression: cleaned,
      left: single[1],
      operator: '',
      right: ''
    };
  }

  return { expression: cleaned, left: '', operator: '', right: '' };
}

/**
 * List supported image files inside a directory in natural numeric order.
 * @param {string} samplesDir
 * @returns {Promise<string[]>}
 */
async function listSampleImages(samplesDir) {
  const entries = await fs.readdir(samplesDir, { withFileTypes: true }).catch((error) => {
    if (error.code === 'ENOENT') return [];
    throw error;
  });

  return entries
    .filter((entry) => entry.isFile() && IMAGE_EXTENSIONS.has(path.extname(entry.name).toLowerCase()))
    .map((entry) => entry.name)
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' }));
}

/**
 * Safely resolve an image filename inside samplesDir, preventing path traversal.
 * @param {string} samplesDir
 * @param {string} filename
 */
function resolveSampleFile(samplesDir, filename) {
  if (!filename || typeof filename !== 'string' || path.basename(filename) !== filename) {
    throw new Error('Invalid sample filename.');
  }
  if (!IMAGE_EXTENSIONS.has(path.extname(filename).toLowerCase())) {
    throw new Error('Unsupported sample file extension.');
  }
  const base = path.resolve(samplesDir);
  const resolved = path.resolve(base, filename);
  if (!resolved.startsWith(`${base}${path.sep}`)) {
    throw new Error('Sample path escapes directory.');
  }
  return resolved;
}

function escapeCsvCell(value) {
  const text = String(value ?? '');
  if (/[",\r\n]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

/**
 * Format labeled rows as RFC-4180 CSV with both combined expression and split parts.
 * @param {Array<{ file: string, expression: string }>} rows
 */
function formatLabelsCsv(rows) {
  const lines = ['file,expression,left,operator,right'];
  for (const row of rows) {
    if (!row || !row.file) continue;
    const parts = parseExpressionParts(row.expression);
    if (!parts.expression) continue;
    lines.push([
      escapeCsvCell(row.file),
      escapeCsvCell(parts.expression),
      escapeCsvCell(parts.left),
      escapeCsvCell(parts.operator),
      escapeCsvCell(parts.right)
    ].join(','));
  }
  return `${lines.join('\n')}\n`;
}

function parseCsvLine(line) {
  const cells = [];
  let current = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === '"' && line[i + 1] === '"') {
        current += '"';
        i++;
      } else if (ch === '"') {
        inQuotes = false;
      } else {
        current += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ',') {
      cells.push(current);
      current = '';
    } else {
      current += ch;
    }
  }
  cells.push(current);
  return cells;
}

/**
 * Load existing labels.csv mapping (filename -> expression) if present.
 * @param {string} csvPath
 * @returns {Promise<Record<string, string>>}
 */
async function readLabelsCsv(csvPath) {
  const raw = await fs.readFile(csvPath, 'utf8').catch((error) => {
    if (error.code === 'ENOENT') return '';
    throw error;
  });
  if (!raw.trim()) return {};

  const lines = raw.split(/\r?\n/).filter((line) => line.trim().length > 0);
  const labels = {};
  for (let i = 0; i < lines.length; i++) {
    const cells = parseCsvLine(lines[i]);
    if (i === 0 && cells[0] === 'file') continue;
    if (cells[0] && cells[1]) {
      labels[cells[0]] = cells[1];
    }
  }
  return labels;
}

module.exports = {
  parseExpressionParts,
  listSampleImages,
  resolveSampleFile,
  formatLabelsCsv,
  readLabelsCsv
};
