const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const fsp = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { preprocess } = require('../src/preprocess');
const { extractGlyphs } = require('../src/preprocess');
const { CharacterModel } = require('../src/model');
const { app } = require('../src/server');
const { evaluateExpression, solveBuffer, solveFolder } = require('../src/solver');
const { parseArgs } = require('../src/solve');
const { parseExpressionParts, listSampleImages, resolveSampleFile, formatLabelsCsv, readLabelsCsv } = require('../src/samples');
const sharp = require('sharp');
const { allowedHosts, extractImageUrl, isPublicAddress, pinnedLookup, fetchAuthorizedImage } = require('../src/url-image');

test('preprocess rejects empty and invalid image data', async () => {
  await assert.rejects(preprocess(Buffer.alloc(0)), /empty/);
  await assert.rejects(preprocess(Buffer.from('not an image')));
});

test('segmentation splits a rendered arithmetic expression into glyph crops', async () => {
  const svg = Buffer.from('<svg xmlns="http://www.w3.org/2000/svg" width="220" height="77"><rect width="100%" height="100%" fill="#f0f0f0"/><text x="60" y="52" font-family="Arial" font-size="40" font-weight="bold" fill="#555">7+4</text></svg>');
  const image = await sharp(svg).png().toBuffer();
  const glyphs = await extractGlyphs(image);
  assert.equal(glyphs.length, 3);
  assert.ok(glyphs[0].bounds.x < glyphs[1].bounds.x);
  assert.ok(glyphs[1].bounds.x < glyphs[2].bounds.x);
});

test('model loading gives a clear message when no trained model exists', async () => {
  const missing = path.join(__dirname, 'missing-model.onnx');
  const model = new CharacterModel(missing);
  await assert.rejects(model.load(), /Model file not found/);
});

test('health endpoint reports service status', () => {
  const route = app.router.stack.find((layer) => layer.route && layer.route.path === '/health');
  assert.ok(route);
  let body;
  route.route.stack[0].handle({}, { json: (value) => { body = value; } });
  assert.equal(body.ok, true);
});

test('a generated ONNX artifact is present when training has been run', { skip: !fs.existsSync(path.join(__dirname,'..','models','model.onnx')) }, () => {
  assert.ok(fs.statSync(path.join(__dirname,'..','models','model.onnx')).size > 0);
});

test('URL host allowlist is exact and case-insensitive', () => {
  const hosts = allowedHosts('images.example.com, CDN.example.net');
  assert.ok(hosts.has('images.example.com'));
  assert.ok(hosts.has('cdn.example.net'));
  assert.ok(!hosts.has('sub.images.example.com'));
});

test('URL endpoints accept url, imageUrl, and plain-text request bodies', () => {
  assert.equal(extractImageUrl({ url: 'https://images.example.com/a.png' }), 'https://images.example.com/a.png');
  assert.equal(extractImageUrl({ imageUrl: 'https://images.example.com/b.png' }), 'https://images.example.com/b.png');
  assert.equal(extractImageUrl(' https://images.example.com/c.png '), 'https://images.example.com/c.png');
  assert.equal(extractImageUrl({}), '');
});

test('pinned DNS lookup supports single-address and all-address callbacks', () => {
  const lookup = pinnedLookup('93.184.216.34', 4);
  lookup('images.example.com', {}, (error, address, family) => {
    assert.ifError(error);
    assert.equal(address, '93.184.216.34');
    assert.equal(family, 4);
  });
  lookup('images.example.com', { all: true }, (error, addresses) => {
    assert.ifError(error);
    assert.deepEqual(addresses, [{ address: '93.184.216.34', family: 4 }]);
  });
});

test('URL fetch rejects private addresses and hosts outside the allowlist', async () => {
  assert.equal(isPublicAddress('127.0.0.1'), false);
  assert.equal(isPublicAddress('10.0.0.2'), false);
  assert.equal(isPublicAddress('::1'), false);
  assert.equal(isPublicAddress('8.8.8.8'), true);
  await assert.rejects(fetchAuthorizedImage('https://images.example.com/a.png', { hosts: new Set() }), /not allowlisted/);
  await assert.rejects(fetchAuthorizedImage('http://127.0.0.1/image.png', { hosts: new Set(['127.0.0.1']) }), /IP-address URLs are not allowed/);
});

test('evaluateExpression supports arithmetic operators, Unicode symbols, and trailing equals', () => {
  assert.equal(evaluateExpression('7+4'), 11);
  assert.equal(evaluateExpression('12 - 5 ='), 7);
  assert.equal(evaluateExpression('6×3'), 18);
  assert.equal(evaluateExpression('8 ÷ 2 ='), 4);
  assert.equal(evaluateExpression('42'), 42);
  assert.equal(evaluateExpression('8÷0'), null);
  assert.equal(evaluateExpression('7=4'), null);
  assert.equal(evaluateExpression('++'), null);
});

test('solveBuffer respects confidence threshold and unparseable expressions', async () => {
  const stubRecognizer = {
    async predict() {
      return {
        text: '7+4',
        predictions: [
          { character: '7', confidence: 0.99 },
          { character: '+', confidence: 0.72 },
          { character: '4', confidence: 0.98 }
        ],
        preprocessingMs: 1,
        inferenceMs: 1,
        processingMs: 2
      };
    }
  };
  const solved = await solveBuffer(stubRecognizer, Buffer.from('x'));
  assert.equal(solved.status, 'solved');
  assert.equal(solved.answer, 11);

  const unsure = await solveBuffer(stubRecognizer, Buffer.from('x'), { threshold: 0.85 });
  assert.equal(unsure.status, 'unsure');
  assert.equal(unsure.answer, null);
  assert.equal(unsure.minConfidence, 0.72);
});

test('parseArgs parses folder path and solver CLI flags', () => {
  const parsed = parseArgs(['./samples', '--threshold', '0.75', '--no-recursive', '--json', '--out', 'out.json']);
  assert.equal(parsed.folder, './samples');
  assert.equal(parsed.threshold, 0.75);
  assert.equal(parsed.recursive, false);
  assert.equal(parsed.json, true);
  assert.equal(parsed.out, 'out.json');
  assert.throws(() => parseArgs(['./samples', '--threshold', '1.5']), /between 0 and 1/);
});

test('solveFolder processes a folder of images end-to-end', { skip: !fs.existsSync(path.join(__dirname, '..', 'models', 'model.onnx')) }, async () => {
  const tempDir = await fsp.mkdtemp(path.join(os.tmpdir(), 'ocr-solve-folder-'));
  try {
    const svg = Buffer.from('<svg xmlns="http://www.w3.org/2000/svg" width="220" height="77"><rect width="100%" height="100%" fill="#ffffff"/><text x="55" y="52" font-family="Arial" font-size="40" font-weight="bold" fill="#000000">7+4</text></svg>');
    await fsp.writeFile(path.join(tempDir, 'expr1.png'), await sharp(svg).png().toBuffer());
    await fsp.writeFile(path.join(tempDir, 'invalid.png'), Buffer.from('not an image'));

    const report = await solveFolder(tempDir);
    assert.equal(report.summary.total, 2);
    assert.equal(report.summary.errors, 1);
    assert.equal(report.results[0].file, 'expr1.png');
    assert.ok(report.results[0].text.length > 0);
    assert.equal(report.results[1].file, 'invalid.png');
    assert.equal(report.results[1].status, 'error');
  } finally {
    await fsp.rm(tempDir, { recursive: true, force: true });
  }
});

test('samples helpers split expressions, sort naturally, guard paths, and round-trip CSV', async () => {
  assert.deepEqual(parseExpressionParts('5 + 5 ='), { expression: '5+5', left: '5', operator: '+', right: '5' });
  assert.deepEqual(parseExpressionParts('12 * 3'), { expression: '12×3', left: '12', operator: '×', right: '3' });
  assert.throws(() => resolveSampleFile('/tmp/samples', '../secret.png'), /Invalid sample filename/);
  assert.throws(() => resolveSampleFile('/tmp/samples', 'notes.txt'), /Unsupported sample file extension/);

  const tempDir = await fsp.mkdtemp(path.join(os.tmpdir(), 'ocr-samples-'));
  try {
    await fsp.writeFile(path.join(tempDir, '10.png'), 'a');
    await fsp.writeFile(path.join(tempDir, '2.png'), 'b');
    await fsp.writeFile(path.join(tempDir, '1.png'), 'c');
    await fsp.writeFile(path.join(tempDir, 'labels.csv'), 'ignore');

    const listed = await listSampleImages(tempDir);
    assert.deepEqual(listed, ['1.png', '2.png', '10.png']);

    const csv = formatLabelsCsv([
      { file: '1.png', expression: '5+5' },
      { file: '2.png', expression: '9-3' }
    ]);
    assert.equal(csv, 'file,expression,left,operator,right\n1.png,5+5,5,+,5\n2.png,9-3,9,-,3\n');

    const csvPath = path.join(tempDir, 'labels.csv');
    await fsp.writeFile(csvPath, csv, 'utf8');
    const loaded = await readLabelsCsv(csvPath);
    assert.deepEqual(loaded, { '1.png': '5+5', '2.png': '9-3' });
  } finally {
    await fsp.rm(tempDir, { recursive: true, force: true });
  }
});
