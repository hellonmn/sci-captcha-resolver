const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { preprocess } = require('../src/preprocess');
const { extractGlyphs } = require('../src/preprocess');
const { CharacterModel } = require('../src/model');
const { app } = require('../src/server');
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
