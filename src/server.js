require('dotenv').config();
const express = require('express');
const multer = require('multer');
const cheerio = require('cheerio');
const fs = require('node:fs/promises');
const path = require('node:path');
const { Recognizer } = require('./predict');
const { evaluateExpression } = require('./solver');
const { listSampleImages, resolveSampleFile, formatLabelsCsv, readLabelsCsv } = require('./samples');
const { fetchAuthorizedImage, extractImageUrl, UrlImageError } = require('./url-image');
const { saveVerifiedGlyphs } = require('./feedback');

const app = express();
const defaultJson = express.json({ limit: '10kb' });
const labelsJson = express.json({ limit: '1mb' });
app.use((req, res, next) => (req.path === '/api/samples/labels' ? labelsJson : defaultJson)(req, res, next));
app.use(express.text({ type: 'text/plain', limit: '2kb' }));

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 5 * 1024 * 1024, files: 1 }
});

const recognizer = new Recognizer();
const feedbackDirectory = path.resolve(process.env.FEEDBACK_DATA_DIR || 'data/generated');
const samplesDirectory = path.resolve(process.env.SAMPLES_DIR || 'data/samples');
app.disable('x-powered-by');

const USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/157.0.0.0 Safari/537.36';
const FORM_PAGE = 'https://www.sci.gov.in/case-status-aor-code/';

/* ==================== Helpers ==================== */

function extractScid(imageUrl) {
  if (!imageUrl) return null;

  const match = String(imageUrl).match(/(?:[?&](?:id|scid)=)([a-z0-9]{15,})/i);
  if (match) return match[1];

  try {
    const u = new URL(imageUrl);
    return u.searchParams.get('id') || u.searchParams.get('scid') || null;
  } catch {
    return null;
  }
}

function bufferToDataUrl(buffer) {
  let mime = 'image/png';
  if (buffer[0] === 0xff && buffer[1] === 0xd8) mime = 'image/jpeg';
  else if (buffer.slice(0, 3).toString('ascii') === 'GIF') mime = 'image/gif';
  else if (buffer.slice(8, 12).toString('ascii') === 'WEBP') mime = 'image/webp';
  return `data:${mime};base64,${buffer.toString('base64')}`;
}

/* ---------- Form parameters ---------- */

const FORM_DEFAULTS = {
  party_type: 'any',
  aor_code: '2357',
  year: '2026',
  case_status: 'P'
};
const FORM_VALUE_PATTERN = /^[A-Za-z0-9_.\- ]{1,40}$/;

function pickFormParams(body) {
  const params = {};
  for (const [key, fallback] of Object.entries(FORM_DEFAULTS)) {
    const raw = body && typeof body === 'object' ? body[key] : undefined;
    if (raw === undefined || raw === null || String(raw).trim() === '') {
      params[key] = fallback;
      continue;
    }
    const value = String(raw).trim();
    if (!FORM_VALUE_PATTERN.test(value)) {
      const err = new Error(`Invalid value for ${key}`);
      err.status = 400;
      throw err;
    }
    params[key] = value;
  }
  return params;
}

/* ---------- Live form options ---------- */

const FORM_CACHE_MS = 30 * 60 * 1000;
let formCache = { at: 0, data: null };

async function loadFormFields({ force = false } = {}) {
  if (!force && formCache.data && Date.now() - formCache.at < FORM_CACHE_MS) return formCache.data;

  const res = await fetch(FORM_PAGE, {
    headers: { 'user-agent': USER_AGENT, accept: 'text/html,application/xhtml+xml' }
  });
  if (!res.ok) throw new Error(`sci.gov.in returned HTTP ${res.status}`);

  const $ = cheerio.load(await res.text());
  const form = $('select[name="aor_code"]').first().closest('form');
  const scope = form.length ? form : $.root();

  const fields = [];
  scope.find('select[name]').each((_, el) => {
    const $el = $(el);
    const id = $el.attr('id');
    const rawLabel = id ? $(`label[for="${id}"]`).first().text() : '';
    const options = $el.find('option').map((__, o) => {
      const $o = $(o);
      const text = $o.text().replace(/\s+/g, ' ').trim();
      return { value: $o.attr('value') ?? text, text, selected: $o.is('[selected]') };
    }).get();

    fields.push({
      name: $el.attr('name'),
      label: rawLabel.replace(/\*/g, '').replace(/\s+/g, ' ').trim() || $el.attr('name'),
      required: $el.is('[required]') || rawLabel.includes('*'),
      options
    });
  });

  if (!fields.length) throw new Error('No form fields found on the sci.gov.in page');

  const data = { source: FORM_PAGE, fetchedAt: new Date().toISOString(), fields };
  formCache = { at: Date.now(), data };
  return data;
}

/* ==================== Routes ==================== */

app.get('/health', (_req, res) => {
  res.json({
    ok: true,
    modelLoaded: Boolean(recognizer.model.sessionPromise && recognizer.digitModel.sessionPromise)
  });
});

app.get('/api/form-fields', async (req, res) => {
  try {
    const data = await loadFormFields({ force: req.query.refresh === '1' });
    res.json({ success: true, ...data });
  } catch (error) {
    res.status(502).json({ success: false, error: error.message });
  }
});

app.get('/api/case-details', async (req, res) => {
  const diary_no = String(req.query.diary_no || '').trim();
  const diary_year = String(req.query.diary_year || '').trim();
  const tab_name = String(req.query.tab_name || '').trim();
  if (!/^\d{1,8}$/.test(diary_no) || !/^\d{4}$/.test(diary_year)) {
    return res.status(400).json({ success: false, error: 'diary_no and diary_year must be numbers' });
  }
  if (tab_name && !/^[a-z_]{1,50}$/.test(tab_name)) {
    return res.status(400).json({ success: false, error: 'Invalid tab_name' });
  }

  try {
    const query = new URLSearchParams({
      diary_no,
      diary_year,
      ...(tab_name ? { tab_name } : {}),
      action: 'get_case_details',
      es_ajax_request: '1',
      language: 'en'
    });
    const upstream = await fetch(`https://www.sci.gov.in/wp-admin/admin-ajax.php?${query}`, {
      headers: {
        accept: 'application/json, text/javascript, */*; q=0.01',
        referer: FORM_PAGE,
        'user-agent': USER_AGENT,
        'x-requested-with': 'XMLHttpRequest'
      }
    });
    const json = await upstream.json().catch(() => null);
    if (!json) {
      return res.status(502).json({ success: false, error: `sci.gov.in returned a non-JSON response (HTTP ${upstream.status})` });
    }
    res.json(json);
  } catch (error) {
    res.status(502).json({ success: false, error: error.message });
  }
});

app.post('/api/recognize', upload.single('image'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'Upload one image using the image field.' });
  if (!req.file.mimetype?.startsWith('image/')) return res.status(415).json({ error: 'The uploaded file must be an image.' });

  try {
    const result = await recognizeAndSave(req.file.buffer, req.body.expectedText);
    res.json({ success: true, ...result });
  } catch (error) {
    const status = error instanceof RangeError ? 422 : 400;
    res.status(status).json({ success: false, error: error.message });
  }
});

app.post(['/ocr-url', '/api/recognize-url'], async (req, res) => {
  const imageUrl = extractImageUrl(req.body);
  if (!imageUrl) return res.status(400).json({ success: false, error: 'Provide imageUrl' });

  try {
    const buffer = await fetchAuthorizedImage(imageUrl);
    const result = await recognizeAndSave(buffer, req.body?.expectedText);
    res.json({ success: true, ...result });
  } catch (error) {
    res.status(400).json({ success: false, error: error.message });
  }
});

/* ==================== MAIN ENDPOINT ==================== */
/* ==================== MAIN ENDPOINT ==================== */
app.post('/solve-captcha', async (req, res) => {
  try {
    let imageUrl = extractImageUrl(req.body);
    const params = pickFormParams(req.body);

    console.log('\n----- /solve-captcha called -----');
    console.log('imageUrl (from client) →', imageUrl);
    console.log('params →', params);

    // 1. Always fetch the form page to get FRESH tokens + current captcha
    const formPageRes = await fetch(FORM_PAGE, {
      headers: {
        'user-agent': USER_AGENT,
        accept: 'text/html,application/xhtml+xml'
      }
    });
    if (!formPageRes.ok) {
      throw new Error(`Could not load form page (HTTP ${formPageRes.status})`);
    }
    const formHtml = await formPageRes.text();

    // Extract dynamic tokens
    const getHidden = (name) => {
      const re = new RegExp(`name=["']${name}["'][^>]*value=["']([^"']*)["']`, 'i');
      const m = formHtml.match(re);
      return m ? m[1] : null;
    };

    const sci_form_nonce = getHidden('sci_form_nonce');
    const _form_time     = getHidden('_form_time');
    const _form_signature = getHidden('_form_signature');
    const scidFromPage   = getHidden('scid');

    // The tok_ field has a dynamic name
    const tokMatch = formHtml.match(/name=["'](tok_[a-f0-9]+)["'][^>]*value=["']([^"']+)["']/i);
    const tokName  = tokMatch ? tokMatch[1] : null;
    const tokValue = tokMatch ? tokMatch[2] : null;

    // Current captcha image URL from the page
    const captchaMatch = formHtml.match(/src=["']([^"']*_siwp_captcha[^"']*)["']/i);
    let freshCaptchaUrl = captchaMatch ? captchaMatch[1].replace(/&amp;/g, '&') : null;
    if (freshCaptchaUrl && freshCaptchaUrl.startsWith('/')) {
      freshCaptchaUrl = 'https://www.sci.gov.in' + freshCaptchaUrl;
    }

    console.log('Fresh tokens →', {
      sci_form_nonce,
      _form_time,
      _form_signature: _form_signature?.slice(0, 16) + '...',
      tokName,
      scidFromPage,
      freshCaptchaUrl
    });

    if (!sci_form_nonce || !_form_time || !_form_signature || !tokName || !tokValue || !scidFromPage) {
      return res.status(502).json({
        success: false,
        error: 'Could not extract required form tokens from sci.gov.in'
      });
    }

    // Prefer the fresh captcha from the page (more reliable)
    if (freshCaptchaUrl) {
      imageUrl = freshCaptchaUrl;
    }
    if (!imageUrl) {
      return res.status(400).json({
        success: false,
        error: 'No captcha image URL available'
      });
    }

    // 2. OCR the captcha
    const buffer = await fetchAuthorizedImage(imageUrl);
    const imageDataUrl = bufferToDataUrl(buffer);
    const ocrResult = await recognizeAndSave(buffer);

    const predicted = (ocrResult.text || '').trim();
    const calculated = evaluateExpression(predicted);
    const scid = extractScid(imageUrl) || scidFromPage;

    console.log('predicted →', predicted);
    console.log('calculated →', calculated);
    console.log('scid →', scid);

    if (calculated === null) {
      return res.json({
        success: false,
        error: 'Could not calculate expression',
        predicted,
        params,
        imageDataUrl,
        scid,
        imageUrl
      });
    }

    // 3. Submit with FRESH tokens
    const form = new URLSearchParams({
      party_type: params.party_type,
      aor_code: params.aor_code,
      year: params.year,
      case_status: params.case_status,
      scid: scid,
      [tokName]: tokValue,                    // dynamic tok_ field
      siwp_captcha_value: String(calculated),
      sci_form_nonce,
      _wp_http_referer: '/case-status-aor-code/',
      _form_time,
      _form_signature,
      es_ajax_request: '1',
      submit: 'Search',
      action: 'get_case_status_aor_code',
      language: 'en'
    });

    const ajaxRes = await fetch('https://www.sci.gov.in/wp-admin/admin-ajax.php', {
      method: 'POST',
      headers: {
        'accept': 'application/json, text/javascript, */*; q=0.01',
        'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'origin': 'https://www.sci.gov.in',
        'referer': 'https://www.sci.gov.in/case-status-aor-code/',
        'user-agent': USER_AGENT,
        'x-requested-with': 'XMLHttpRequest'
      },
      body: form.toString()
    });

    const apiResponse = await ajaxRes.json().catch(() => null);

    console.log('Upstream status →', ajaxRes.status);
    console.log('apiResponse keys →', apiResponse ? Object.keys(apiResponse) : null);

    return res.json({
      success: true,
      predicted,
      calculated,
      scid,
      params,
      imageDataUrl,
      imageUrl,
      apiStatus: ajaxRes.status,
      apiResponse,
      ocr: ocrResult
    });

  } catch (error) {
    console.error(error);
    res.status(error.status || 500).json({
      success: false,
      error: error.message || 'Something went wrong'
    });
  }
});

/* ==================== Shared ==================== */

async function recognizeAndSave(buffer, expectedText) {
  const hasVerifiedLabel = expectedText !== undefined;
  const result = await recognizer.predict(buffer, { includeGlyphBuffers: hasVerifiedLabel });

  let savedTrainingExamples = 0;
  if (hasVerifiedLabel) {
    const paths = await saveVerifiedGlyphs(result.trainingCrops, expectedText, feedbackDirectory);
    savedTrainingExamples = paths.length;
  }

  const { trainingCrops, ...publicResult } = result;
  return { ...publicResult, ...(hasVerifiedLabel ? { savedTrainingExamples } : {}) };
}

app.get('/label', (_req, res) => {
  res.sendFile(path.join(__dirname, '..', 'public', 'label.html'));
});

app.get('/api/samples', async (_req, res) => {
  try {
    const files = await listSampleImages(samplesDirectory);
    const labels = await readLabelsCsv(path.join(samplesDirectory, 'labels.csv'));
    res.json({ success: true, files, labels });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

app.get('/api/samples/image/:name', async (req, res) => {
  try {
    const filePath = resolveSampleFile(samplesDirectory, req.params.name);
    await fs.access(filePath);
    res.sendFile(filePath);
  } catch (error) {
    const status = error.code === 'ENOENT' ? 404 : 400;
    res.status(status).json({ success: false, error: error.message });
  }
});

app.post('/api/samples/labels', async (req, res) => {
  try {
    const rows = Array.isArray(req.body?.rows) ? req.body.rows : [];
    const csv = formatLabelsCsv(rows);
    await fs.mkdir(samplesDirectory, { recursive: true });
    const csvPath = path.join(samplesDirectory, 'labels.csv');
    await fs.writeFile(csvPath, csv, 'utf8');
    const savedCount = Math.max(0, csv.trim().split(/\r?\n/).length - 1);
    res.json({ success: true, savedCount, path: 'data/samples/labels.csv', csv });
  } catch (error) {
    res.status(400).json({ success: false, error: error.message });
  }
});

app.use(express.static(path.join(__dirname, '..', 'public')));

app.use((error, _req, res, _next) => {
  res.status(400).json({ error: error.message || 'Request could not be processed.' });
});

async function start(port = Number(process.env.PORT || 3000)) {
  await recognizer.load();
  return app.listen(port, () => {
    console.log(`Local CNN API listening on http://localhost:${port}`);
    console.log(`Labeling UI       → GET  http://localhost:${port}/label`);
    console.log(`New endpoint ready → POST http://localhost:${port}/solve-captcha`);
    console.log(`Form options      → GET  http://localhost:${port}/api/form-fields`);
    console.log(`Case details      → GET  http://localhost:${port}/api/case-details?diary_no=7850&diary_year=2024`);
  });
}

if (require.main === module) {
  start().catch((err) => {
    console.error(err);
    process.exitCode = 1;
  });
}

module.exports = { app, start };