const crypto = require('node:crypto');
const fs = require('node:fs/promises');
const path = require('node:path');

const DEFAULT_URL = 'https://www.sci.gov.in/?_siwp_captcha&id=ldengq206b6i02n1581wmxo3cdsww1r65286kiom';
const DEFAULT_OUT = 'data/samples';
const DEFAULT_COUNT = 20;
const DEFAULT_DELAY_MS = 250;
const ALPHABET = '0123456789abcdefghijklmnopqrstuvwxyz';
const USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/157.0.0.0 Safari/537.36';

const USAGE = [
  'Usage: node scripts/download-images.js [options]',
  '   or: npm run download-images -- [options]',
  '',
  'Options:',
  `  --url <url>      Base image URL with id parameter (default: ${DEFAULT_URL})`,
  `  --count <n>      Number of images to download (default: ${DEFAULT_COUNT})`,
  `  --out <dir>      Output folder path (default: ${DEFAULT_OUT})`,
  `  --delay <ms>     Delay between downloads in ms (default: ${DEFAULT_DELAY_MS})`,
  '  --all-random     Randomize the id on the first download too',
  '  -h, --help       Show this help message'
].join('\n');

function parseArgs(argv) {
  const options = {
    url: DEFAULT_URL,
    count: DEFAULT_COUNT,
    out: DEFAULT_OUT,
    delayMs: DEFAULT_DELAY_MS,
    allRandom: false,
    help: false
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '-h' || arg === '--help') {
      options.help = true;
    } else if (arg === '--all-random') {
      options.allRandom = true;
    } else if (arg === '--url') {
      options.url = argv[++i];
      if (!options.url) throw new Error('--url requires a URL.');
    } else if (arg === '--count') {
      const value = Number(argv[++i]);
      if (!Number.isInteger(value) || value < 1) throw new Error('--count must be a positive integer.');
      options.count = value;
    } else if (arg === '--out') {
      options.out = argv[++i];
      if (!options.out) throw new Error('--out requires a directory path.');
    } else if (arg === '--delay') {
      const value = Number(argv[++i]);
      if (!Number.isFinite(value) || value < 0) throw new Error('--delay must be >= 0.');
      options.delayMs = value;
    } else {
      throw new Error(`Unknown option: ${arg}`);
    }
  }
  return options;
}

function randomId(length = 40) {
  let out = '';
  for (let i = 0; i < length; i++) {
    out += ALPHABET[crypto.randomInt(ALPHABET.length)];
  }
  return out;
}

function extensionForMime(contentType, buffer) {
  const mime = String(contentType || '').split(';')[0].trim().toLowerCase();
  if (mime === 'image/png' || (buffer[0] === 0x89 && buffer[1] === 0x50)) return '.png';
  if (mime === 'image/jpeg' || (buffer[0] === 0xff && buffer[1] === 0xd8)) return '.jpg';
  if (mime === 'image/gif' || buffer.subarray(0, 3).toString('ascii') === 'GIF') return '.gif';
  if (mime === 'image/webp' || buffer.subarray(8, 12).toString('ascii') === 'WEBP') return '.webp';
  return '.img';
}

function buildRequestUrl(baseUrl, useOriginalId) {
  const parsed = new URL(baseUrl);
  const currentId = parsed.searchParams.get('id') || '';
  const idLength = currentId.length >= 8 ? currentId.length : 40;
  const id = useOriginalId && currentId ? currentId : randomId(idLength);
  parsed.searchParams.set('id', id);
  return { url: parsed.toString(), id };
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function downloadOne(url) {
  const response = await fetch(url, {
    headers: {
      accept: 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
      'user-agent': USER_AGENT
    },
    signal: AbortSignal.timeout(10000)
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status} ${response.statusText}`);
  }

  const contentType = response.headers.get('content-type') || '';
  if (!contentType.toLowerCase().startsWith('image/')) {
    throw new Error(`Expected image/* response but received "${contentType || 'unknown'}"`);
  }

  const buffer = Buffer.from(await response.arrayBuffer());
  if (buffer.length === 0) {
    throw new Error('Received empty image body');
  }

  return { buffer, contentType };
}

async function nextSequenceNumber(outDir) {
  const entries = await fs.readdir(outDir, { withFileTypes: true });
  let maxNumber = 0;
  for (const entry of entries) {
    if (!entry.isFile()) continue;
    const match = entry.name.match(/^(\d+)\./);
    if (match) maxNumber = Math.max(maxNumber, Number(match[1]));
  }
  return maxNumber + 1;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    console.log(USAGE);
    return;
  }

  const outDir = path.resolve(options.out);
  await fs.mkdir(outDir, { recursive: true });

  const startIndex = await nextSequenceNumber(outDir);
  const usedIds = new Set();
  let downloaded = 0;

  for (let i = 0; i < options.count; i++) {
    let target = buildRequestUrl(options.url, i === 0 && !options.allRandom);
    while (usedIds.has(target.id)) {
      target = buildRequestUrl(options.url, false);
    }
    usedIds.add(target.id);

    const { buffer, contentType } = await downloadOne(target.url);
    const ext = extensionForMime(contentType, buffer);
    const fileName = `${startIndex + i}${ext}`;
    const filePath = path.join(outDir, fileName);

    await fs.writeFile(filePath, buffer);
    downloaded++;
    console.log(`[${downloaded}/${options.count}] Saved ${fileName} (${buffer.length} bytes, id=${target.id})`);

    if (i + 1 < options.count && options.delayMs > 0) {
      await sleep(options.delayMs);
    }
  }

  console.log(`\nDone. Downloaded ${downloaded} images to ${outDir}`);
}

if (require.main === module) {
  main().catch((error) => {
    console.error(`Error: ${error.message}`);
    process.exitCode = 1;
  });
}

module.exports = { parseArgs, randomId, buildRequestUrl, nextSequenceNumber };
