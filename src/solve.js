const fs = require('node:fs/promises');
const path = require('node:path');
const { solveFolder } = require('./solver');

const USAGE = [
  'Usage: node src/solve.js <image-folder> [options]',
  '   or: npm run solve -- <image-folder> [options]',
  '',
  'Options:',
  '  --threshold <0..1>    Mark predictions below confidence as unsure (default: 0)',
  '  --no-recursive        Only scan top-level files in the folder',
  '  --model <path>        Override primary ONNX model path',
  '  --digit-model <path>  Override fallback MNIST ONNX model path',
  '  --out <file.json>     Save full JSON results to a file',
  '  --json                Print full JSON output to stdout',
  '  -h, --help            Show this help message'
].join('\n');

function parseArgs(argv) {
  const options = {
    folder: null,
    threshold: 0,
    recursive: true,
    json: false,
    out: null,
    modelPath: undefined,
    digitModelPath: undefined,
    help: false
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '-h' || arg === '--help') {
      options.help = true;
    } else if (arg === '--json') {
      options.json = true;
    } else if (arg === '--no-recursive') {
      options.recursive = false;
    } else if (arg === '--threshold') {
      const value = Number(argv[++i]);
      if (!Number.isFinite(value) || value < 0 || value > 1) {
        throw new Error('--threshold must be a number between 0 and 1.');
      }
      options.threshold = value;
    } else if (arg === '--model') {
      options.modelPath = argv[++i];
      if (!options.modelPath) throw new Error('--model requires a file path.');
    } else if (arg === '--digit-model') {
      options.digitModelPath = argv[++i];
      if (!options.digitModelPath) throw new Error('--digit-model requires a file path.');
    } else if (arg === '--out') {
      options.out = argv[++i];
      if (!options.out) throw new Error('--out requires a file path.');
    } else if (arg === '--folder') {
      options.folder = argv[++i];
      if (!options.folder) throw new Error('--folder requires a directory path.');
    } else if (arg.startsWith('-')) {
      throw new Error(`Unknown option: ${arg}`);
    } else if (!options.folder) {
      options.folder = arg;
    } else {
      throw new Error(`Unexpected argument: ${arg}`);
    }
  }

  return options;
}

function formatTable(report) {
  const lines = [];
  for (const item of report.results) {
    if (item.status === 'error') {
      lines.push(`[error]  ${item.file} -> ${item.error}`);
    } else if (item.status === 'unsure') {
      lines.push(`[unsure] ${item.file} -> "${item.text}" (minConf=${item.minConfidence.toFixed(3)}, ${item.processingMs.toFixed(2)} ms)`);
    } else {
      lines.push(`[solved] ${item.file} -> ${item.text} = ${item.answer} (minConf=${item.minConfidence.toFixed(3)}, ${item.processingMs.toFixed(2)} ms)`);
    }
  }

  const s = report.summary;
  lines.push('');
  lines.push(
    `Summary: total=${s.total}, solved=${s.solved}, unsure=${s.unsure}, errors=${s.errors}, ` +
    `coverage=${(s.coverage * 100).toFixed(2)}%, avg=${s.averageMs.toFixed(2)} ms, p95=${s.p95Ms.toFixed(2)} ms`
  );
  return lines.join('\n');
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    console.log(USAGE);
    return;
  }
  if (!options.folder) {
    console.error(`${USAGE}\n`);
    throw new Error('Missing required <image-folder> argument.');
  }

  const report = await solveFolder(options.folder, {
    threshold: options.threshold,
    recursive: options.recursive,
    modelPath: options.modelPath,
    digitModelPath: options.digitModelPath
  });

  if (options.out) {
    const outPath = path.resolve(options.out);
    await fs.mkdir(path.dirname(outPath), { recursive: true });
    await fs.writeFile(outPath, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
  }

  if (options.json) {
    console.log(JSON.stringify(report, null, 2));
  } else {
    console.log(formatTable(report));
  }
}

if (require.main === module) {
  main().catch((error) => {
    console.error(`Error: ${error.message}`);
    process.exitCode = 1;
  });
}

module.exports = { parseArgs, formatTable };
