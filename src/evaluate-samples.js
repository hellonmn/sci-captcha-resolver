const fs = require('node:fs/promises');
const path = require('node:path');
const { Recognizer } = require('./predict');
const { parseExpressionParts, readLabelsCsv, resolveSampleFile } = require('./samples');

const USAGE = 'Usage: npm run evaluate:samples -- [--csv data/samples/labels.csv] [--model models/model.sample_candidate.onnx] [--out results.json]';

function parseArgs(argv) {
  const options = { csv: 'data/samples/labels.csv', model: 'models/model.sample_candidate.onnx', out: null };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') options.help = true;
    else if (arg === '--csv') options.csv = argv[++i];
    else if (arg === '--model') options.model = argv[++i];
    else if (arg === '--out') options.out = argv[++i];
    else throw new Error(`Unknown option: ${arg}\n${USAGE}`);
    if (['--csv', '--model', '--out'].includes(arg) && !argv[i]) throw new Error(`${arg} requires a value.`);
  }
  return options;
}

function normalizeExpression(text) {
  return parseExpressionParts(text).expression;
}

function editDistance(reference, hypothesis) {
  let previous = Array.from({ length: hypothesis.length + 1 }, (_, i) => i);
  for (let i = 1; i <= reference.length; i++) {
    const current = [i];
    for (let j = 1; j <= hypothesis.length; j++) {
      current[j] = Math.min(
        previous[j] + 1,
        current[j - 1] + 1,
        previous[j - 1] + (reference[i - 1] === hypothesis[j - 1] ? 0 : 1)
      );
    }
    previous = current;
  }
  return previous[hypothesis.length];
}

async function evaluateSamples({ csv, model }) {
  const csvPath = path.resolve(csv);
  const samplesDir = path.dirname(csvPath);
  const labels = await readLabelsCsv(csvPath);
  const rows = Object.entries(labels);
  if (!rows.length) throw new Error(`No labeled samples found in ${csvPath}`);

  const recognizer = new Recognizer({ modelPath: path.resolve(model) });
  await recognizer.load();
  const results = [];
  let exact = 0, edits = 0, referenceCharacters = 0, confidenceTotal = 0, predictedCount = 0;

  for (const [file, rawExpected] of rows) {
    const expected = normalizeExpression(rawExpected);
    if (!expected) throw new Error(`Empty or invalid label for ${file}`);
    try {
      const imagePath = resolveSampleFile(samplesDir, file);
      const prediction = await recognizer.predict(await fs.readFile(imagePath));
      const actual = normalizeExpression(prediction.text);
      const distance = editDistance(expected, actual);
      const matched = actual === expected;
      if (matched) exact++;
      edits += distance;
      referenceCharacters += expected.length;
      const meanConfidence = prediction.predictions.length
        ? prediction.predictions.reduce((sum, item) => sum + item.confidence, 0) / prediction.predictions.length
        : 0;
      confidenceTotal += meanConfidence;
      predictedCount++;
      results.push({ file, expected, actual, exactMatch: matched, editDistance: distance, meanConfidence: Number(meanConfidence.toFixed(6)) });
    } catch (error) {
      edits += expected.length;
      referenceCharacters += expected.length;
      results.push({ file, expected, actual: '', exactMatch: false, editDistance: expected.length, error: error.message });
    }
  }

  return {
    model: path.resolve(model),
    csv: csvPath,
    evaluationScope: 'All CSV rows; this may include images used during fine-tuning and is not an independent generalization score.',
    summary: {
      total: rows.length,
      exactMatches: exact,
      exactAccuracy: exact / rows.length,
      characterErrors: edits,
      referenceCharacters,
      characterErrorRate: referenceCharacters ? edits / referenceCharacters : 0,
      averageConfidence: predictedCount ? confidenceTotal / predictedCount : 0
    },
    results
  };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) { console.log(USAGE); return; }
  const report = await evaluateSamples(options);
  if (options.out) {
    const outPath = path.resolve(options.out);
    await fs.mkdir(path.dirname(outPath), { recursive: true });
    await fs.writeFile(outPath, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
  }

  for (const row of report.results) {
    const marker = row.exactMatch ? 'PASS' : 'MISS';
    console.log(`[${marker}] ${row.file}: expected "${row.expected}", got "${row.actual}"${row.error ? ` (${row.error})` : ''}`);
  }
  const s = report.summary;
  console.log(`\nExact expression accuracy: ${s.exactMatches}/${s.total} (${(s.exactAccuracy * 100).toFixed(2)}%)`);
  console.log(`Character error rate: ${s.characterErrors}/${s.referenceCharacters} (${(s.characterErrorRate * 100).toFixed(2)}%)`);
  console.log(`Average character confidence: ${(s.averageConfidence * 100).toFixed(2)}%`);
  if (options.out) console.log(`Results saved to ${path.resolve(options.out)}`);
}

if (require.main === module) {
  main().catch((error) => { console.error(`Error: ${error.message}`); process.exitCode = 1; });
}

module.exports = { editDistance, evaluateSamples, normalizeExpression, parseArgs };
