const fs = require('node:fs');
const path = require('node:path');
const { performance } = require('node:perf_hooks');
const { Recognizer } = require('../src/predict');

const modelPath = process.env.MODEL_PATH || path.join(__dirname, '..', 'models', 'model.onnx');
const digitModelPath = process.env.DIGIT_MODEL_PATH || path.join(__dirname, '..', 'models', 'mnist-12.onnx');
const imagePath = process.env.IMAGE_PATH;
const runs = Number(process.env.RUNS || 500);
const percentile = (values, p) => [...values].sort((a,b) => a-b)[Math.ceil(p * values.length)-1];

async function main() {
  if (!imagePath || !fs.existsSync(imagePath)) throw new Error('Set IMAGE_PATH to a representative generated image.');
  const image = fs.readFileSync(imagePath);
  const recognizer = new Recognizer({ modelPath, digitModelPath });
  const rssStart = process.memoryUsage().rss;
  const loadStart = performance.now(); await recognizer.load(); const loadMs = performance.now() - loadStart;
  const prep=[], infer=[], total=[];
  for (let i=0; i<runs+20; i++) {
    const result = await recognizer.predict(image);
    if (i >= 20) { prep.push(result.preprocessingMs); infer.push(result.inferenceMs); total.push(result.processingMs); }
  }
  const report = {
    cpu: `${require('node:os').cpus().length} logical processors reported (not necessarily a 4-core test)`,
    executionProvider: 'CPU',
    modelFiles: [modelPath, digitModelPath].map((file) => ({ path: file, sizeBytes: fs.statSync(file).size })),
    modelSizeBytes: fs.statSync(modelPath).size + fs.statSync(digitModelPath).size,
    runs, warmupRuns: 20, loadMs,
    preprocessingMs: { average: prep.reduce((a,b)=>a+b,0)/runs, p95: percentile(prep,.95) },
    inferenceMs: { average: infer.reduce((a,b)=>a+b,0)/runs, p95: percentile(infer,.95) },
    totalMs: { average: total.reduce((a,b)=>a+b,0)/runs, p95: percentile(total,.95) },
    memory: { rssBeforeBytes: rssStart, rssAfterBytes: process.memoryUsage().rss, peakRssBytes: process.resourceUsage().maxRSS * 1024 }
  };
  console.log(JSON.stringify(report,null,2));
}
main().catch(e=>{ console.error(e.message); process.exitCode=1; });
