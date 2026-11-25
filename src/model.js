const fs = require('node:fs');
const path = require('node:path');
const ort = require('onnxruntime-node');

class CharacterModel {
  constructor(modelPath = path.join(__dirname, '..', 'models', 'model.onnx')) {
    this.modelPath = modelPath;
    this.sessionPromise = null;
    this.classes = ['0','1','2','3','4','5','6','7','8','9','+','-','×','÷','='];
  }
  async load() {
    if (!this.sessionPromise) {
      this.sessionPromise = (async () => {
        if (!fs.existsSync(this.modelPath)) throw new Error(`Model file not found: ${this.modelPath}. Train/export it first.`);
        const classPath = this.modelPath.replace(/\.onnx$/i, '.classes.json');
        if (fs.existsSync(classPath)) this.classes = JSON.parse(fs.readFileSync(classPath, 'utf8'));
        return ort.InferenceSession.create(this.modelPath, { executionProviders: ['cpu'], intraOpNumThreads: 4, interOpNumThreads: 1 });
      })().catch((error) => { this.sessionPromise = null; throw error; });
    }
    return this.sessionPromise;
  }
  async predictBatch(items) {
    if (!Array.isArray(items) || items.length < 1) throw new TypeError('Expected at least one input tensor');
    const session = await this.load();
    const input = new Float32Array(items.length * 32 * 32);
    items.forEach((item, i) => {
      if (!item || item.length !== 1024) throw new TypeError('Each input must contain 1024 normalized pixels');
      input.set(item, i * 1024);
    });
    const result = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', input, [items.length, 1, 32, 32]) });
    const scores = result[session.outputNames[0]].data;
    const n = this.classes.length;
    return items.map((_, row) => {
      let max = -Infinity, index = 0;
      for (let j = 0; j < n; j++) { const v = scores[row * n + j]; if (v > max) { max = v; index = j; } }
      let denom = 0;
      for (let j = 0; j < n; j++) denom += Math.exp(scores[row * n + j] - max);
      return { character: this.classes[index], classIndex: index, confidence: 1 / denom };
    });
  }
  async predict(tensor) { return (await this.predictBatch([tensor]))[0]; }
}

class PretrainedDigitModel {
  constructor(modelPath = path.join(__dirname, '..', 'models', 'mnist-12.onnx')) {
    this.modelPath = modelPath;
    this.sessionPromise = null;
  }
  async load() {
    if (!this.sessionPromise) {
      this.sessionPromise = (async () => {
        if (!fs.existsSync(this.modelPath)) throw new Error(`Pretrained digit model not found: ${this.modelPath}`);
        return ort.InferenceSession.create(this.modelPath, { executionProviders: ['cpu'], intraOpNumThreads: 2, interOpNumThreads: 1 });
      })().catch((error) => { this.sessionPromise = null; throw error; });
    }
    return this.sessionPromise;
  }
  async predict(tensor28) {
    if (!tensor28 || tensor28.length !== 28 * 28) throw new TypeError('Digit input must contain 784 pixels');
    const session = await this.load();
    const result = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', tensor28, [1, 1, 28, 28]) });
    const scores = result[session.outputNames[0]].data;
    let max = -Infinity, index = 0;
    for (let i = 0; i < 10; i++) if (scores[i] > max) { max = scores[i]; index = i; }
    let denom = 0;
    for (let i = 0; i < 10; i++) denom += Math.exp(scores[i] - max);
    return { character: String(index), classIndex: index, confidence: 1 / denom, source: 'pretrained-mnist' };
  }
}

module.exports = { CharacterModel, PretrainedDigitModel };
