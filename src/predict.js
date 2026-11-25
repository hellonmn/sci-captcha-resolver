const { preprocess, extractGlyphs, toMnistTensor } = require('./preprocess');
const { CharacterModel, PretrainedDigitModel } = require('./model');

const DIGIT_FALLBACK_CONFIDENCE = 0.995;

class Recognizer {
  constructor(options) {
    this.model = new CharacterModel(options && options.modelPath);
    this.digitModel = new PretrainedDigitModel(options && options.digitModelPath);
  }
  async load() { await Promise.all([this.model.load(), this.digitModel.load()]); }
  async predict(buffer, options = {}) {
    const start = performance.now();
    const prepStart = performance.now();
    const glyphs = await extractGlyphs(buffer, { includeBuffers: options.includeGlyphBuffers === true });
    const tensors = glyphs.map((glyph) => glyph.tensor);
    const preprocessingMs = performance.now() - prepStart;
    const inferenceStart = performance.now();
    const primary = await this.model.predictBatch(tensors);
    const results = await Promise.all(primary.map(async (prediction, index) => {
      if (prediction.classIndex < 10 && prediction.confidence < DIGIT_FALLBACK_CONFIDENCE) {
        return this.digitModel.predict(toMnistTensor(tensors[index]));
      }
      return { ...prediction, source: 'arithmetic-cnn' };
    }));
    const inferenceMs = performance.now() - inferenceStart;
    const predictions = results.map((prediction, index) => ({ ...prediction, bounds: glyphs[index].bounds }));
    const text = predictions.map((prediction) => prediction.character).join('');
    return {
      ...(predictions.length === 1 ? predictions[0] : {}),
      text,
      predictions,
      ...(options.includeGlyphBuffers ? { trainingCrops: glyphs.map((glyph) => glyph.buffer) } : {}),
      preprocessingMs,
      inferenceMs,
      processingMs: performance.now() - start
    };
  }
  async predictBatch(buffers) {
    const preprocessingStart = performance.now();
    const tensors = await Promise.all(buffers.map(preprocess));
    const preprocessingMs = performance.now() - preprocessingStart;
    const inferenceStart = performance.now();
    const primary = await this.model.predictBatch(tensors);
    const predictions = await Promise.all(primary.map(async (prediction, index) => {
      if (prediction.classIndex < 10 && prediction.confidence < DIGIT_FALLBACK_CONFIDENCE) return this.digitModel.predict(toMnistTensor(tensors[index]));
      return { ...prediction, source: 'arithmetic-cnn' };
    }));
    return { predictions, preprocessingMs, inferenceMs: performance.now() - inferenceStart };
  }
}
module.exports = { Recognizer };
