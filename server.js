// ══════════════════════════════════════════════════════════════════
//  server.js  —  Salary Predictor ML  ·  Express + Gemini AI
//  ML inference runs entirely in Node.js using exported model_data.json
// ══════════════════════════════════════════════════════════════════
require('dotenv').config();
const express  = require('express');
const cors     = require('cors');
const path     = require('path');
const fs       = require('fs');
const { GoogleGenerativeAI } = require('@google/generative-ai');

const app  = express();
const PORT = process.env.PORT || 3000;

// ── Load trained model data ─────────────────────────────────────
const MODEL_PATH = path.join(__dirname, 'ml', 'model_data.json');
let MODEL = null;
try {
  MODEL = JSON.parse(fs.readFileSync(MODEL_PATH, 'utf8'));
  console.log('✅  model_data.json loaded — Bagging R²:', MODEL.bagging.r2);
} catch (e) {
  console.error('❌  Could not load ml/model_data.json. Run: cd ml && python train_model.py');
}

// ── Middleware ──────────────────────────────────────────────────
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// ── Gemini AI client ────────────────────────────────────────────
if (!process.env.GEMINI_API_KEY) {
  console.warn('\n⚠️  WARNING: GEMINI_API_KEY not found. Copy .env → .env\n');
}
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || '');

// ════════════════════════════════════════════════════════════════
//  ML INFERENCE ENGINE (pure JS — mirrors Python pipeline exactly)
// ════════════════════════════════════════════════════════════════

/** Label-encode a single categorical value using stored mapping */
function labelEncode(value, classes) {
  const v = String(value).trim();
  const idx = classes.indexOf(v);
  return idx === -1 ? 0 : idx; // unknown → 0 (most common class)
}

/** Standardize numeric features using stored scaler params */
function scaleNumeric(values, mean, std) {
  return values.map((v, i) => (v - mean[i]) / (std[i] || 1));
}

/** Build feature vector from input profile */
function buildFeatureVector(profile, model) {
  const { encoders, scaler, meta } = model;

  const numeric = [
    Number(profile.age),
    Number(profile.educationNum),
    Number(profile.capitalGain),
    Number(profile.capitalLoss),
    Number(profile.hoursPerWeek),
  ];

  const categorical = [
    labelEncode(profile.workclass,     encoders['workclass']),
    labelEncode(profile.education,     encoders['education']),
    labelEncode(profile.maritalStatus, encoders['marital.status']),
    labelEncode(profile.occupation,    encoders['occupation']),
    labelEncode(profile.relationship,  encoders['relationship']),
    labelEncode(profile.race,          encoders['race']),
    labelEncode(profile.sex,           encoders['sex']),
    labelEncode(profile.nativeCountry, encoders['native.country']),
  ];

  const scaledNumeric = scaleNumeric(numeric, scaler.mean, scaler.std);
  return [...scaledNumeric, ...categorical];
}

/** Ridge regression prediction */
function ridgePredict(features, ridge) {
  let pred = ridge.intercept;
  for (let i = 0; i < features.length; i++) pred += features[i] * ridge.coef[i];
  return pred;
}

/** Traverse a serialized decision tree */
function treePredict(features, node) {
  if (node.leaf) return node.value;
  return features[node.feature] <= node.threshold
    ? treePredict(features, node.left)
    : treePredict(features, node.right);
}

/** Bagging ensemble: average exported trees */
function baggingPredict(features, bagging) {
  const preds = bagging.trees.map(tree => treePredict(features, tree));
  return preds.reduce((a, b) => a + b, 0) / preds.length;
}

/** LDA binary classification (>50K probability) */
function ldaProba(features, lda) {
  let score = lda.intercept[0];
  for (let i = 0; i < features.length; i++) score += features[i] * lda.coef[0][i];
  return 1 / (1 + Math.exp(-score)); // sigmoid
}

/** Full salary prediction with confidence interval */
function predictSalary(profile, model) {
  const features = buildFeatureVector(profile, model);

  const ridgePred   = ridgePredict(features, model.ridge);
  const treePred    = treePredict(features, model.tree.root);
  const baggingPred = baggingPredict(features, model.bagging);
  const highIncomeProba = ldaProba(features, model.lda);

  // Ensemble: weight by inverse MAE
  const wRidge   = 1 / model.ridge.mae;
  const wTree    = 1 / model.tree.mae;
  const wBagging = 1 / model.bagging.mae;
  const wTotal   = wRidge + wTree + wBagging;

  const ensemble = (
    ridgePred   * wRidge   +
    treePred    * wTree    +
    baggingPred * wBagging
  ) / wTotal;

  // Calibrate with income class knowledge
  let salary = ensemble;
  if (highIncomeProba < 0.3 && salary > 50_000) salary = salary * 0.85;
  if (highIncomeProba > 0.7 && salary < 51_000) salary = salary * 1.15;

  // Hard clamp
  salary = Math.max(15_000, Math.min(185_000, salary));

  // Confidence interval using class-specific residual std (1-sigma ≈ 68%)
  const stdResid = salary <= 50_000 ? model.bagging.std_low : model.bagging.std_high;
  const zScore   = 1.0; // 68% CI; use 1.645 for 90% CI
  const lower    = Math.max(15_000, Math.round(salary - zScore * stdResid));
  const upper    = Math.min(185_000, Math.round(salary + zScore * stdResid));

  // Confidence: higher when models agree
  const spread    = Math.abs(ridgePred - baggingPred);
  const normSpread = Math.min(1, spread / 30_000);
  const confidence = Math.round(Math.max(55, Math.min(94, 90 - normSpread * 35)));

  // Feature importance heuristic (ranked by Ridge abs-coef × value)
  const featureNames = model.meta.features;
  const importance   = featureNames.map((name, i) => ({
    name,
    score: Math.abs(model.ridge.coef[i] * features[i]),
  })).sort((a, b) => b.score - a.score);

  return {
    estimatedSalary: Math.round(salary),
    lowerBound: lower,
    upperBound: upper,
    confidence,
    highIncomeProba: Math.round(highIncomeProba * 100),
    individual: {
      ridge:   Math.round(Math.max(15_000, Math.min(185_000, ridgePred))),
      tree:    Math.round(Math.max(15_000, Math.min(185_000, treePred))),
      bagging: Math.round(Math.max(15_000, Math.min(185_000, baggingPred))),
    },
    topFeatures: importance.slice(0, 5),
    modelMetrics: {
      ridgeR2:   model.ridge.r2,
      treeR2:    model.tree.r2,
      baggingR2: model.bagging.r2,
      ldaAcc:    model.lda.accuracy,
    },
  };
}

// ════════════════════════════════════════════════════════════════
//  ROUTES
// ════════════════════════════════════════════════════════════════

// ── POST /api/predict  ──────────────────────────────────────────
app.post('/api/predict', async (req, res) => {
  if (!MODEL) {
    return res.status(503).json({
      ok: false,
      error: 'Model not loaded. Run: cd ml && python train_model.py',
    });
  }

  const {
    age, educationNum, hoursPerWeek,
    workclass, education, maritalStatus, occupation,
    relationship, race, sex, nativeCountry,
    capitalGain = 0, capitalLoss = 0,
  } = req.body;

  if (!age || !educationNum || !hoursPerWeek) {
    return res.status(400).json({ ok: false, error: 'Missing required fields: age, educationNum, hoursPerWeek' });
  }

  // ── Step 1: ML prediction ─────────────────────────────────────
  let mlResult;
  try {
    mlResult = predictSalary(req.body, MODEL);
  } catch (err) {
    console.error('ML prediction error:', err.message);
    return res.status(500).json({ ok: false, error: 'ML prediction failed: ' + err.message });
  }

  // ── Step 2: Gemini explanation (never overrides ML numbers) ───
  let geminiResult = null;
  if (process.env.GEMINI_API_KEY) {
    const topFeaturesText = mlResult.topFeatures
      .map((f, i) => `${i + 1}. ${f.name} (importance score: ${f.score.toFixed(1)})`)
      .join('\n');

    const prompt = `You are an ML explainability assistant for a salary prediction system.

The ML model (trained on UCI Adult Census data) has already predicted a salary. Your job is ONLY to explain the prediction — you must NOT change or invent salary numbers.

ML MODEL OUTPUT (ground truth — do not alter):
- Estimated annual salary: $${mlResult.estimatedSalary.toLocaleString()}
- 68% confidence interval: $${mlResult.lowerBound.toLocaleString()} – $${mlResult.upperBound.toLocaleString()}
- Model confidence: ${mlResult.confidence}%
- Probability of >$50K income: ${mlResult.highIncomeProba}%

CANDIDATE PROFILE:
- Age: ${age}, Education years: ${educationNum}
- Occupation: ${occupation}, Workclass: ${workclass}
- Marital status: ${maritalStatus}, Relationship: ${relationship}
- Hours/week: ${hoursPerWeek}, Sex: ${sex}
- Capital gain: $${capitalGain}, Capital loss: $${capitalLoss}

TOP CONTRIBUTING FEATURES (by ML model):
${topFeaturesText}

Respond ONLY with valid JSON — no markdown, no backticks. Schema:
{
  "summary": "<2-3 sentence plain-English explanation of WHY the model predicted this salary, referencing the top factors>",
  "topFactor": "<single most influential feature name, human-readable>",
  "salaryDrivers": ["<driver 1>", "<driver 2>", "<driver 3>"],
  "careerInsight": "<one concrete, actionable recommendation to increase salary in the next 1-2 years>",
  "riskFactors": ["<factor that may lower salary 1>", "<factor 2>"]
}`;

    try {
      const model = genAI.getGenerativeModel({ model: 'gemini-3.5-flash-lite' });
      const result = await model.generateContent(prompt);
      const raw = result.response.text().trim().replace(/```json|```/g, '').trim();
      geminiResult = JSON.parse(raw);
    } catch (err) {
      console.warn('Gemini explanation error:', err.message);
      // Non-fatal — return ML result without explanation
    }
  }

  return res.json({
    ok: true,
    ml: mlResult,
    gemini: geminiResult,
  });
});

// ── GET /api/model-info  ────────────────────────────────────────
app.get('/api/model-info', (_req, res) => {
  if (!MODEL) return res.status(503).json({ ok: false });
  res.json({
    ok: true,
    metrics: MODEL && {
      ridgeR2:   MODEL.ridge.r2,
      treeR2:    MODEL.tree.r2,
      baggingR2: MODEL.bagging.r2,
      ldaAcc:    MODEL.lda.accuracy,
      ridgeMAE:  MODEL.ridge.mae,
      treeMAE:   MODEL.tree.mae,
      baggingMAE:MODEL.bagging.mae,
      nTrain:    MODEL.meta.n_train,
      salaryStats: MODEL.salary_stats,
    },
  });
});

// ── GET /api/health  ────────────────────────────────────────────
app.get('/api/health', (_req, res) => {
  res.json({
    status: 'ok',
    modelLoaded:  !!MODEL,
    geminiKeySet: !!process.env.GEMINI_API_KEY,
    timestamp:    new Date().toISOString(),
  });
});

// ── Catch-all: serve index.html ─────────────────────────────────
app.get('*', (_req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// ── Start ───────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n✅  Salary Predictor → http://localhost:${PORT}`);
  console.log(`    Model:      ${MODEL ? '✅ loaded' : '❌ missing — run: cd ml && python train_model.py'}`);
  console.log(`    Gemini key: ${process.env.GEMINI_API_KEY ? '✅ loaded' : '⚠️  missing (optional)'}\n`);
});
