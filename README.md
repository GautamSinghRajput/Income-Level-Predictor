Income Level Predictor — Numeric Salary Prediction Engine
## B.Tech CSE End-to-End Machine Learning Project

A full-stack ML web application that predicts **actual dollar salary values** (not just income classes) using an ensemble of five algorithms trained on the UCI Adult Census dataset, with optional Gemini AI explanations.

---

## ✦ What's New vs. Original Project

| Feature | Original | This Version |
|---|---|---|
| Output | Binary (≤50K / >50K) | **Numeric salary + CI** |
| Target | Classification | **Regression** |
| Models used | Rule-based scoring | **Real trained ML weights** |
| Confidence | Heuristic % | **Residual-based statistical CI** |
| AI role | Generates predictions | **Explains ML predictions only** |
| Prediction output | Tier label | **`{estimatedSalary, lowerBound, upperBound, confidence}`** |

---

## ⚡ Quick Start

### Step 1 — Install Python dependencies & train models

```bash
cd ml
pip install -r requirements.txt
python train_model.py
```

This generates `ml/model_data.json` (~234 KB) with all trained model weights.

### Step 2 — Install Node.js dependencies

```bash
cd ..          # back to project root
npm install
```

### Step 3 — Configure environment (optional — Gemini is optional)

```bash
cp .env .env
# Edit .env and add your Gemini API key if you want AI explanations
# Get a FREE key at: https://aistudio.google.com/app/apikey
```

### Step 4 — Start the server

```bash
npm start          # production
npm run dev        # development (auto-reload with nodemon)
```

### Step 5 — Open in browser

```
http://localhost:3000
```

---

## 🗂 Project Structure

```
salaryiq/
├── public/
│   ├── index.html              # Full-page UI (predictor + charts + viva Q&A)
│   ├── css/style.css           # Complete styling
│   └── js/
│       ├── predictor.js        # Form submit, result rendering
│       └── main.js             # Charts, DBSCAN canvas, Viva Q&A
├── ml/
│   ├── train_model.py          # ← RUN THIS FIRST
│   ├── model_data.json         # Generated — Ridge/Tree/Bagging/LDA weights
│   └── requirements.txt        # Python deps
├── server.js                   # Express server + ML inference + Gemini
├── package.json
├── adult.csv                   # UCI Adult dataset (30,162 rows)
├── .env.example                # Copy → .env
└── README.md
```

---

## 🤖 How Prediction Works

```
User Input → Feature Engineering → Ensemble ML Prediction → Gemini Explanation
                                         ↓
                             [Ridge + DecisionTree + Bagging]
                                  weighted average
                                         ↓
                              LDA calibration (>$50K prob)
                                         ↓
                              Salary + CI + Confidence %
                                         ↓
                            Gemini explains (never overrides)
```

### ML Inference (Node.js, server.js)

1. **Feature encoding** — categorical features label-encoded using stored mappings
2. **Normalization** — numeric features standardized with stored scaler (μ, σ)
3. **Ridge Regression** — linear prediction using stored coefficients
4. **Decision Tree** — recursive JSON tree traversal
5. **Bagging** — average of 5 exported bootstrap trees
6. **LDA** — sigmoid of linear score → P(income > $50K)
7. **Ensemble** — weighted average by inverse MAE
8. **Calibration** — LDA probability nudges estimate up/down
9. **Confidence Interval** — ±1σ using test-set residual std

---

## 📊 Model Performance

| Model | Type | R² Score | MAE |
|---|---|---|---|
| LDA | Classification | — | 80.9% accuracy |
| Ridge Regression | Linear | 0.413 | ~$20K |
| Decision Tree | Tree | 0.705 | ~$11K |
| **Bagging ★** | **Ensemble** | **0.714** | **~$11K** |

---

## 🛠 API Endpoints

### POST `/api/predict`

```json
{
  "age": 35,
  "educationNum": 13,
  "hoursPerWeek": 45,
  "workclass": "Private",
  "education": "Bachelors",
  "maritalStatus": "Married-civ-spouse",
  "occupation": "Exec-managerial",
  "relationship": "Husband",
  "race": "White",
  "sex": "Male",
  "nativeCountry": "United-States",
  "capitalGain": 0,
  "capitalLoss": 0
}
```

Response:

```json
{
  "ok": true,
  "ml": {
    "estimatedSalary": 87500,
    "lowerBound": 61000,
    "upperBound": 114000,
    "confidence": 82,
    "highIncomeProba": 78,
    "individual": { "ridge": 72000, "tree": 95000, "bagging": 90000 },
    "topFeatures": [ ... ]
  },
  "gemini": {
    "summary": "Your executive occupation combined with...",
    "topFactor": "Occupation",
    "careerInsight": "...",
    "salaryDrivers": [...],
    "riskFactors": [...]
  }
}
```

### GET `/api/model-info`
Returns R², MAE, accuracy metrics and salary distribution statistics.

### GET `/api/health`
Server status, model loaded flag, Gemini key presence.

---

## 🎓 ML Algorithms Covered

| # | Algorithm | Type | Role in Project |
|---|---|---|---|
| 1 | LDA | Supervised / Classification | Calibrates regression via P(>$50K) |
| 2 | Decision Tree | Regression Tree | Individual salary estimate |
| 3 | Ridge Regression | Regularized Linear | Interpretable linear estimate |
| 4 | Bagging | Ensemble | Primary prediction backbone |
| 5 | DBSCAN | Unsupervised Clustering | Visualizes natural income groups |

---

## 🔒 Security

- `.env` is git-ignored — your Gemini API key is never committed
- Gemini is called server-side only — API key is never exposed to frontend
- ML inference runs in Node.js — no Python required at runtime
