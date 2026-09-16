// predictor.js — handles form submission, calls /api/predict, renders results
// Preserved from the original working SalaryIQ project.

async function runPrediction() {
  const btn = document.getElementById('btnPredict');
  btn.classList.add('loading');
  btn.querySelector('.btn-icon').textContent = '◌';

  // Collect form values
  const profile = {
    age:           parseInt(document.getElementById('age').value) || 35,
    educationNum:  parseInt(document.getElementById('eduNum').value) || 13,
    hoursPerWeek:  parseInt(document.getElementById('hours').value) || 40,
    workclass:     document.getElementById('workclass').value,
    education:     document.getElementById('education').value,
    maritalStatus: document.getElementById('marital').value,
    occupation:    document.getElementById('occupation').value,
    relationship:  document.getElementById('relationship').value,
    race:          document.getElementById('race').value,
    sex:           document.getElementById('sex').value,
    nativeCountry: document.getElementById('nativeCountry').value,
    capitalGain:   parseInt(document.getElementById('capGain').value) || 0,
    capitalLoss:   parseInt(document.getElementById('capLoss').value) || 0,
  };

  try {
    const resp = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(profile),
    });
    const data = await resp.json();

    if (!data.ok) throw new Error(data.error || 'Prediction failed');

    renderResults(data.ml, data.gemini);
  } catch (err) {
    alert('Prediction error: ' + err.message);
  } finally {
    btn.classList.remove('loading');
    btn.querySelector('.btn-icon').textContent = '◈';
  }
}

function fmt(n) {
  return '$' + Math.round(n).toLocaleString('en-US');
}

function renderResults(ml, gemini) {
  // Show results container
  document.getElementById('resultsPlaceholder').style.display = 'none';
  document.getElementById('resultsContent').style.display = 'block';

  // ── Salary card ──────────────────────────────────────────────
  document.getElementById('salaryValue').textContent = fmt(ml.estimatedSalary);
  document.getElementById('salaryLow').textContent   = fmt(ml.lowerBound);
  document.getElementById('salaryHigh').textContent  = fmt(ml.upperBound);

  // Position needle: fraction within [lowerBound, upperBound]
  const range    = ml.upperBound - ml.lowerBound;
  const fraction = range > 0 ? (ml.estimatedSalary - ml.lowerBound) / range : 0.5;
  document.getElementById('salaryFill').style.width   = '100%';
  document.getElementById('salaryNeedle').style.left  = (fraction * 100).toFixed(1) + '%';

  // ── Stats ──────────────────────────────────────────────────
  document.getElementById('confVal').textContent   = ml.confidence + '%';
  document.getElementById('hiIncProb').textContent = ml.highIncomeProba + '%';

  // ── Individual model bars (max salary = 185K) ─────────────
  const maxSal = 185_000;
  function setBar(barId, valId, salary) {
    const pct = Math.min(100, (salary / maxSal) * 100).toFixed(1);
    document.getElementById(barId).style.width = pct + '%';
    document.getElementById(valId).textContent = fmt(salary);
  }
  setBar('barRidge', 'valRidge', ml.individual.ridge);
  setBar('barTree',  'valTree',  ml.individual.tree);
  setBar('barBag',   'valBag',   ml.individual.bagging);

  // ── Feature importance ──────────────────────────────────────
  const container = document.getElementById('featureBars');
  container.innerHTML = '';
  const maxScore = ml.topFeatures[0]?.score || 1;
  ml.topFeatures.forEach(f => {
    const pct = ((f.score / maxScore) * 100).toFixed(1);
    const cleanName = f.name.replace(/\./g, ' ').replace(/-/g, ' ');
    container.innerHTML += `
      <div class="feat-row">
        <span class="feat-name">${cleanName}</span>
        <div class="feat-bar-wrap"><div class="feat-bar" style="width:${pct}%"></div></div>
        <span class="feat-score">${f.score.toFixed(0)}</span>
      </div>`;
  });

  // ── Gemini ──────────────────────────────────────────────────
  if (gemini) {
    renderGemini(gemini);
  } else {
    // Show loading indicator only if key may be set
    document.getElementById('geminiLoading').style.display = 'flex';
    document.getElementById('geminiCard').style.display = 'none';
  }

  // Scroll to results
  document.getElementById('resultsContent').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function renderGemini(g) {
  document.getElementById('geminiLoading').style.display = 'none';
  const card = document.getElementById('geminiCard');
  card.style.display = 'block';

  document.getElementById('geminiSummary').textContent    = g.summary || '';
  document.getElementById('geminiTopFactor').textContent  = g.topFactor || '—';
  document.getElementById('geminiInsight').textContent    = g.careerInsight || '—';

  const drivers = document.getElementById('geminiDrivers');
  drivers.innerHTML = '';
  const tags = [...(g.salaryDrivers || []), ...(g.riskFactors || []).map(r => '⚠ ' + r)];
  tags.forEach(t => {
    drivers.innerHTML += `<span class="driver-tag">${t}</span>`;
  });
}
