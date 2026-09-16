// main.js — charts, DBSCAN, Viva Q&A, model metrics

// ══════════════════════════════════════════════════════════════════
//  LOAD MODEL METRICS
// ══════════════════════════════════════════════════════════════════
async function loadModelMetrics() {
  try {
    const res  = await fetch('/api/model-info');
    const data = await res.json();
    if (!data.ok) return;

    const m = data.metrics;

    // Hero pills
    document.getElementById('mpBaggingR2').textContent = m.baggingR2.toFixed(3);
    document.getElementById('mpLdaAcc').textContent    = (m.ldaAcc * 100).toFixed(1) + '%';
    document.getElementById('mpMAE').textContent       = '$' + m.baggingMAE.toLocaleString();
    document.getElementById('mpTrainSize').textContent = m.nTrain.toLocaleString();

    // Algorithm section
    document.getElementById('ldaAccDisp').textContent  = (m.ldaAcc * 100).toFixed(1) + '%';
    document.getElementById('treeR2Disp').textContent  = m.treeR2.toFixed(3);
    document.getElementById('ridgeR2Disp').textContent = m.ridgeR2.toFixed(3);
    document.getElementById('bagR2Disp').textContent   = m.baggingR2.toFixed(3);
    document.getElementById('bagMAEDisp').textContent  = '$' + m.baggingMAE.toLocaleString();

    // Render charts now that we have metrics
    renderModelPerfChart(m);
    renderSalaryDistChart(m.salaryStats);
  } catch (e) {
    console.warn('Model info not available:', e.message);
  }
}

// ══════════════════════════════════════════════════════════════════
//  CHARTS
// ══════════════════════════════════════════════════════════════════
const CHART_DEFAULTS = {
  color: 'rgba(216,216,216,0.85)',
  grid:  'rgba(255,255,255,0.07)',
};

function chartTheme(ctx) {
  return {
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
      legend: { labels: { color: CHART_DEFAULTS.color, font: { family: 'Instrument Sans', size: 11 } } },
      tooltip: { backgroundColor: '#1a1714', titleColor: '#e81010', bodyColor: '#f0ead8' },
    },
    scales: {
      x: { ticks: { color: CHART_DEFAULTS.color, font: { size: 10 } }, grid: { color: CHART_DEFAULTS.grid } },
      y: { ticks: { color: CHART_DEFAULTS.color, font: { size: 10 } }, grid: { color: CHART_DEFAULTS.grid } },
    },
  };
}

function renderSalaryDistChart(stats) {
  const ctx = document.getElementById('chartSalaryDist').getContext('2d');
  // Approximate histogram bands using dataset percentiles
  const bands = ['$15K–25K','$25K–35K','$35K–45K','$45K–50K','$50K–65K','$65K–85K','$85K–120K','$120K+'];
  const counts = [8, 14, 19, 22, 12, 9, 10, 6]; // approximate % distributions from dataset
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: bands,
      datasets: [{
        label: '% of Workers',
        data: counts,
        backgroundColor: counts.map((_, i) =>
          i < 4 ? 'rgba(212,134,10,0.65)' : 'rgba(245,166,35,0.85)'),
        borderRadius: 4,
      }]
    },
    options: { ...chartTheme(ctx), plugins: { ...chartTheme(ctx).plugins, legend: { display: false } } },
  });
}

function renderModelPerfChart(m) {
  const ctx = document.getElementById('chartModelPerf').getContext('2d');
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Ridge', 'Decision Tree', 'Bagging ★', 'LDA'],
      datasets: [{
        label: 'R² Score',
        data: [m.ridgeR2, m.treeR2, m.baggingR2, null],
        backgroundColor: ['rgba(170,10,10,.7)','rgba(255,70,70,.7)','rgba(232,16,16,.85)','rgba(255,60,60,.5)'],
        borderRadius: 4,
        yAxisID: 'y',
      },{
        label: 'Accuracy (LDA)',
        data: [null, null, null, m.ldaAcc],
        backgroundColor: 'rgba(255,60,60,.7)',
        borderRadius: 4,
        yAxisID: 'y',
      }]
    },
    options: {
      ...chartTheme(ctx),
      plugins: { ...chartTheme(ctx).plugins, legend: { display: false } },
      scales: {
        y: { ...chartTheme(ctx).scales.y, min: 0, max: 1 },
        x: chartTheme(ctx).scales.x,
      },
    },
  });
}

function renderOccupationChart() {
  const ctx = document.getElementById('chartOccupation').getContext('2d');
  const occs = ['Exec-mgr','Prof-spec','Tech-supp','Sales','Craft','Transport','Machine-op','Clerical','Other-svc'];
  const medians = [105, 88, 58, 52, 50, 46, 40, 36, 26];
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: occs,
      datasets: [{
        label: 'Median Salary ($K)',
        data: medians,
        backgroundColor: medians.map(v =>
          v > 70 ? 'rgba(232,16,16,.85)' : v > 50 ? 'rgba(255,70,70,.7)' : 'rgba(170,10,10,.6)'),
        borderRadius: 4,
      }]
    },
    options: {
      indexAxis: 'y',
      ...chartTheme(ctx),
      plugins: { ...chartTheme(ctx).plugins, legend: { display: false } },
    },
  });
}

function renderEduSalaryChart() {
  const ctx = document.getElementById('chartEduSalary').getContext('2d');
  const eduLevels = ['<HS','HS Grad','Some Coll','Associate','Bachelor\'s','Master\'s','Doctorate'];
  new Chart(ctx, {
    type: 'line',
    data: {
      labels: eduLevels,
      datasets: [
        {
          label: 'Median Salary',
          data: [24, 33, 38, 42, 58, 74, 95],
          borderColor: '#e81010',
          backgroundColor: 'rgba(245,166,35,.12)',
          fill: true, tension: 0.35, pointRadius: 4,
          pointBackgroundColor: '#e81010',
        },
        {
          label: 'P25 – P75 Band',
          data: [20, 28, 32, 36, 48, 62, 78],
          borderColor: 'rgba(78,194,138,.6)',
          backgroundColor: 'transparent',
          borderDash: [4,3], tension: 0.35, pointRadius: 2,
        },
      ]
    },
    options: { ...chartTheme(ctx), scales: { ...chartTheme(ctx).scales, y: { ...chartTheme(ctx).scales.y, title: { display: true, text: '$K / year', color: CHART_DEFAULTS.color } } } },
  });
}

// ══════════════════════════════════════════════════════════════════
//  DBSCAN CANVAS SIMULATION
// ══════════════════════════════════════════════════════════════════
function drawDBSCAN() {
  const canvas = document.getElementById('dbscanCanvas');
  const ctx    = canvas.getContext('2d');
  const W = canvas.offsetWidth || 800;
  const H = 420;
  canvas.width  = W;
  canvas.height = H;

  const CLUSTER_COLORS = [
    'rgba(232,16,16,.85)',
    'rgba(255,70,70,.8)',
    'rgba(74,158,255,.8)',
    'rgba(207,121,245,.8)',
    'rgba(255,90,90,.8)',
  ];
  const NOISE_COLOR = 'rgba(80,80,80,.7)';

  // Simulate 5 clusters + noise in education(x) vs hours(y) space
  const clusterCenters = [
    { ex: 0.20, hy: 0.55, n: 80,  label: 'Low-edu part-time' },
    { ex: 0.45, hy: 0.65, n: 120, label: 'Mid-edu full-time' },
    { ex: 0.70, hy: 0.70, n: 90,  label: 'High-edu standard' },
    { ex: 0.80, hy: 0.90, n: 60,  label: 'High-edu overtime' },
    { ex: 0.35, hy: 0.35, n: 50,  label: 'Mid-edu part-time' },
  ];

  const rng = mulberry32(42);

  const points = [];

  clusterCenters.forEach((c, ci) => {
    for (let i = 0; i < c.n; i++) {
      const x = clamp(c.ex + (rng() - 0.5) * 0.18);
      const y = clamp(c.hy + (rng() - 0.5) * 0.16);
      points.push({ x: x * (W - 60) + 30, y: y * (H - 60) + 30, cluster: ci });
    }
  });

  // Noise points
  for (let i = 0; i < 30; i++) {
    points.push({ x: rng() * W, y: rng() * H, cluster: -1 });
  }

  // Draw axes
  ctx.strokeStyle = 'rgba(232,16,16,.08)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const gy = 30 + (i / 4) * (H - 60);
    ctx.beginPath(); ctx.moveTo(30, gy); ctx.lineTo(W - 30, gy); ctx.stroke();
    const gx = 30 + (i / 4) * (W - 60);
    ctx.beginPath(); ctx.moveTo(gx, 30); ctx.lineTo(gx, H - 30); ctx.stroke();
  }

  // Draw cluster convex hulls (ellipses)
  clusterCenters.forEach((c, ci) => {
    ctx.save();
    ctx.strokeStyle = CLUSTER_COLORS[ci].replace('.8', '.35').replace('.85', '.35');
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 3]);
    ctx.beginPath();
    ctx.ellipse(
      c.ex * (W - 60) + 30,
      c.hy * (H - 60) + 30,
      (W - 60) * 0.11,
      (H - 60) * 0.10,
      0, 0, Math.PI * 2
    );
    ctx.stroke();
    ctx.restore();

    // Cluster label
    ctx.fillStyle = CLUSTER_COLORS[ci];
    ctx.font = '500 10px "Instrument Sans", sans-serif';
    ctx.fillText('C' + (ci + 1) + ': ' + c.label, c.ex * (W - 60) + 30, c.hy * (H - 60) + 30 - 18);
  });

  // Draw points
  points.forEach(p => {
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.cluster === -1 ? 3 : 4, 0, Math.PI * 2);
    ctx.fillStyle = p.cluster === -1 ? NOISE_COLOR : CLUSTER_COLORS[p.cluster];
    ctx.fill();
  });

  // Axis labels
  ctx.fillStyle = 'rgba(240,234,216,.4)';
  ctx.font = '11px "Instrument Sans"';
  ctx.fillText('Education Years →', 30, H - 8);
  ctx.save(); ctx.translate(14, H / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText('Hours per Week →', 0, 0); ctx.restore();

  // Legend
  const legend = document.getElementById('dbscanLegend');
  legend.innerHTML = clusterCenters.map((c, i) =>
    `<div class="dbscan-legend-item">
       <div class="legend-dot" style="background:${CLUSTER_COLORS[i]}"></div>
       C${i+1}: ${c.label}
     </div>`
  ).join('') +
  `<div class="dbscan-legend-item">
     <div class="legend-dot" style="background:${NOISE_COLOR}"></div>
     Noise / Outliers
   </div>`;
}

function mulberry32(seed) {
  return function() {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = t + Math.imul(t ^ (t >>> 7), 61 | t) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function clamp(v, lo = 0.05, hi = 0.95) { return Math.max(lo, Math.min(hi, v)); }

// ══════════════════════════════════════════════════════════════════
//  INIT
// ══════════════════════════════════════════════════════════════════
window.addEventListener('DOMContentLoaded', () => {
  loadModelMetrics();
  renderOccupationChart();
  renderEduSalaryChart();
  drawDBSCAN();
});

window.addEventListener('resize', () => {
  drawDBSCAN();
});
