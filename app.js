const STORAGE_KEY = "bodyEvalMvpRecords";
const MIN_SCORE = 0.25;

const KEYPOINT_NAMES = {
  nose: "鼻",
  left_ear: "左耳",
  right_ear: "右耳",
  left_shoulder: "左肩",
  right_shoulder: "右肩",
  left_hip: "左股関節",
  right_hip: "右股関節",
  left_knee: "左膝",
  right_knee: "右膝",
  left_ankle: "左足首",
  right_ankle: "右足首"
};

const METRICS = {
  neck: {
    label: "首疲労：頭部前方偏位",
    unit: "%",
    manual: ["耳", "肩"],
    autoKeys: ["nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder"],
    calc(points) {
      const ear = midpoint(points.left_ear, points.right_ear) || points.nose || points.ear;
      const shoulder = midpoint(points.left_shoulder, points.right_shoulder) || points.shoulder;
      const shoulderWidth = distance(points.left_shoulder, points.right_shoulder) || imageScale(points);
      return Math.abs(ear.x - shoulder.x) / shoulderWidth * 100;
    }
  },
  shoulder: {
    label: "肩疲労：肩の高さ左右差",
    unit: "%",
    manual: ["左肩", "右肩"],
    autoKeys: ["left_shoulder", "right_shoulder"],
    calc(points) {
      const width = distance(points.left_shoulder, points.right_shoulder) || imageScale(points);
      return Math.abs(points.left_shoulder.y - points.right_shoulder.y) / width * 100;
    }
  },
  lowerBack: {
    label: "腰疲労：体幹の傾き",
    unit: "度",
    manual: ["胸中央", "骨盤中央"],
    autoKeys: ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    calc(points) {
      const chest = midpoint(points.left_shoulder, points.right_shoulder) || points.chest;
      const pelvis = midpoint(points.left_hip, points.right_hip) || points.pelvis;
      const dx = chest.x - pelvis.x;
      const dy = Math.abs(chest.y - pelvis.y);
      return Math.abs(Math.atan2(dx, dy) * 180 / Math.PI);
    }
  },
  knee: {
    label: "膝疲労：膝角度",
    unit: "度",
    manual: ["股関節", "膝", "足首"],
    autoKeys: ["left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle"],
    calc(points) {
      if (points.hip && points.knee && points.ankle) return angle(points.hip, points.knee, points.ankle);
      const leftScore = confidenceSum(points, ["left_hip", "left_knee", "left_ankle"]);
      const rightScore = confidenceSum(points, ["right_hip", "right_knee", "right_ankle"]);
      const side = leftScore >= rightScore ? "left" : "right";
      return angle(points[side + "_hip"], points[side + "_knee"], points[side + "_ankle"]);
    }
  }
};

const state = {
  detector: null,
  modelReady: false,
  latestResult: null,
  before: makeSideState("before"),
  after: makeSideState("after")
};

const els = {
  modelStatus: document.getElementById("modelStatus"),
  messageArea: document.getElementById("messageArea"),
  privacyConfirm: document.getElementById("privacyConfirm"),
  themeSelect: document.getElementById("themeSelect"),
  modeSelect: document.getElementById("modeSelect"),
  guide: document.getElementById("guide"),
  beforeCanvas: document.getElementById("beforeCanvas"),
  afterCanvas: document.getElementById("afterCanvas"),
  beforeStatus: document.getElementById("beforeStatus"),
  afterStatus: document.getElementById("afterStatus"),
  runAutoBtn: document.getElementById("runAutoBtn"),
  compareBtn: document.getElementById("compareBtn"),
  saveBtn: document.getElementById("saveBtn"),
  csvBtn: document.getElementById("csvBtn"),
  resultBox: document.getElementById("resultBox"),
  historyBox: document.getElementById("historyBox"),
  clearAllDataBtn: document.getElementById("clearAllDataBtn"),
  clearRecordsBtn: document.getElementById("clearRecordsBtn")
};

function makeSideState(side) {
  return {
    side,
    image: null,
    imageName: "",
    points: {},
    manualPoints: [],
    source: "none",
    canvas: null,
    ctx: null,
    pose: null
  };
}

function boot() {
  state.before.canvas = els.beforeCanvas;
  state.after.canvas = els.afterCanvas;
  state.before.ctx = els.beforeCanvas.getContext("2d");
  state.after.ctx = els.afterCanvas.getContext("2d");

  bindFile("beforeCamera", "before");
  bindFile("beforeFile", "before");
  bindFile("afterCamera", "after");
  bindFile("afterFile", "after");

  els.themeSelect.addEventListener("change", resetForMetric);
  els.modeSelect.addEventListener("change", updateGuide);
  els.runAutoBtn.addEventListener("click", runAutoForBoth);
  els.compareBtn.addEventListener("click", () => compare(true));
  els.saveBtn.addEventListener("click", saveLatest);
  els.csvBtn.addEventListener("click", exportCsv);
  els.clearAllDataBtn.addEventListener("click", clearAllDeviceData);
  els.clearRecordsBtn.addEventListener("click", clearRecords);

  document.querySelectorAll("[data-action]").forEach((button) => {
    button.addEventListener("click", () => {
      const side = button.dataset.side;
      if (button.dataset.action === "undo") undoPoint(side);
      if (button.dataset.action === "clear") clearPoints(side);
    });
  });

  ["before", "after"].forEach((side) => {
    state[side].canvas.addEventListener("pointerdown", (event) => addManualPoint(side, event));
  });

  updateGuide();
  updateStatus();
  drawPlaceholder(state.before);
  drawPlaceholder(state.after);
  renderHistory();
  initModel();
}

async function initModel() {
  setModelStatus("モデル読込中", "");
  try {
    if (!window.tf || !window.poseDetection) {
      throw new Error("TensorFlow.jsまたはpose-detectionを読み込めませんでした。通信環境を確認してください。");
    }
    await tf.setBackend("webgl");
    await tf.ready();
    state.detector = await poseDetection.createDetector(
      poseDetection.SupportedModels.MoveNet,
      { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
    );
    state.modelReady = true;
    setModelStatus("自動推定OK", "ok");
    showMessage("自動骨格推定を使えます。写真を入れた後に「骨格推定する」を押してください。", "info");
  } catch (error) {
    state.modelReady = false;
    setModelStatus("手動モード", "error");
    showMessage("自動推定を準備できませんでした。手動タップで評価できます。詳細：" + error.message, "error");
  }
}

function bindFile(inputId, side) {
  document.getElementById(inputId).addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0];
    if (file && requirePrivacyConfirmation()) loadImage(file, side);
    event.target.value = "";
  });
}

function loadImage(file, side) {
  const reader = new FileReader();
  reader.onload = () => {
    const image = new Image();
    image.onload = async () => {
      const target = state[side];
      target.image = image;
      target.imageName = file.name || side + " photo";
      target.points = {};
      target.manualPoints = [];
      target.pose = null;
      target.source = "image";
      state.latestResult = null;
      resizeCanvas(target);
      redraw(target);
      updateStatus();
      updateResultPrompt();
      if (els.modeSelect.value === "auto" && state.modelReady) await runAuto(side);
    };
    image.onerror = () => showMessage("画像を読み込めませんでした。別の写真で試してください。", "error");
    image.src = reader.result;
  };
  reader.onerror = () => showMessage("写真ファイルを読み込めませんでした。", "error");
  reader.readAsDataURL(file);
}

function resizeCanvas(target) {
  const maxWidth = 960;
  const maxHeight = 720;
  const scale = Math.min(maxWidth / target.image.naturalWidth, maxHeight / target.image.naturalHeight, 1);
  target.canvas.width = Math.max(320, Math.round(target.image.naturalWidth * scale));
  target.canvas.height = Math.max(240, Math.round(target.image.naturalHeight * scale));
}

async function runAutoForBoth() {
  if (!requirePrivacyConfirmation()) return;
  if (!state.modelReady) {
    showMessage("自動推定モデルが使えないため、手動タップで入力してください。", "warn");
    return;
  }
  await runAuto("before");
  await runAuto("after");
}

async function runAuto(side) {
  const target = state[side];
  if (!target.image) {
    showMessage((side === "before" ? "Before" : "After") + "写真を先に入れてください。", "warn");
    return;
  }
  try {
    redraw(target);
    setSideStatus(side, "骨格推定中");
    const poses = await state.detector.estimatePoses(target.canvas, { maxPoses: 1, flipHorizontal: false });
    const pose = poses && poses[0];
    if (!pose || !pose.keypoints || pose.keypoints.length === 0) {
      throw new Error("人物を検出できませんでした。全身が写った写真で試すか、手動タップしてください。");
    }
    target.pose = pose;
    target.points = keypointsToMap(pose.keypoints);
    target.manualPoints = [];
    target.source = "auto";
    state.latestResult = null;
    redraw(target);
    updateStatus();
    autoCompareIfReady();
  } catch (error) {
    target.source = "manual";
    redraw(target);
    updateStatus();
    showMessage((side === "before" ? "Before" : "After") + "の自動推定に失敗しました。手動タップで続けられます。詳細：" + error.message, "error");
  }
}

function keypointsToMap(keypoints) {
  const map = {};
  keypoints.forEach((point) => {
    map[point.name] = {
      x: point.x,
      y: point.y,
      score: point.score || 0
    };
  });
  return map;
}

function addManualPoint(side, event) {
  if (!requirePrivacyConfirmation()) return;
  const target = state[side];
  if (!target.image) {
    showMessage((side === "before" ? "Before" : "After") + "写真を先に入れてください。", "warn");
    return;
  }
  const metric = METRICS[els.themeSelect.value];
  const max = metric.manual.length;
  if (target.manualPoints.length >= max) {
    showMessage("必要な点は入力済みです。やり直す場合は「1点戻す」または「点を削除」を使ってください。", "warn");
    return;
  }
  const rect = target.canvas.getBoundingClientRect();
  const point = {
    x: (event.clientX - rect.left) * target.canvas.width / rect.width,
    y: (event.clientY - rect.top) * target.canvas.height / rect.height,
    score: 1
  };
  target.manualPoints.push(point);
  target.points = manualPointsToMap(target.manualPoints, metric);
  target.source = "manual";
  state.latestResult = null;
  redraw(target);
  updateStatus();
  autoCompareIfReady();
}

function manualPointsToMap(points, metric) {
  const map = {};
  if (metric === METRICS.neck) {
    map.ear = points[0];
    map.shoulder = points[1];
  } else if (metric === METRICS.shoulder) {
    map.left_shoulder = points[0];
    map.right_shoulder = points[1];
  } else if (metric === METRICS.lowerBack) {
    map.chest = points[0];
    map.pelvis = points[1];
  } else if (metric === METRICS.knee) {
    map.hip = points[0];
    map.knee = points[1];
    map.ankle = points[2];
  }
  map.__scale = { x: 0, y: 0, width: state.before.canvas.width || 900, height: state.before.canvas.height || 680 };
  return map;
}

function undoPoint(side) {
  const target = state[side];
  target.manualPoints.pop();
  target.points = manualPointsToMap(target.manualPoints, METRICS[els.themeSelect.value]);
  target.source = target.manualPoints.length ? "manual" : "image";
  state.latestResult = null;
  redraw(target);
  updateStatus();
  updateResultPrompt();
}

function clearPoints(side) {
  const target = state[side];
  target.points = {};
  target.manualPoints = [];
  target.pose = null;
  target.source = target.image ? "image" : "none";
  state.latestResult = null;
  redraw(target);
  updateStatus();
  updateResultPrompt();
}

function resetForMetric() {
  ["before", "after"].forEach((side) => {
    state[side].points = {};
    state[side].manualPoints = [];
    state[side].pose = null;
    state[side].source = state[side].image ? "image" : "none";
    redraw(state[side]);
  });
  state.latestResult = null;
  updateGuide();
  updateStatus();
  updateResultPrompt();
}

function compare(showWarning) {
  if (!requirePrivacyConfirmation()) return null;
  const metric = METRICS[els.themeSelect.value];
  const beforeReady = hasRequiredPoints(state.before, metric);
  const afterReady = hasRequiredPoints(state.after, metric);
  if (!beforeReady || !afterReady) {
    if (showWarning) showMessage("BeforeとAfterの両方で必要な身体ポイントを入力してください。", "warn");
    updateResultPrompt();
    return null;
  }

  const beforeValue = metric.calc(state.before.points);
  const afterValue = metric.calc(state.after.points);
  if (!Number.isFinite(beforeValue) || !Number.isFinite(afterValue)) {
    showMessage("評価値を計算できませんでした。点の位置を確認してください。", "error");
    return null;
  }

  const change = afterValue - beforeValue;
  state.latestResult = {
    date: new Date().toLocaleString("ja-JP"),
    theme: metric.label,
    before: round(beforeValue),
    after: round(afterValue),
    change: round(change),
    unit: metric.unit,
    beforeSource: state.before.source,
    afterSource: state.after.source
  };
  renderResult(state.latestResult);
  return state.latestResult;
}

function autoCompareIfReady() {
  if (hasRequiredPoints(state.before, METRICS[els.themeSelect.value]) && hasRequiredPoints(state.after, METRICS[els.themeSelect.value])) {
    compare(false);
  } else {
    updateResultPrompt();
  }
}

function hasRequiredPoints(target, metric) {
  if (!target.image) return false;
  if (target.source === "manual") return target.manualPoints.length >= metric.manual.length;
  const metricKey = els.themeSelect.value;
  if (metricKey === "neck") {
    return (isReliable(target.points.nose) || (isReliable(target.points.left_ear) && isReliable(target.points.right_ear))) &&
      isReliable(target.points.left_shoulder) &&
      isReliable(target.points.right_shoulder);
  }
  if (metricKey === "knee") {
    const leftOk = ["left_hip", "left_knee", "left_ankle"].every((key) => isReliable(target.points[key]));
    const rightOk = ["right_hip", "right_knee", "right_ankle"].every((key) => isReliable(target.points[key]));
    return leftOk || rightOk;
  }
  return metric.autoKeys.every((key) => isReliable(target.points[key]));
}

function renderResult(result) {
  const sign = Number(result.change) >= 0 ? "+" : "";
  els.resultBox.innerHTML =
    '<div class="result-main">変化：' + sign + result.change + " " + result.unit + "</div>" +
    "<div>評価：" + escapeHtml(result.theme) + "</div>" +
    "<div>Before：" + result.before + " " + result.unit + "（" + sourceLabel(result.beforeSource) + "）</div>" +
    "<div>After：" + result.after + " " + result.unit + "（" + sourceLabel(result.afterSource) + "）</div>" +
    "<div>数値は撮影距離・角度・衣服・タップ位置の影響を受けます。診断ではなく前後比較の目安です。</div>";
}

function saveLatest() {
  if (!requirePrivacyConfirmation()) return;
  const result = state.latestResult || compare(true);
  if (!result) return;
  const records = getRecords();
  records.push(result);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(records));
  renderHistory();
  showMessage("結果を保存しました。", "info");
}

function getRecords() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
  } catch {
    return [];
  }
}

function renderHistory() {
  const records = getRecords();
  if (!records.length) {
    els.historyBox.textContent = "保存記録はありません。";
    return;
  }
  const rows = records.slice().reverse().map((record) => (
    "<tr>" +
    "<td>" + escapeHtml(record.date) + "</td>" +
    "<td>" + escapeHtml(record.theme) + "</td>" +
    "<td>" + record.before + " " + record.unit + "</td>" +
    "<td>" + record.after + " " + record.unit + "</td>" +
    "<td>" + (Number(record.change) >= 0 ? "+" : "") + record.change + " " + record.unit + "</td>" +
    "<td>" + sourceLabel(record.beforeSource) + " / " + sourceLabel(record.afterSource) + "</td>" +
    "</tr>"
  )).join("");
  els.historyBox.innerHTML =
    "<table><thead><tr><th>日時</th><th>評価</th><th>Before</th><th>After</th><th>変化</th><th>入力</th></tr></thead><tbody>" +
    rows +
    "</tbody></table>";
}

function exportCsv() {
  if (!requirePrivacyConfirmation()) return;
  const records = getRecords();
  if (!records.length) {
    showMessage("CSV出力できる保存記録がありません。", "warn");
    return;
  }
  const header = ["日時", "評価", "Before", "After", "変化", "単位", "Before入力", "After入力"];
  const lines = records.map((record) => [
    record.date,
    record.theme,
    record.before,
    record.after,
    record.change,
    record.unit,
    sourceLabel(record.beforeSource),
    sourceLabel(record.afterSource)
  ]);
  const csv = "\uFEFF" + [header, ...lines].map((row) => row.map(csvCell).join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "posture-evaluation-records.csv";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function clearRecords() {
  if (!confirm("保存記録をすべて削除しますか？")) return;
  localStorage.removeItem(STORAGE_KEY);
  renderHistory();
}

function clearAllDeviceData() {
  if (!confirm("この端末内の保存記録、現在の写真、入力点、比較結果をすべて削除しますか？")) return;
  localStorage.removeItem(STORAGE_KEY);
  ["before", "after"].forEach((side) => {
    const target = state[side];
    target.image = null;
    target.imageName = "";
    target.points = {};
    target.manualPoints = [];
    target.pose = null;
    target.source = "none";
    target.canvas.width = 900;
    target.canvas.height = 680;
    drawPlaceholder(target);
  });
  state.latestResult = null;
  renderHistory();
  updateStatus();
  updateResultPrompt();
  showMessage("端末内データを削除しました。写真データは保存していません。", "info");
}

function requirePrivacyConfirmation() {
  if (els.privacyConfirm.checked) return true;
  showMessage("先にプライバシー欄を確認し、同意チェックを入れてください。", "warn");
  return false;
}

function updateGuide() {
  const metric = METRICS[els.themeSelect.value];
  els.guide.innerHTML =
    "<strong>手動タップ順：</strong>" + metric.manual.map((name, index) => (index + 1) + ". " + name).join(" / ") +
    "<br><strong>自動推定：</strong>MoveNetで必要点を検出し、信頼度が低い場合は手動入力に切り替えます。";
}

function updateStatus() {
  setSideStatus("before", sideStatusText("before"));
  setSideStatus("after", sideStatusText("after"));
}

function sideStatusText(side) {
  const target = state[side];
  const label = side === "before" ? "Before" : "After";
  if (!target.image) return "写真未選択";
  const metric = METRICS[els.themeSelect.value];
  if (hasRequiredPoints(target, metric)) return label + " 入力完了";
  if (target.source === "manual") return label + " 手動 " + target.manualPoints.length + "/" + metric.manual.length + "点";
  if (target.source === "auto") return label + " 自動点の信頼度不足";
  return label + " 写真読込済み";
}

function setSideStatus(side, text) {
  (side === "before" ? els.beforeStatus : els.afterStatus).textContent = text;
}

function updateResultPrompt() {
  if (state.latestResult) return;
  const metric = METRICS[els.themeSelect.value];
  els.resultBox.textContent = "Before / Afterの写真を入れ、自動推定または手動で " + metric.manual.length + "点ずつ入力してください。";
}

function setModelStatus(text, className) {
  els.modelStatus.textContent = text;
  els.modelStatus.className = "status-pill" + (className ? " " + className : "");
}

function showMessage(text, type) {
  const message = document.createElement("div");
  message.className = "message " + (type || "info");
  message.textContent = text;
  els.messageArea.prepend(message);
  while (els.messageArea.children.length > 3) els.messageArea.lastElementChild.remove();
}

function redraw(target) {
  if (!target.image) {
    drawPlaceholder(target);
    return;
  }
  const ctx = target.ctx;
  ctx.clearRect(0, 0, target.canvas.width, target.canvas.height);
  ctx.drawImage(target.image, 0, 0, target.canvas.width, target.canvas.height);
  drawSkeleton(target);
  drawMetricPoints(target);
}

function drawPlaceholder(target) {
  const ctx = target.ctx;
  ctx.clearRect(0, 0, target.canvas.width, target.canvas.height);
  ctx.fillStyle = "#eef2f3";
  ctx.fillRect(0, 0, target.canvas.width, target.canvas.height);
  ctx.fillStyle = "#60717c";
  ctx.font = "24px sans-serif";
  ctx.fillText(target.side === "before" ? "Before写真を入れてください" : "After写真を入れてください", 28, 54);
}

function drawSkeleton(target) {
  if (!target.pose || !target.pose.keypoints) return;
  const ctx = target.ctx;
  const pairs = [
    ["left_shoulder", "right_shoulder"],
    ["left_shoulder", "left_hip"],
    ["right_shoulder", "right_hip"],
    ["left_hip", "right_hip"],
    ["left_hip", "left_knee"],
    ["left_knee", "left_ankle"],
    ["right_hip", "right_knee"],
    ["right_knee", "right_ankle"]
  ];
  ctx.strokeStyle = "#f0b43f";
  ctx.lineWidth = 4;
  pairs.forEach(([a, b]) => {
    const p1 = target.points[a];
    const p2 = target.points[b];
    if (!p1 || !p2 || p1.score < MIN_SCORE || p2.score < MIN_SCORE) return;
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
  });
}

function drawMetricPoints(target) {
  const metric = METRICS[els.themeSelect.value];
  const ctx = target.ctx;
  const autoKeys = visibleAutoKeys(target, metric);
  const points = target.source === "manual" ? target.manualPoints : autoKeys.map((key) => target.points[key]).filter(Boolean);
  points.forEach((point, index) => {
    if (!point || point.score < MIN_SCORE) return;
    ctx.beginPath();
    ctx.fillStyle = target.side === "before" ? "#d1495b" : "#2f7d4f";
    ctx.arc(point.x, point.y, 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 13px sans-serif";
    ctx.fillText(String(index + 1), point.x - 4, point.y + 5);
    const label = target.source === "manual" ? metric.manual[index] : KEYPOINT_NAMES[autoKeys[index]];
    ctx.fillStyle = "#17202a";
    ctx.font = "16px sans-serif";
    ctx.fillText(label || "", point.x + 12, point.y - 10);
  });
}

function visibleAutoKeys(target, metric) {
  if (metric === METRICS.knee) {
    const leftOk = ["left_hip", "left_knee", "left_ankle"].every((key) => isReliable(target.points[key]));
    const rightKeys = ["right_hip", "right_knee", "right_ankle"];
    return leftOk ? ["left_hip", "left_knee", "left_ankle"] : rightKeys;
  }
  if (metric === METRICS.neck) {
    const headKey = isReliable(target.points.nose) ? "nose" : null;
    return [headKey, "left_ear", "right_ear", "left_shoulder", "right_shoulder"].filter(Boolean);
  }
  return metric.autoKeys;
}

function isReliable(point) {
  return Boolean(point && point.score >= MIN_SCORE);
}

function midpoint(a, b) {
  if (!a || !b || a.score < MIN_SCORE || b.score < MIN_SCORE) return null;
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, score: Math.min(a.score, b.score) };
}

function distance(a, b) {
  if (!a || !b || a.score < MIN_SCORE || b.score < MIN_SCORE) return 0;
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function angle(a, b, c) {
  if (!a || !b || !c) return NaN;
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };
  const dot = ab.x * cb.x + ab.y * cb.y;
  const mag = Math.hypot(ab.x, ab.y) * Math.hypot(cb.x, cb.y);
  if (!mag) return NaN;
  return Math.acos(Math.max(-1, Math.min(1, dot / mag))) * 180 / Math.PI;
}

function confidenceSum(points, keys) {
  return keys.reduce((sum, key) => sum + ((points[key] && points[key].score) || 0), 0);
}

function imageScale(points) {
  return (points.__scale && points.__scale.height) || 680;
}

function round(value) {
  return (Math.round(value * 10) / 10).toFixed(1);
}

function sourceLabel(source) {
  if (source === "auto") return "自動";
  if (source === "manual") return "手動";
  return "未入力";
}

function csvCell(value) {
  return '"' + String(value).replace(/"/g, '""') + '"';
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  }[char]));
}

boot();
