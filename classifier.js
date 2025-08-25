<script>
// =========================
//  參數與類別
// =========================
const CLASS_14 = ['將','士','象','車','馬','包','卒','帥','仕','相','俥','傌','炮','兵'];
let mobileModel = null; // MobileNet
const FEATURE_LAYER = 'global_average'; // 輕量/穩定

// 以「最近質心」做小分類器：label -> {centroid:Tensor1D, count:int}
const protoStore = new Map();

// 你手上的 5 個 ROI canvas（外部丟進來）
let currentRois = []; // [Canvas, Canvas, ...] 共 5 個

// =========================
//  載入 MobileNet（一次）
// =========================
async function loadMobileNetOnce() {
  if (mobileModel) return mobileModel;
  mobileModel = await mobilenet.load({ version: 2, alpha: 0.5 });
  console.log('MobileNet loaded');
  return mobileModel;
}

// =========================
//  顏色：紅/黑 判定（ROI -> boolean）
//  小技巧：統計紅色像素比例（H 在 0–10 或 170–179）
// =========================
function isRedPieceCanvas(canvas) {
  // 用原生 API 做 HSV 估計（快速版）；要更穩可改用 OpenCV
  const ctx = canvas.getContext('2d');
  const { width:w, height:h } = canvas;
  const img = ctx.getImageData(0,0,w,h).data;
  let redCount = 0, total = 0;
  for (let i=0;i<img.length;i+=4){
    const r=img[i], g=img[i+1], b=img[i+2], a=img[i+3];
    if (a < 10) continue;
    // 粗轉 HSV：這裡用近似演算法；為穩定可換 OpenCV HSV
    const mx = Math.max(r,g,b), mn = Math.min(r,g,b);
    const c = mx - mn; const v = mx/255;
    let hdeg = 0;
    if (c === 0) hdeg = 0;
    else if (mx === r) hdeg = ((g-b)/c)%6 * 60;
    else if (mx === g) hdeg = ((b-r)/c + 2) * 60;
    else hdeg = ((r-g)/c + 4) * 60;
    if (hdeg < 0) hdeg += 360;
    // 紅色區域粗判
    const isRedHue = (hdeg <= 12) || (hdeg >= 348);
    if (isRedHue && v > 0.2) redCount++;
    total++;
  }
  const ratio = redCount / Math.max(1,total);
  return ratio > 0.04; // 視現場微調
}

// =========================
//  （可選）OpenCV 前處理：產生「去邊的字形」debug 圖
//  實際丟 TF.js 我們用原圖（224×224，歸一化到 [-1,1]）
//  這個 glyph 可用來 debug 或做模板法備援
// =========================
function opencvGlyph(canvasIn) {
  const src = cv.imread(canvasIn);               // RGBA
  const gray = new cv.Mat(); cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
  const bin = new cv.Mat();
  cv.adaptiveThreshold(gray, bin, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                       cv.THRESH_BINARY_INV, 25, 10);
  const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3,3));
  cv.morphologyEx(bin, bin, cv.MORPH_CLOSE, kernel);

  // 內圈遮罩（避免外框干擾）
  const mask = new cv.Mat.zeros(bin.rows, bin.cols, cv.CV_8UC1);
  const center = new cv.Point(bin.cols/2, bin.rows/2);
  const radius = Math.min(bin.cols, bin.rows) * 0.42;
  cv.circle(mask, center, radius, new cv.Scalar(255), -1);
  const glyph = new cv.Mat(); cv.bitwise_and(bin, mask, glyph);

  const out = document.createElement('canvas');
  cv.imshow(out, glyph);

  // 釋放
  [src, gray, bin, kernel, mask, glyph].forEach(m => m.delete());
  return out;
}

// =========================
//  影像 -> Tensor（TF.js，給 MobileNet 用）
//  - 將 ROI 畫到 224×224
//  - 正規化到 [-1,1]
// =========================
function roiToTensor(canvasIn) {
  const S = 224;
  const tmp = document.createElement('canvas');
  tmp.width = tmp.height = S;
  const tctx = tmp.getContext('2d');
  tctx.drawImage(canvasIn, 0,0,S,S);

  return tf.tidy(() => {
    const t = tf.browser.fromPixels(tmp).toFloat();
    // [-1,1]
    return t.div(127.5).sub(1).expandDims(0); // [1,224,224,3]
  });
}

// =========================
//  取特徵向量（MobileNet global_average）
// =========================
async function embed(canvasIn) {
  await loadMobileNetOnce();
  const input = roiToTensor(canvasIn);
  const feat = mobileModel.infer(input, FEATURE_LAYER); // [1,depth]
  input.dispose();
  return feat.squeeze(); // [depth]
}

// =========================
//  新增樣本到「最近質心」分類器
//  多次加入同一類，會自動更新質心
// =========================
async function addPrototypeSample(label, roiCanvas) {
  const f = await embed(roiCanvas); // Tensor1D
  let rec = protoStore.get(label);
  if (!rec) rec = { sum: null, count: 0, centroid: null };
  // sum = sum + f
  if (!rec.sum) rec.sum = f.clone();
  else {
    const old = rec.sum; rec.sum = tf.tidy(()=> tf.add(old, f)); old.dispose();
  }
  rec.count += 1;
  // centroid = sum / count
  if (rec.centroid) rec.centroid.dispose();
  rec.centroid = tf.tidy(()=> tf.div(rec.sum, rec.count));
  protoStore.set(label, rec);
  f.dispose();
  console.log(`[Prototype] ${label} count=${rec.count}`);
}

// =========================
//  分類：與所有質心做 cosine 相似度，回傳 Top-1
// =========================
async function classifyRoi(roiCanvas) {
  if (protoStore.size === 0) throw new Error('尚未建立任何樣本/質心');
  const f = await embed(roiCanvas); // [d]
  let best = { label: null, score: -1 };
  // normalize f
  const fN = tf.tidy(()=> {
    const n = tf.norm(f);
    return tf.div(f, n.add(1e-8));
  });
  for (const [label, rec] of protoStore) {
    const cN = tf.tidy(()=> {
      const n = tf.norm(rec.centroid);
      return tf.div(rec.centroid, n.add(1e-8));
    });
    const sim = await tf.tidy(()=> tf.sum(tf.mul(fN, cN))).data();
    if (sim[0] > best.score) best = { label, score: sim[0] };
    cN.dispose();
  }
  f.dispose(); fN.dispose();
  return best; // {label:'車'(功能名或最終名), score: 0.x}
}

// =========================
//  輔助：把「功能名 + 顏色」轉成最終字
//  方案1：直接用 14 類（建議）；
//  方案2：若你先用 7 類功能（車/馬/包/卒 + 四尊），就用 isRed 轉字
// =========================
function toFinalLabel(class14 /* '車'...'兵' */, isRed) {
  // 若你已使用 14 類分類，直接回傳 class14 即可
  return class14;

  // 如果你先分類成「功能名」再轉，可改用下列邏輯：
  // const map = { '車': ['車','俥'], '馬':['馬','傌'], '包':['包','炮'], '卒':['卒','兵'] };
  // if (map[class14]) return isRed ? map[class14][1] : map[class14][0];
  // return class14;
}

// =========================
//  一次分類 5 個 ROI（回傳陣列與信心）
// =========================
async function classifyFiveRois(roiCanvases) {
  const results = [];
  for (let i=0;i<roiCanvases.length;i++){
    const roi = roiCanvases[i];
    const red = isRedPieceCanvas(roi);
    const pred = await classifyRoi(roi); // {label, score}
    const final = toFinalLabel(pred.label, red);
    results.push({ slot:i+1, label: final, score: Number(pred.score.toFixed(3)), isRed: red });
  }
  return results;
}

// ======== 你可以用的「上層觸發」範例 ========
// 假設你頁面裡：
// 1) 有 5 個 ROI canvas（已經擷取好了），放到 currentRois
// 2) 每一格旁邊有一個 <select> 讓你選 14 類，點「加入樣本」時呼叫下列函式

async function addAllFiveAsSamplesUsingSelections() {
  // 例如你在 ROI 清單每卡片有 <select data-slot="1">...
  const sels = document.querySelectorAll('.roi-card select');
  for (const sel of sels){
    const slot = Number(sel.dataset.slot);
    const label = sel.value; // 14 類之一
    await addPrototypeSample(label, currentRois[slot-1]);
  }
  alert('已加入 5 筆樣本/更新質心');
}

async function runClassifyCurrentRois() {
  const res = await classifyFiveRois(currentRois);
  console.table(res);
  // 你也可以把結果回填到 UI
}

// （可選）顯示 OpenCV 二值化 Glyph 做 debug
function showGlyphDebug(idx /* 1..5 */) {
  const g = opencvGlyph(currentRois[idx-1]);
  document.body.append(g); // 或放到你指定的 debug 區域
}

// （可選）匯出/載入質心（讓你保存目前小模型）
function exportCentroidsJson() {
  const out = {};
  for (const [label, rec] of protoStore) {
    out[label] = Array.from(rec.centroid.dataSync());
  }
  const blob = new Blob([JSON.stringify(out)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'xq_centroids.json';
  a.click();
}
async function importCentroidsJson(file) {
  const text = await file.text();
  const obj = JSON.parse(text);
  protoStore.clear();
  for (const label of Object.keys(obj)) {
    const v = tf.tensor1d(obj[label]);
    protoStore.set(label, { sum: v.clone(), count: 1, centroid: v });
  }
  alert('已載入質心');
}
</script>
