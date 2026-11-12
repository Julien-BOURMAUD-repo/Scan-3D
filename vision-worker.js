'use strict';
/* Detection de la planche ChArUco, calcul de pose et calibration (OpenCV.js). */

const OPENCV_URL = 'https://cdn.jsdelivr.net/npm/@techstark/opencv-js@5.0.0-release.1/dist/opencv.js';

let cv = null;
let planche = null;
let detecteur = null;
let cfg = null;
let calibObj = [];
let calibImg = [];

async function chargerOpenCV() {
  if (cv) return;
  importScripts(OPENCV_URL);
  let c = self.cv;
  if (c && typeof c.then === 'function') c = await c;
  if (!c.Mat) await new Promise((r) => { c.onRuntimeInitialized = r; });
  cv = c;
}

function configurer(p) {
  cfg = p;
  const dico = cv.getPredefinedDictionary(cv.DICT_5X5_250);
  const ids = new cv.Mat();
  planche = new cv.aruco_CharucoBoard(new cv.Size(p.nx, p.ny), p.carre, p.marqueur, dico, ids);
  const dp = new cv.aruco_DetectorParameters();
  try { dp.cornerRefinementMethod = cv.CORNER_REFINE_SUBPIX; } catch (e) { /* option facultative */ }
  detecteur = new cv.aruco_CharucoDetector(planche, new cv.aruco_CharucoParameters(), dp,
    new cv.aruco_RefineParameters(10, 3, true));
}

/* Retourne les coins detectes : points 3D sur la planche (mm) et points 2D image (px). */
function detecter(gris, w, h) {
  const img = new cv.Mat(h, w, cv.CV_8UC1);
  img.data.set(gris);
  const cc = new cv.Mat(), ci = new cv.Mat(), mc = new cv.MatVector(), mi = new cv.Mat();
  try {
    detecteur.detectBoard(img, cc, ci, mc, mi);
    const n = ci.rows * ci.cols;
    const obj = new Float64Array(n * 3);
    const pts = new Float64Array(n * 2);
    const par = cfg.nx - 1;
    for (let k = 0; k < n; k++) {
      const id = ci.data32S[k];
      obj[3 * k] = ((id % par) + 1) * cfg.carre;
      obj[3 * k + 1] = (Math.floor(id / par) + 1) * cfg.carre;
      pts[2 * k] = cc.data32F[2 * k];
      pts[2 * k + 1] = cc.data32F[2 * k + 1];
    }
    return { n, obj, pts };
  } finally {
    img.delete(); cc.delete(); ci.delete(); mc.delete(); mi.delete();
  }
}

function projeter(R, t, K, d, X, Y, Z) {
  const x = R[0] * X + R[1] * Y + R[2] * Z + t[0];
  const y = R[3] * X + R[4] * Y + R[5] * Z + t[1];
  const z = R[6] * X + R[7] * Y + R[8] * Z + t[2];
  const xn = x / z, yn = y / z;
  const r2 = xn * xn + yn * yn;
  const rad = 1 + r2 * (d[0] + r2 * (d[1] + r2 * (d[4] || 0)));
  const xd = xn * rad + 2 * d[2] * xn * yn + d[3] * (r2 + 2 * xn * xn);
  const yd = yn * rad + d[2] * (r2 + 2 * yn * yn) + 2 * d[3] * xn * yn;
  return [K[0] * xd + K[1] * yd + K[2], K[4] * yd + K[5]];
}

function pose(m) {
  const det = detecter(m.gris, m.w, m.h);
  if (det.n < 8) return { n: det.n };
  const objM = cv.matFromArray(det.n, 3, cv.CV_64F, Array.from(det.obj));
  const imgM = cv.matFromArray(det.n, 2, cv.CV_64F, Array.from(det.pts));
  const K = cv.matFromArray(3, 3, cv.CV_64F, m.K);
  const D = cv.matFromArray(1, 5, cv.CV_64F, m.dist.slice(0, 5));
  const rv = new cv.Mat(), tv = new cv.Mat(), Rm = new cv.Mat();
  try {
    if (!cv.solvePnP(objM, imgM, K, D, rv, tv, false, 0)) return { n: det.n };
    cv.Rodrigues(rv, Rm);
    const R = Array.from(Rm.data64F);
    const t = Array.from(tv.data64F);
    let s = 0;
    for (let k = 0; k < det.n; k++) {
      const [u, v] = projeter(R, t, m.K, m.dist, det.obj[3 * k], det.obj[3 * k + 1], 0);
      s += (u - det.pts[2 * k]) ** 2 + (v - det.pts[2 * k + 1]) ** 2;
    }
    return { n: det.n, R, t, err: Math.sqrt(s / det.n) };
  } finally {
    [objM, imgM, K, D, rv, tv, Rm].forEach((x) => x.delete());
  }
}

function calibrer(w, h) {
  const objV = new cv.MatVector(), imgV = new cv.MatVector();
  const mats = [];
  for (let i = 0; i < calibObj.length; i++) {
    const n = calibObj[i].length / 3;
    const o = cv.matFromArray(n, 3, cv.CV_32F, Array.from(calibObj[i]));
    const p = cv.matFromArray(n, 2, cv.CV_32F, Array.from(calibImg[i]));
    objV.push_back(o); imgV.push_back(p); mats.push(o, p);
  }
  const K = new cv.Mat(), D = new cv.Mat(), rv = new cv.MatVector(), tv = new cv.MatVector();
  const s1 = new cv.Mat(), s2 = new cv.Mat(), pe = new cv.Mat();
  try {
    const crit = new cv.TermCriteria(cv.TermCriteria_COUNT + cv.TermCriteria_EPS, 50, 1e-7);
    const rms = cv.calibrateCameraExtended(objV, imgV, new cv.Size(w, h), K, D, rv, tv, s1, s2, pe, 0, crit);
    const dist = Array.from(D.data64F).slice(0, 5);
    while (dist.length < 5) dist.push(0);
    return { rms, K: Array.from(K.data64F), dist, parVue: Array.from(pe.data64F) };
  } finally {
    [objV, imgV, K, D, rv, tv, s1, s2, pe, ...mats].forEach((x) => x.delete());
  }
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    let r;
    if (m.type === 'init') {
      await chargerOpenCV();
      configurer(m.planche);
      r = {};
    } else if (m.type === 'pose') {
      r = pose(m);
    } else if (m.type === 'calib-reset') {
      calibObj = []; calibImg = [];
      r = {};
    } else if (m.type === 'calib-ajouter') {
      const det = detecter(m.gris, m.w, m.h);
      if (det.n >= 12) { calibObj.push(det.obj); calibImg.push(det.pts); }
      r = { n: det.n };
    } else if (m.type === 'calibrer') {
      r = calibrer(m.w, m.h);
    } else {
      throw new Error('Message inconnu : ' + m.type);
    }
    self.postMessage({ id: m.id, ...r });
  } catch (err) {
    self.postMessage({ id: m.id, erreur: err && err.message ? err.message : String(err) });
  }
};
