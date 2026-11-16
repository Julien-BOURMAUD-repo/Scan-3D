'use strict';
/* Detection de la planche ChArUco, calcul de pose et calibration (OpenCV.js). */

const OPENCV_URL = 'https://cdn.jsdelivr.net/npm/@techstark/opencv-js@5.0.0-release.1/dist/opencv.js';

let cv = null;
let planche = null;
let detecteur = null;
let cfg = null;
let calibObj = [];
let calibImg = [];
let derniereDetection = null; // derniere image analysee par pose(), gardee pour la calibration en direct

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
  derniereDetection = det;
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

/* ---------- Detourage par la planche ----------
   La pose donne, pour chaque pixel, le point de la feuille vu a travers lui. Le motif imprime
   etant connu, un pixel dont la couleur ne correspond pas a ce motif appartient a l'objet.
   Masque : 0 = fond (retire le volume), 1 = objet, 2 = indetermine (ne retire rien). */

const MOTIF_PXMM = 4;
const PAGE = { x0: -15, y0: -12, x1: 195, yTexte: 262 }; // feuille A4 autour de la planche (mm), texte exclu
let motif = null;

function preparerMotif() {
  const w = cfg.nx * cfg.carre * MOTIF_PXMM, h = cfg.ny * cfg.carre * MOTIF_PXMM;
  const img = new cv.Mat();
  planche.generateImage(new cv.Size(w, h), img, 0, 1);
  motif = { data: img.data.slice(), w, h };
  img.delete();
}

/* Classe attendue au point (x, y) de la planche, en mm : 0 noir, 1 blanc, -1 inconnu (hors feuille, texte). */
function classePlan(x, y) {
  const L = cfg.nx * cfg.carre, H = cfg.ny * cfg.carre;
  if (x >= 0 && y >= 0 && x < L && y < H) {
    return motif.data[((y * MOTIF_PXMM) | 0) * motif.w + ((x * MOTIF_PXMM) | 0)] < 128 ? 0 : 1;
  }
  if (x > PAGE.x0 + 2 && x < PAGE.x1 - 2 && y > PAGE.y0 + 2 && y < PAGE.yTexte) return 1;
  return -1;
}

function enveloppeConvexe(pts) {
  const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const croix = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const bas = [], haut = [];
  for (const q of p) { while (bas.length >= 2 && croix(bas[bas.length - 2], bas[bas.length - 1], q) <= 0) bas.pop(); bas.push(q); }
  for (let i = p.length - 1; i >= 0; i--) { const q = p[i]; while (haut.length >= 2 && croix(haut[haut.length - 2], haut[haut.length - 1], q) <= 0) haut.pop(); haut.push(q); }
  return bas.slice(0, -1).concat(haut.slice(0, -1));
}

/* Modele d'eclairage lisse (polynome de degre 2 en u, v) ajuste de facon robuste sur les pixels d'une classe. */
function ajusterNiveau(lum, cls, classe, mw, mh, garder) {
  const base = (u, v) => { const x = u / mw - 0.5, y = v / mh - 0.5; return [1, x, y, x * x, x * y, y * y]; };
  let coef = null;
  for (let it = 0; it < 4; it++) {
    const A = new Float64Array(36), b = new Float64Array(6);
    let n = 0;
    for (let v = 0; v < mh; v += 2) for (let u = 0; u < mw; u += 2) {
      const i = v * mw + u;
      if (cls[i] !== classe) continue;
      const f = base(u, v);
      if (coef && !garder(lum[i], f.reduce((s, x, k) => s + x * coef[k], 0))) continue;
      for (let r = 0; r < 6; r++) { b[r] += f[r] * lum[i]; for (let c = 0; c < 6; c++) A[r * 6 + c] += f[r] * f[c]; }
      n++;
    }
    if (n < 50) return null;
    for (let r = 0; r < 6; r++) A[r * 6 + r] += 1e-6 * n;
    coef = resoudre(A, b, 6);
  }
  return (u, v) => base(u, v).reduce((s, x, k) => s + x * coef[k], 0);
}

function resoudre(A, b, n) {
  const M = Array.from({ length: n }, (_, r) => [...A.slice(r * n, r * n + n), b[r]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < n; r++) {
      if (r === c || !M[c][c]) continue;
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((ligne, r) => (M[r][r] ? ligne[n] / M[r][r] : 0));
}

function teinteDe(r, g, b, c) {
  const mx = Math.max(r, g, b);
  const h = mx === r ? ((g - b) / c) % 6 : mx === g ? (b - r) / c + 2 : (r - g) / c + 4;
  return (h * 60 + 360) % 360;
}

/* Detourage sur planche couleur. Retourne null si la planche est en noir et blanc. */
function detourerCouleur(rgba, chroma, net, dedans, planX, planY, L, Hb, n) {
  const teinte = new Float32Array(n), vive = new Uint8Array(n), surPlanche = new Uint8Array(n);
  const histo = [new Float32Array(36), new Float32Array(36)];
  const total = [0, 0], vives = [0, 0];
  for (let i = 0; i < n; i++) {
    if (!(planX[i] >= 0 && planX[i] < L && planY[i] >= 0 && planY[i] < Hb)) continue;
    surPlanche[i] = 1;
    const r = rgba[4 * i], g = rgba[4 * i + 1], b = rgba[4 * i + 2];
    const mx = Math.max(r, g, b), c = chroma[i];
    if (mx >= 40 && c >= 0.3 * mx) { vive[i] = 1; teinte[i] = teinteDe(r, g, b, c); }
    const k = net[i];
    if (k < 0) continue;
    total[k]++;
    if (vive[i]) { vives[k]++; histo[k][Math.floor(teinte[i] / 10) % 36]++; }
  }
  if (total[0] < 200 || total[1] < 200 || vives[0] < 0.5 * total[0] || vives[1] < 0.5 * total[1]) return null;
  const ref = histo.map((h) => {
    let best = 0, bestV = -1;
    for (let k = 0; k < 36; k++) { const v = h[(k + 35) % 36] + 2 * h[k] + h[(k + 1) % 36]; if (v > bestV) { bestV = v; best = k; } }
    return best * 10 + 5;
  });
  const masque = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (!dedans[i]) continue;
    const k = net[i];
    if (k < 0 || !surPlanche[i]) { masque[i] = 2; continue; }
    const ecart = Math.abs(((teinte[i] - ref[k] + 540) % 360) - 180);
    masque[i] = vive[i] && ecart <= 30 ? 0 : 1;
  }
  return masque;
}

/* Fin commune du detourage : filtrage des detections fines, protection des trous, comptage. */
function finirMasque(masque, dedans, mw, mh, infos) {
  const n = mw * mh;
  // Detections de moins de 3 px d'epaisseur (bords de carres decales quand la feuille n'est pas
  // tout a fait plane) : indeterminees. Ouverture morphologique 3 x 3 : les vrais objets restent entiers.
  const huit = [-1, 1, -mw, mw, -mw - 1, -mw + 1, mw - 1, mw + 1];
  const erode = new Uint8Array(n);
  for (let v = 1; v < mh - 1; v++) for (let u = 1; u < mw - 1; u++) {
    const i = v * mw + u;
    if (masque[i] !== 1) continue;
    let plein = true;
    for (const d of huit) if (masque[i + d] !== 1) { plein = false; break; }
    if (plein) erode[i] = 1;
  }
  for (let v = 1; v < mh - 1; v++) for (let u = 1; u < mw - 1; u++) {
    const i = v * mw + u;
    if (masque[i] !== 1 || erode[i]) continue;
    let garde = false;
    for (const d of huit) if (erode[i + d]) { garde = true; break; }
    if (!garde) masque[i] = 2;
  }
  // Zones de fond enfermees dans l'objet (reflet, logo clair) : indeterminees, pour ne pas creuser l'objet.
  // Le fond reel est relie au bord de l'image sans traverser l'objet.
  const relie = new Uint8Array(n);
  const file = new Int32Array(n);
  let tete = 0, queue = 0;
  const graine = (i) => { if (!relie[i] && masque[i] !== 1) { relie[i] = 1; file[queue++] = i; } };
  for (let i = 0; i < n; i++) if (!dedans[i]) graine(i);
  for (let u = 0; u < mw; u++) { graine(u); graine((mh - 1) * mw + u); }
  for (let v = 0; v < mh; v++) { graine(v * mw); graine(v * mw + mw - 1); }
  while (tete < queue) {
    const i = file[tete++], u = i % mw;
    for (const j of [u > 0 ? i - 1 : -1, u < mw - 1 ? i + 1 : -1, i - mw, i + mw]) {
      if (j >= 0 && j < n && !relie[j] && masque[j] !== 1) { relie[j] = 1; file[queue++] = j; }
    }
  }
  let objet = 0;
  for (let i = 0; i < n; i++) {
    if (masque[i] === 0 && !relie[i]) masque[i] = 2;
    if (masque[i] === 1) objet++;
  }
  return { masque, objet, ...infos };
}

function detourer(m) {
  if (!motif) preparerMotif();
  const { rgba, mw, mh, R, t, K, dist, emprise, hauteur, fondColore } = m;
  const n = mw * mh;
  const L = cfg.nx * cfg.carre, Hb = cfg.ny * cfg.carre;
  const C = [
    -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]),
    -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]),
    -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]),
  ];
  const sgn = C[2] > 0 ? 1 : -1;

  // Projection du volume de travail : l'objet ne peut etre qu'a l'interieur
  const e = emprise / 2;
  const coins = [];
  let derriere = false;
  for (const X of [L / 2 - e, L / 2 + e]) for (const Y of [Hb / 2 - e, Hb / 2 + e]) for (const Z of [0, sgn * hauteur]) {
    const zc = R[6] * X + R[7] * Y + R[8] * Z + t[2];
    if (zc <= 1) derriere = true;
    coins.push(projeter(R, t, K, dist, X, Y, Z));
  }
  let env = derriere ? null : enveloppeConvexe(coins);
  if (env) {
    const gx = env.reduce((s, p) => s + p[0], 0) / env.length, gy = env.reduce((s, p) => s + p[1], 0) / env.length;
    env = env.map(([x, y]) => { const d = Math.hypot(x - gx, y - gy) || 1; return [x + ((x - gx) / d) * 4, y + ((y - gy) / d) * 4]; });
  }
  const dansEnv = (u, v) => {
    if (!env) return true;
    for (let i = 0; i < env.length; i++) {
      const a = env[i], b = env[(i + 1) % env.length];
      if ((b[0] - a[0]) * (v - a[1]) - (b[1] - a[1]) * (u - a[0]) < 0) return false;
    }
    return true;
  };

  // Classe attendue de chaque pixel (rayon intersecte avec le plan de la feuille)
  const fx = K[0], sk = K[1], cx = K[2], fy = K[4], cy = K[5];
  const [k1, k2, p1, p2, k3] = [dist[0] || 0, dist[1] || 0, dist[2] || 0, dist[3] || 0, dist[4] || 0];
  const cls = new Int8Array(n);
  const planX = new Float32Array(n), planY = new Float32Array(n);
  const dedans = new Uint8Array(n);
  const lum = new Float32Array(n), chroma = new Float32Array(n);
  for (let v = 0; v < mh; v++) for (let u = 0; u < mw; u++) {
    const i = v * mw + u;
    const r = rgba[4 * i], g = rgba[4 * i + 1], b = rgba[4 * i + 2];
    lum[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    chroma[i] = Math.max(r, g, b) - Math.min(r, g, b);
    dedans[i] = dansEnv(u, v) ? 1 : 0;
    const yd = (v - cy) / fy, xd = (u - cx - sk * yd) / fx;
    let x = xd, y = yd;
    for (let it = 0; it < 5; it++) {
      const r2 = x * x + y * y;
      const rad = 1 + r2 * (k1 + r2 * (k2 + r2 * k3));
      const dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x), dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;
      x = (xd - dx) / rad; y = (yd - dy) / rad;
    }
    const Dx = R[0] * x + R[3] * y + R[6], Dy = R[1] * x + R[4] * y + R[7], Dz = R[2] * x + R[5] * y + R[8];
    const s = -C[2] / Dz;
    cls[i] = s > 0 ? classePlan(C[0] + s * Dx, C[1] + s * Dy) : -1;
    planX[i] = s > 0 ? C[0] + s * Dx : NaN; planY[i] = s > 0 ? C[1] + s * Dy : NaN;
  }
  // Pixels a cheval sur une transition noir / blanc : indetermines (flou, imprecision de pose)
  const net = new Int8Array(cls);

  // Fond colore : une feuille de couleur vive posee au centre de la planche, sous l'objet.
  // Sur cette feuille, tout ce qui n'a pas sa couleur est l'objet (blanc, gris ou noir compris).
  let papier = null, memeTeinte = null;
  if (fondColore) {
    const teinte = new Float32Array(n), vive = new Uint8Array(n);
    const histo = new Float32Array(36);
    for (let i = 0; i < n; i++) {
      const r = rgba[4 * i], g = rgba[4 * i + 1], b = rgba[4 * i + 2];
      const mx = Math.max(r, g, b), c = chroma[i];
      if (mx < 50 || c < 0.3 * mx) continue;
      let h = mx === r ? ((g - b) / c) % 6 : mx === g ? (b - r) / c + 2 : (r - g) / c + 4;
      h = (h * 60 + 360) % 360;
      teinte[i] = h; vive[i] = 1;
      if (planX[i] >= 0 && planX[i] < L && planY[i] >= 0 && planY[i] < Hb) histo[Math.floor(h / 10) % 36]++;
    }
    let best = 0, bestV = -1;
    for (let k = 0; k < 36; k++) { const v = histo[(k + 35) % 36] + 2 * histo[k] + histo[(k + 1) % 36]; if (v > bestV) { bestV = v; best = k; } }
    const h0 = best * 10 + 5;
    memeTeinte = new Uint8Array(n);
    const xs = [], ys = [];
    for (let i = 0; i < n; i++) {
      if (!vive[i]) continue;
      const dh = Math.abs(((teinte[i] - h0 + 540) % 360) - 180);
      if (dh > 25) continue;
      memeTeinte[i] = 1;
      if (planX[i] >= 0 && planX[i] < L && planY[i] >= 0 && planY[i] < Hb) { xs.push(planX[i]); ys.push(planY[i]); }
    }
    if (xs.length < n * 0.005) throw new Error('Fond coloré non trouvé');
    xs.sort((a, b) => a - b); ys.sort((a, b) => a - b);
    const q = (a, f) => a[Math.min(a.length - 1, Math.floor(a.length * f))];
    const bx0 = q(xs, 0.01) + 3, bx1 = q(xs, 0.99) - 3, by0 = q(ys, 0.01) + 3, by1 = q(ys, 0.99) - 3;
    papier = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      if (planX[i] > bx0 && planX[i] < bx1 && planY[i] > by0 && planY[i] < by1) { papier[i] = 1; net[i] = -1; }
    }
  }
  for (let v = 1; v < mh - 1; v++) for (let u = 1; u < mw - 1; u++) {
    const i = v * mw + u, c = cls[i];
    if (c < 0) continue;
    for (const d of [-1, 1, -mw, mw, -mw - 1, -mw + 1, mw - 1, mw + 1]) if (cls[i + d] !== c) { net[i] = -1; break; }
  }

  // Planche couleur (bleu fonce / jaune clair) : reconnue toute seule quand les deux classes de carres
  // sont vivement colorees. Un pixel qui n'a pas la teinte attendue est l'objet, meme blanc, gris ou noir.
  const couleur = detourerCouleur(rgba, chroma, net, dedans, planX, planY, L, Hb, n);
  if (couleur) return finirMasque(couleur, dedans, mw, mh, { sombre: false, colore: false, planche: 'couleur' });

  // Niveaux du blanc et du noir de la feuille, selon l'eclairage
  const blanc = ajusterNiveau(lum, net, 1, mw, mh, (l, f) => l > f * 0.85);
  const noir = ajusterNiveau(lum, net, 0, mw, mh, (l, f) => l < f + 25);
  if (!blanc) throw new Error('Feuille peu visible');
  const B0 = noir ? null : 30;

  const masque = new Uint8Array(n);
  const blancPix = new Float32Array(n);
  const sombres = [];
  const teintes = []; // saturation relative des pixels d'objet vus sur du blanc
  for (let v = 0; v < mh; v++) for (let u = 0; u < mw; u++) {
    const i = v * mw + u;
    if (!dedans[i]) continue;
    if (memeTeinte && memeTeinte[i]) continue;          // couleur du fond : fond
    if (papier && papier[i]) { masque[i] = 4; continue; } // sur la feuille coloree, autre couleur : objet
    const W = blanc(u, v), Bk = noir ? noir(u, v) : B0;
    blancPix[i] = W;
    const ecart = Math.max(20, W - Bk);
    const colore = chroma[i] > 0.25 * W;
    const c = net[i];
    if (c === 1) {
      if (colore || lum[i] < W - 0.35 * ecart) { masque[i] = 1; sombres.push((lum[i] - Bk) / ecart); teintes.push(chroma[i] / W); }
    } else if (c === 0) {
      // Sur le noir imprime : seuil fin, zone ambigue decidee selon la teinte de l'objet
      if (colore || lum[i] > Bk + 0.15 * ecart) masque[i] = 1;
      else if (lum[i] < Bk + 0.07 * ecart) masque[i] = 0;
      else masque[i] = 3;
    } else {
      masque[i] = colore ? 1 : 2;
    }
  }
  // Objet tres sombre : il se confond avec le noir imprime, la zone ambigue reste alors indeterminee
  sombres.sort((a, b) => a - b);
  const objetSombre = sombres.length > 20 && sombres[sombres.length >> 1] < 0.2;
  // Objet nettement colore : un ecart sans couleur (ombre, bord de carre decale) est du fond.
  // Le seuil suit la saturation de l'objet, pour ignorer aussi le reflet colore de l'objet sur le papier.
  // (les faux positifs, eux, sont incolores : on juge sur la part de pixels colores)
  const colores = teintes.filter((x) => x > 0.25).sort((a, b) => a - b);
  const objetColore = colores.length > Math.max(300, n * 0.002) && colores.length > 0.4 * teintes.length;
  if (objetColore) {
    const seuil = Math.max(0.25, 0.4 * colores[colores.length >> 1]);
    for (let i = 0; i < n; i++) {
      if (!dedans[i] || masque[i] === 0 || masque[i] === 4) continue;
      if (chroma[i] > seuil * blancPix[i]) masque[i] = 1;
      else masque[i] = net[i] === -1 ? 2 : 0;
    }
  } else {
    for (let i = 0; i < n; i++) if (masque[i] === 3) masque[i] = objetSombre ? 2 : 0;
  }

  for (let i = 0; i < n; i++) if (masque[i] === 4) masque[i] = 1;
  return finirMasque(masque, dedans, mw, mh, { sombre: objetSombre, colore: objetColore, planche: 'noir et blanc' });
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
    } else if (m.type === 'detourer') {
      r = detourer(m);
      self.postMessage({ id: m.id, ...r }, [r.masque.buffer]);
      return;
    } else if (m.type === 'calib-garder') {
      const det = derniereDetection;
      const garde = !!det && det.n >= 12;
      if (garde) { calibObj.push(det.obj); calibImg.push(det.pts); }
      r = { garde, total: calibObj.length };
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
