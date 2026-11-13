/* Scan 3D : interface, enchainement des etapes, ajustement des dimensions et export STL. */

const PLANCHE = { nx: 9, ny: 13, carre: 20, marqueur: 15 };
const TAILLE_TRAVAIL = 1920;
const TAILLE_MASQUE = 800;
const TAILLE_SEG = 1024;
const MIN_VUES = 8;
const ERREUR_POSE_MAX = 2.0;
const MODELE_DEFAUT = 'https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model_quantized.onnx';
const CLE_CALIB = 'scan3d.calibration';
const CLE_REGLAGES = 'scan3d.reglages';
const AXES = ['x', 'y', 'z'];

const $ = (id) => document.getElementById(id);

function lire(cle) {
  try { const v = localStorage.getItem(cle); return v ? JSON.parse(v) : null; } catch (e) { return null; }
}
function ecrire(cle, val) {
  try { if (val === null) localStorage.removeItem(cle); else localStorage.setItem(cle, JSON.stringify(val)); } catch (e) { /* stockage indisponible */ }
}

const etat = {
  calib: lire(CLE_CALIB),
  vues: [],
  modele: null,
  occupe: false,
  segPret: false,
  visionPret: false,
};

/* ---------- Workers ---------- */

function creerWorker(url) {
  const w = new Worker(url);
  let n = 0;
  const attente = new Map();
  const api = { surProgres: null };
  w.onmessage = (e) => {
    const m = e.data;
    if (m.type === 'progres') { if (api.surProgres) api.surProgres(m); return; }
    const p = attente.get(m.id);
    if (!p) return;
    attente.delete(m.id);
    if (m.erreur) p.reject(new Error(m.erreur)); else p.resolve(m);
  };
  w.onerror = (e) => {
    const err = new Error(e.message || 'Erreur interne du calcul');
    for (const p of attente.values()) p.reject(err);
    attente.clear();
  };
  api.appel = (type, data = {}, transfert = []) => {
    const id = ++n;
    return new Promise((resolve, reject) => {
      attente.set(id, { resolve, reject });
      w.postMessage({ id, type, ...data }, transfert);
    });
  };
  return api;
}

const vision = creerWorker('vision-worker.js');
const seg = creerWorker('seg-worker.js');
const recon = creerWorker('recon-worker.js');

async function initVision() {
  if (etat.visionPret) return;
  message('Chargement d\'OpenCV (environ 13 Mo)...');
  await vision.appel('init', { planche: PLANCHE });
  etat.visionPret = true;
}

async function initSeg() {
  if (etat.segPret) return;
  const barre = $('photos-progres');
  seg.surProgres = (m) => {
    if (m.etape === 'modele') {
      barre.hidden = false;
      barre.value = m.valeur;
      message(`Modèle de détourage : ${m.texte} (téléchargement unique, ensuite gardé en cache)`);
    }
  };
  await seg.appel('init', { url: reglages().modele || MODELE_DEFAUT });
  etat.segPret = true;
}

/* ---------- Reglages ---------- */

const CHAMPS = { barre: 100, voxel: 0.5, emprise: 100, hauteur: 100, tolerance: 1, lissage: 10, modele: '' };

function reglages() {
  const r = {};
  for (const [k, def] of Object.entries(CHAMPS)) {
    const el = $('r-' + k);
    if (typeof def === 'number') {
      const v = parseFloat(String(el.value).replace(',', '.'));
      r[k] = Number.isFinite(v) ? v : def;
    } else {
      r[k] = el.value.trim();
    }
  }
  r.tolerance = Math.max(0, Math.round(r.tolerance));
  r.lissage = Math.max(0, Math.round(r.lissage));
  r.voxel = Math.min(5, Math.max(0.2, r.voxel));
  return r;
}

function chargerReglages() {
  const s = lire(CLE_REGLAGES) || {};
  for (const [k, def] of Object.entries(CHAMPS)) {
    $('r-' + k).value = s[k] !== undefined ? s[k] : def;
    $('r-' + k).addEventListener('change', () => {
      ecrire(CLE_REGLAGES, reglages());
      if (k === 'modele') etat.segPret = false;
    });
  }
}

function facteurPlanche() {
  const b = reglages().barre;
  return b > 50 && b < 150 ? b / 100 : 1;
}

/* ---------- Messages ---------- */

function message(texte, type = '') {
  const el = $('message');
  el.textContent = texte;
  el.dataset.type = type;
}

function occuper(oui) {
  etat.occupe = oui;
  document.querySelectorAll('[data-bloquable]').forEach((el) => {
    if ('disabled' in el) el.disabled = oui;
    el.classList.toggle('desactive', oui);
  });
  if (!oui) majBoutons();
}

function fmt(v, d = 2) {
  return Number(v).toFixed(d).replace('.', ',');
}

/* ---------- Images ---------- */

async function chargerImage(fichier) {
  const bmp = await createImageBitmap(fichier);
  const s = Math.min(1, TAILLE_TRAVAIL / Math.max(bmp.width, bmp.height));
  return { bmp, W: Math.round(bmp.width * s), H: Math.round(bmp.height * s) };
}

function pixels(bmp, w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bmp, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h).data;
}

function versGris(rgba) {
  const n = rgba.length / 4;
  const g = new Uint8Array(n);
  for (let i = 0; i < n; i++) g[i] = (rgba[4 * i] * 77 + rgba[4 * i + 1] * 150 + rgba[4 * i + 2] * 29) >> 8;
  return g;
}

/* ---------- Calibration ---------- */

function majCalib() {
  const c = etat.calib;
  const el = $('calib-etat');
  if (c) {
    const src = c.source === 'direct' ? 'en direct' : c.source === 'photos' ? 'avec des photos' : 'importé';
    el.textContent = `Calibré ${src} : ${c.images} images, erreur ${fmt(c.rms, 3)} px, ${c.largeur} x ${c.hauteur}.`;
    el.dataset.type = c.rms <= 1 ? 'ok' : 'alerte';
  } else {
    el.textContent = 'Pas encore calibré.';
    el.dataset.type = 'alerte';
  }
  $('calib-exporter').disabled = !c;
  $('calib-effacer').disabled = !c;
}

async function calibrer(fichiers) {
  occuper(true);
  const barre = $('calib-progres');
  barre.hidden = false; barre.value = 0;
  try {
    await initVision();
    await vision.appel('calib-reset');
    let taille = null, ok = 0;
    const rejets = [];
    for (let i = 0; i < fichiers.length; i++) {
      const f = fichiers[i];
      message(`Calibration : photo ${i + 1} / ${fichiers.length}`);
      barre.value = i / fichiers.length;
      const { bmp, W, H } = await chargerImage(f);
      if (!taille) taille = [W, H];
      if (W !== taille[0] || H !== taille[1]) {
        rejets.push(`${f.name} (orientation ou taille différente)`);
        bmp.close();
        continue;
      }
      const gris = versGris(pixels(bmp, W, H));
      bmp.close();
      const r = await vision.appel('calib-ajouter', { gris, w: W, h: H }, [gris.buffer]);
      if (r.n >= 12) ok++; else rejets.push(`${f.name} (planche non détectée)`);
    }
    await calculerCalibration(taille, ok, rejets, 'photos');
  } catch (err) {
    message('Calibration : ' + err.message, 'erreur');
  } finally {
    barre.hidden = true;
    majCalib();
    occuper(false);
  }
}

/* Calcul final, commun aux photos et a la prise en direct (images deja ajoutees au worker). */
async function calculerCalibration(taille, ok, rejets, source) {
  if (ok < 8) throw new Error(`Seulement ${ok} images exploitables (8 minimum, 15 à 25 conseillées).`);
  message('Calcul de la calibration...');
  const r = await vision.appel('calibrer', { w: taille[0], h: taille[1] });
  etat.calib = {
    largeur: taille[0], hauteur: taille[1], K: r.K, dist: r.dist, rms: r.rms,
    images: ok, source, date: new Date().toISOString(),
  };
  ecrire(CLE_CALIB, etat.calib);
  let txt = `Calibration terminée : erreur ${fmt(r.rms, 3)} px sur ${ok} images.`;
  if (r.rms > 1) txt += ' Erreur élevée : vérifier la netteté et la planéité de la feuille.';
  if (rejets.length) txt += ` Ignorées : ${rejets.join(', ')}.`;
  message(txt, r.rms > 1 ? 'alerte' : 'ok');
  if (etat.vues.length) {
    message(txt + ' Les photos de l\'objet déjà prises doivent être refaites.', 'alerte');
    viderPhotos();
  }
}

function exporterCalib() {
  if (!etat.calib) return;
  telecharger(new Blob([JSON.stringify(etat.calib, null, 2)], { type: 'application/json' }), 'calibration-telephone.json');
}

async function importerCalib(fichier) {
  try {
    const c = JSON.parse(await fichier.text());
    if (!Array.isArray(c.K) || c.K.length !== 9 || !Array.isArray(c.dist) || !c.largeur || !c.hauteur) {
      throw new Error('fichier non reconnu');
    }
    etat.calib = { images: 0, rms: 0, ...c };
    ecrire(CLE_CALIB, etat.calib);
    majCalib();
    message('Calibration importée.', 'ok');
  } catch (err) {
    message('Import de la calibration : ' + err.message, 'erreur');
  }
}

/* ---------- Photos de l'objet ---------- */

function creerVignette(vue) {
  const el = document.createElement('button');
  el.type = 'button';
  el.className = 'vignette';
  const c = document.createElement('canvas');
  const leg = document.createElement('span');
  leg.className = 'legende';
  el.append(c, leg);
  el.addEventListener('click', () => {
    if (vue.etat !== 'ok' || etat.occupe) return;
    vue.actif = !vue.actif;
    majVignette(vue);
    majResume();
  });
  vue.el = el; vue.canvas = c; vue.legende = leg;
  $('photos-grille').append(el);
}

function majVignette(vue) {
  vue.el.dataset.etat = vue.etat;
  vue.el.classList.toggle('exclue', vue.etat === 'ok' && !vue.actif);
  vue.el.setAttribute('aria-pressed', vue.etat === 'ok' && vue.actif ? 'true' : 'false');
  if (vue.etat === 'attente') vue.legende.textContent = 'Analyse...';
  else if (vue.etat === 'erreur') vue.legende.textContent = vue.raison;
  else vue.legende.textContent = vue.actif ? `${vue.n} coins, ${fmt(vue.err, 2)} px` : 'Exclue';
}

function dessinerMasque(vue) {
  const tmp = document.createElement('canvas');
  tmp.width = vue.mw; tmp.height = vue.mh;
  const ctx = tmp.getContext('2d');
  const img = ctx.createImageData(vue.mw, vue.mh);
  // Fond assombri, objet detoure laisse intact : un masque faux se voit immediatement
  for (let i = 0; i < vue.masque.length; i++) {
    if (!vue.masque[i]) img.data[4 * i + 3] = 175;
  }
  ctx.putImageData(img, 0, 0);
  vue.canvas.getContext('2d').drawImage(tmp, 0, 0, vue.canvas.width, vue.canvas.height);
}

function majResume() {
  const total = etat.vues.length;
  const ok = etat.vues.filter((v) => v.etat === 'ok').length;
  const actives = etat.vues.filter((v) => v.etat === 'ok' && v.actif).length;
  $('photos-resume').textContent = total
    ? `${total} photos, ${ok} exploitables, ${actives} retenues. L'objet détouré reste clair, le fond est assombri : toucher une vignette pour l'exclure si le détourage est faux.`
    : '';
  majBoutons();
}

function majBoutons() {
  const actives = etat.vues.filter((v) => v.etat === 'ok' && v.actif).length;
  $('recon-lancer').disabled = etat.occupe || actives < MIN_VUES;
  $('photos-ajouter').classList.toggle('desactive', etat.occupe || !etat.calib);
  $('photos-direct').disabled = etat.occupe || !etat.calib;
  $('photos-effacer').disabled = etat.occupe || !etat.vues.length;
}

function viderPhotos() {
  etat.vues = [];
  $('photos-grille').replaceChildren();
  majResume();
}

async function ajouterPhotos(fichiers) {
  if (!etat.calib) { message('Faire la calibration du téléphone avant de charger les photos de l\'objet.', 'alerte'); return; }
  if (etat.calib.source === 'direct') {
    message('La calibration a été faite en direct : le cadrage de la caméra en direct diffère de celui de l\'appareil photo. Utiliser la prise en direct, ou refaire la calibration avec des photos.', 'alerte');
    return;
  }
  occuper(true);
  const barre = $('photos-progres');
  barre.hidden = false; barre.value = 0;
  try {
    await initVision();
    await initSeg();
    const c = etat.calib;
    for (let i = 0; i < fichiers.length; i++) {
      const f = fichiers[i];
      message(`Analyse de la photo ${i + 1} / ${fichiers.length} : planche et détourage`);
      barre.value = i / fichiers.length;
      const vue = { nom: f.name, etat: 'attente', actif: true };
      etat.vues.push(vue);
      creerVignette(vue);
      majVignette(vue);
      let bmp = null;
      try {
        const img = await chargerImage(f);
        bmp = img.bmp;
        const { W, H } = img;
        vue.canvas.width = 160;
        vue.canvas.height = Math.round((160 * H) / W);
        vue.canvas.getContext('2d').drawImage(bmp, 0, 0, vue.canvas.width, vue.canvas.height);
        if (W !== c.largeur || H !== c.hauteur) {
          throw new Error(W === c.hauteur && H === c.largeur ? 'Orientation différente de la calibration' : 'Taille différente de la calibration');
        }
        const gris = versGris(pixels(bmp, W, H));
        const rgba = pixels(bmp, TAILLE_SEG, TAILLE_SEG);
        bmp.close(); bmp = null;
        const ms = TAILLE_MASQUE / Math.max(W, H);
        vue.mw = Math.round(W * ms); vue.mh = Math.round(H * ms);
        const [p, s] = await Promise.all([
          vision.appel('pose', { gris, w: W, h: H, K: c.K, dist: c.dist }, [gris.buffer]),
          seg.appel('segmenter', { rgba, S: TAILLE_SEG, mw: vue.mw, mh: vue.mh }, [rgba.buffer]),
        ]);
        vue.masque = s.masque;
        dessinerMasque(vue);
        if (!p.R) throw new Error('Planche non détectée');
        let surface = 0;
        for (let k = 0; k < s.masque.length; k++) surface += s.masque[k];
        if (surface < s.masque.length * 0.001) throw new Error('Objet non détouré');
        if (p.err > ERREUR_POSE_MAX) throw new Error(`Pose imprecise (${fmt(p.err, 1)} px)`);
        Object.assign(vue, { R: p.R, t: p.t, n: p.n, err: p.err, etat: 'ok' });
      } catch (err) {
        vue.etat = 'erreur';
        vue.raison = err.message;
        if (bmp) bmp.close();
      }
      majVignette(vue);
      majResume();
    }
    message(`${etat.vues.filter((v) => v.etat === 'ok').length} photos exploitables sur ${etat.vues.length}.`, 'ok');
  } catch (err) {
    message('Photos : ' + err.message, 'erreur');
  } finally {
    barre.hidden = true;
    occuper(false);
    majResume();
  }
}

/* ---------- Prise en direct ---------- */

const DIRECT = {
  vitesseMax: 120,      // deplacement maximal des coins de la planche dans l'image (px/s) pour prendre une image
  coinsMin: 12,         // coins ChArUco minimum pour une vue d'objet
  calibImages: 20,      // images a prendre pour la calibration
  calibCoinsMin: 20,    // coins minimum pour une image de calibration
  calibEcartDeg: 12,    // ecart d'angle minimum entre deux images de calibration
  couverture: 0.85,     // part des cases du dome a remplir pour terminer automatiquement
};
// Dome de prise de vue : anneaux d'elevation (degres) decoupes en cases d'azimut
const ANNEAUX = [
  { min: 20, max: 40, n: 16 },
  { min: 40, max: 60, n: 16 },
  { min: 60, max: 78, n: 12 },
  { min: 78, max: 90.01, n: 4 },
];
const NB_CASES = ANNEAUX.reduce((s, a) => s + a.n, 0);
const CASES_VISEES = Math.ceil(NB_CASES * DIRECT.couverture);

let direct = null;

const pause = (ms) => new Promise((r) => setTimeout(r, ms));

function projeterPoint(K, d, R, t, X, Y, Z) {
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

/* Direction de la camera vue du centre de la planche (coordonnees nominales de la planche). */
function directionCamera(v) {
  const R = v.R, t = v.t;
  const C = [
    -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]),
    -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]),
    -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]),
  ];
  const dx = C[0] - (PLANCHE.nx * PLANCHE.carre) / 2;
  const dy = C[1] - (PLANCHE.ny * PLANCHE.carre) / 2;
  const dz = Math.abs(C[2]);
  const n = Math.hypot(dx, dy, dz) || 1;
  const elev = (Math.atan2(dz, Math.hypot(dx, dy)) * 180) / Math.PI;
  const az = ((Math.atan2(dy, dx) * 180) / Math.PI + 360) % 360;
  return { elev, az, u: [dx / n, dy / n, dz / n] };
}

function caseDe(dir) {
  const i = ANNEAUX.findIndex((a) => dir.elev >= a.min && dir.elev < a.max);
  if (i < 0) return null;
  return `${i}-${Math.floor(dir.az / (360 / ANNEAUX[i].n)) % ANNEAUX[i].n}`;
}

function coinsPlanche() {
  const L = PLANCHE.nx * PLANCHE.carre, H = PLANCHE.ny * PLANCHE.carre;
  return [[0, 0], [L, 0], [L, H], [0, H]];
}

function cameraDirect(d, W, H) {
  if (d.mode === 'calib') {
    const f = 0.8 * Math.max(W, H); // estimation grossiere, suffisante pour juger la diversite des angles
    return { K: [f, 0, (W - 1) / 2, 0, f, (H - 1) / 2, 0, 0, 1], dist: [0, 0, 0, 0, 0] };
  }
  const c = etat.calib;
  if (W === c.largeur && H === c.hauteur) return { K: c.K, dist: c.dist };
  if (W === c.hauteur && H === c.largeur) return { erreur: 'Tourner le téléphone dans l\'orientation utilisée pour la calibration.' };
  return { erreur: `La caméra donne ${W} x ${H}, la calibration a été faite en ${c.largeur} x ${c.hauteur} : refaire la calibration en direct.` };
}

function consigneScan(d, dir) {
  const i = ANNEAUX.findIndex((a) => dir.elev >= a.min && dir.elev < a.max);
  const vides = ANNEAUX.map((a, k) => {
    let n = 0;
    for (let b = 0; b < a.n; b++) if (!d.cases.has(`${k}-${b}`)) n++;
    return n;
  });
  if (vides[i] > 0) return 'Vue déjà prise : continuer à tourner autour de l\'objet';
  const plusHaut = vides.findIndex((n, k) => k > i && n > 0);
  const plusBas = vides.findIndex((n, k) => k < i && n > 0);
  if (plusBas >= 0) return 'Cet étage est complet : descendre le téléphone';
  if (plusHaut >= 0) return 'Cet étage est complet : monter le téléphone';
  return 'Couverture complète';
}

function evaluer(d, p, t, cam) {
  if (!p.R || p.n < DIRECT.coinsMin) {
    d.precedent = null;
    return { consigne: 'Planche non détectée : la garder dans le cadre, au moins en partie', niveau: 'alerte' };
  }
  const coins = coinsPlanche().map(([X, Y]) => projeterPoint(cam.K, cam.dist, p.R, p.t, X, Y, 0));
  let vitesse = Infinity;
  if (d.precedent && t - d.precedent.t < 1500) {
    let m = 0;
    coins.forEach((q, i) => { m = Math.max(m, Math.hypot(q[0] - d.precedent.coins[i][0], q[1] - d.precedent.coins[i][1])); });
    vitesse = m / ((t - d.precedent.t) / 1000);
  }
  d.precedent = { t, coins };
  const direction = directionCamera(p);
  if (vitesse > DIRECT.vitesseMax) return { direction, consigne: 'Stabiliser le téléphone', niveau: '' };
  if (d.mode === 'scan') {
    if (p.err > ERREUR_POSE_MAX) return { direction, consigne: 'Image peu nette : stabiliser le téléphone', niveau: 'alerte' };
    const c = caseDe(direction);
    if (!c) return { direction, consigne: 'Vue trop rasante : monter le téléphone', niveau: 'alerte' };
    if (d.cases.has(c)) return { direction, consigne: consigneScan(d, direction), niveau: '' };
    return { ok: true, direction, cle: c };
  }
  if (direction.elev < 20) return { direction, consigne: 'Vue trop rasante : monter le téléphone', niveau: 'alerte' };
  if (p.n < DIRECT.calibCoinsMin) return { direction, consigne: 'Cadrer une plus grande partie de la planche', niveau: 'alerte' };
  const proche = d.directions.some((u) => (Math.acos(Math.min(1, u[0] * direction.u[0] + u[1] * direction.u[1] + u[2] * direction.u[2])) * 180) / Math.PI < DIRECT.calibEcartDeg);
  if (proche) return { direction, consigne: 'Changer d\'angle : tourner autour de la planche, plus haut ou plus bas', niveau: '' };
  return { ok: true, direction };
}

function dessinerCalque(d, p, cam, W, H, ok) {
  const cnv = $('direct-calque');
  const r = cnv.getBoundingClientRect();
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const cw = Math.round(r.width * dpr), ch = Math.round(r.height * dpr);
  if (cnv.width !== cw || cnv.height !== ch) { cnv.width = cw; cnv.height = ch; }
  const ctx = cnv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, r.width, r.height);
  if (!p || !p.R) return;
  const s = Math.min(r.width / W, r.height / H);
  const ox = (r.width - W * s) / 2, oy = (r.height - H * s) / 2;
  const tracer = (pts) => {
    ctx.beginPath();
    pts.forEach(([X, Y], i) => {
      const [u, v] = projeterPoint(cam.K, cam.dist, p.R, p.t, X, Y, 0);
      if (i === 0) ctx.moveTo(ox + u * s, oy + v * s); else ctx.lineTo(ox + u * s, oy + v * s);
    });
    ctx.closePath();
    ctx.stroke();
  };
  const contour = (x0, y0, x1, y1) => {
    const pts = [];
    const n = 12;
    for (let i = 0; i < n; i++) pts.push([x0 + ((x1 - x0) * i) / n, y0]);
    for (let i = 0; i < n; i++) pts.push([x1, y0 + ((y1 - y0) * i) / n]);
    for (let i = 0; i < n; i++) pts.push([x1 - ((x1 - x0) * i) / n, y1]);
    for (let i = 0; i < n; i++) pts.push([x0, y1 - ((y1 - y0) * i) / n]);
    return pts;
  };
  const L = PLANCHE.nx * PLANCHE.carre, Hp = PLANCHE.ny * PLANCHE.carre;
  ctx.lineWidth = 3;
  ctx.strokeStyle = ok ? '#4fd1a5' : 'rgba(255, 255, 255, 0.85)';
  tracer(contour(0, 0, L, Hp));
  if (d.mode === 'scan') {
    // Zone de travail : l'objet doit tenir dans ce carre
    const e = reglages().emprise / facteurPlanche() / 2;
    ctx.setLineDash([8, 6]);
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(255, 214, 102, 0.9)';
    tracer(contour(L / 2 - e, Hp / 2 - e, L / 2 + e, Hp / 2 + e));
    ctx.setLineDash([]);
  }
}

function dessinerDome(d, courant) {
  const cnv = $('direct-dome');
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const taille = 150;
  if (cnv.width !== taille * dpr) { cnv.width = taille * dpr; cnv.height = taille * dpr; }
  const ctx = cnv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, taille, taille);
  const c = taille / 2, R = taille / 2 - 6;
  const rayon = (e) => (R * (90 - Math.min(90, e))) / 70;
  ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
  ctx.beginPath(); ctx.arc(c, c, R + 4, 0, 2 * Math.PI); ctx.fill();
  if (d.mode === 'scan') {
    ANNEAUX.forEach((a, i) => {
      const pas = (2 * Math.PI) / a.n;
      for (let b = 0; b < a.n; b++) {
        ctx.beginPath();
        ctx.arc(c, c, rayon(a.min), b * pas, (b + 1) * pas);
        ctx.arc(c, c, Math.max(0, rayon(a.max)), (b + 1) * pas, b * pas, true);
        ctx.closePath();
        ctx.fillStyle = d.cases.has(`${i}-${b}`) ? 'rgba(79, 209, 165, 0.9)' : 'rgba(255, 255, 255, 0.12)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    });
  } else {
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
    for (const e of [20, 40, 60, 78]) { ctx.beginPath(); ctx.arc(c, c, rayon(e), 0, 2 * Math.PI); ctx.stroke(); }
    ctx.fillStyle = 'rgba(79, 209, 165, 0.95)';
    for (const u of d.directions) {
      const e = (Math.asin(Math.min(1, u[2])) * 180) / Math.PI, a = Math.atan2(u[1], u[0]);
      ctx.beginPath(); ctx.arc(c + rayon(e) * Math.cos(a), c + rayon(e) * Math.sin(a), 4, 0, 2 * Math.PI); ctx.fill();
    }
  }
  if (courant) {
    const a = (courant.az * Math.PI) / 180, r = rayon(Math.max(20, courant.elev));
    ctx.beginPath(); ctx.arc(c + r * Math.cos(a), c + r * Math.sin(a), 6, 0, 2 * Math.PI);
    ctx.fillStyle = '#ffffff'; ctx.fill();
    ctx.lineWidth = 2; ctx.strokeStyle = '#000000'; ctx.stroke();
  }
}

function majDirect(d, consigne, niveau) {
  if (consigne !== undefined) {
    $('direct-consigne').textContent = consigne;
    $('direct-consigne').dataset.niveau = niveau || '';
  }
  if (d.mode === 'calib') {
    $('direct-compteur').textContent = `${d.prises} / ${DIRECT.calibImages} images`;
    $('direct-terminer').disabled = d.prises < 8;
  } else {
    let txt = `${d.cases.size} / ${NB_CASES} vues`;
    if (d.enCours) txt += ` - détourage : ${d.enCours} en attente`;
    $('direct-compteur').textContent = txt;
    $('direct-terminer').disabled = d.cases.size < MIN_VUES;
  }
}

function signalerPrise() {
  const f = $('direct-flash');
  f.classList.remove('actif');
  void f.offsetWidth;
  f.classList.add('actif');
  try { navigator.vibrate && navigator.vibrate(30); } catch (e) { /* non supporte */ }
}

async function prendre(d, p, ev, W, H) {
  signalerPrise();
  if (d.mode === 'calib') {
    const r = await vision.appel('calib-garder');
    if (r.garde) { d.directions.push(ev.direction.u); d.prises++; d.taille = [W, H]; }
    return;
  }
  d.cases.add(ev.cle);
  const vue = { nom: `direct-${etat.vues.length + 1}`, etat: 'attente', actif: true, R: p.R, t: p.t, n: p.n, err: p.err };
  etat.vues.push(vue);
  creerVignette(vue);
  vue.canvas.width = 160;
  vue.canvas.height = Math.round((160 * H) / W);
  vue.canvas.getContext('2d').drawImage(d.toile, 0, 0, vue.canvas.width, vue.canvas.height);
  majVignette(vue);
  const ms = TAILLE_MASQUE / Math.max(W, H);
  vue.mw = Math.round(W * ms); vue.mh = Math.round(H * ms);
  const rgba = pixels(d.toile, TAILLE_SEG, TAILLE_SEG);
  d.enCours++;
  // Detourage en file d'attente : la prise de vue continue pendant le calcul
  d.file = d.file.then(async () => {
    try {
      const s = await seg.appel('segmenter', { rgba, S: TAILLE_SEG, mw: vue.mw, mh: vue.mh }, [rgba.buffer]);
      vue.masque = s.masque;
      dessinerMasque(vue);
      let surface = 0;
      for (let k = 0; k < s.masque.length; k++) surface += s.masque[k];
      if (surface < s.masque.length * 0.001) throw new Error('Objet non détouré');
      vue.etat = 'ok';
    } catch (err) {
      vue.etat = 'erreur';
      vue.raison = err.message;
    }
    d.enCours--;
    majVignette(vue);
    majResume();
    if (d.actif) majDirect(d);
  });
}

/* Scan termine : couverture globale atteinte et chaque etage du dome (dessus compris) rempli aux trois quarts. */
function fini(d) {
  if (d.mode === 'calib') return d.prises >= DIRECT.calibImages;
  if (d.cases.size < CASES_VISEES) return false;
  return ANNEAUX.every((a, i) => {
    let n = 0;
    for (let b = 0; b < a.n; b++) if (d.cases.has(`${i}-${b}`)) n++;
    return n >= Math.ceil(a.n * 0.75);
  });
}

async function ouvrirDirect(mode) {
  if (etat.occupe || direct) return;
  if (mode === 'scan' && !etat.calib) { message('Faire d\'abord la calibration en direct.', 'alerte'); return; }
  if (mode === 'scan' && etat.calib.source === 'photos') {
    message('La calibration a été faite avec des photos : le cadrage de la caméra en direct est différent. Refaire la calibration en direct.', 'alerte');
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    message('Caméra inaccessible : ouvrir l\'appli en HTTPS (GitHub Pages) dans un navigateur récent.', 'erreur');
    return;
  }
  occuper(true);
  let flux = null;
  try {
    await initVision();
    if (mode === 'scan') await initSeg();
    if (mode === 'calib') await vision.appel('calib-reset');
    flux = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1440 } },
    });
  } catch (err) {
    occuper(false);
    const refus = err && (err.name === 'NotAllowedError' || err.name === 'SecurityError');
    message('Caméra : ' + (refus ? 'accès refusé. Autoriser la caméra pour ce site dans les réglages du navigateur.' : err.message), 'erreur');
    return;
  }
  const video = $('direct-video');
  video.srcObject = flux;
  const d = {
    mode, flux, video, actif: true, action: 'arreter',
    cases: new Set(), directions: [], prises: 0, taille: null,
    precedent: null, file: Promise.resolve(), enCours: 0,
    toile: document.createElement('canvas'), verrou: null,
  };
  direct = d;
  if (mode === 'scan') {
    // Reprise : les vues deja prises comptent dans la couverture
    for (const v of etat.vues) if (v.R && v.etat !== 'erreur') { const c = caseDe(directionCamera(v)); if (c) d.cases.add(c); }
  }
  flux.getVideoTracks().forEach((tr) => { tr.onended = () => { d.actif = false; }; });
  try { if (navigator.wakeLock) d.verrou = await navigator.wakeLock.request('screen'); } catch (e) { d.verrou = null; }
  $('direct-titre').textContent = mode === 'calib' ? 'Calibration en direct' : 'Scan en direct';
  $('direct-terminer').textContent = mode === 'calib' ? 'Calculer la calibration' : 'Terminer et reconstruire';
  $('direct').hidden = false;
  document.body.classList.add('direct-ouvert');
  majDirect(d, mode === 'calib'
    ? 'Planche seule, sans objet : tourner autour en variant hauteur et distance'
    : 'Tourner lentement autour de l\'objet, les images sont prises automatiquement');
  dessinerDome(d, null);
  try { await video.play(); } catch (e) { /* lecture relancee par l'utilisateur */ }
  boucleDirect(d);
}

async function boucleDirect(d) {
  const ctx = d.toile.getContext('2d', { willReadFrequently: true });
  try {
    while (d.actif) {
      const v = d.video;
      if (v.readyState < 2 || !v.videoWidth) { await pause(60); continue; }
      const s = Math.min(1, TAILLE_TRAVAIL / Math.max(v.videoWidth, v.videoHeight));
      const W = Math.round(v.videoWidth * s), H = Math.round(v.videoHeight * s);
      if (d.toile.width !== W || d.toile.height !== H) { d.toile.width = W; d.toile.height = H; d.precedent = null; }
      const cam = cameraDirect(d, W, H);
      if (cam.erreur) { majDirect(d, cam.erreur, 'alerte'); dessinerCalque(d, null); await pause(400); continue; }
      ctx.drawImage(v, 0, 0, W, H);
      const t = performance.now();
      const gris = versGris(ctx.getImageData(0, 0, W, H).data);
      let p;
      try {
        p = await vision.appel('pose', { gris, w: W, h: H, K: cam.K, dist: cam.dist }, [gris.buffer]);
      } catch (err) {
        majDirect(d, 'Erreur d\'analyse : ' + err.message, 'alerte');
        await pause(300);
        continue;
      }
      if (!d.actif) break;
      const ev = evaluer(d, p, t, cam);
      dessinerCalque(d, p, cam, W, H, ev.ok);
      dessinerDome(d, ev.direction);
      if (ev.ok) {
        await prendre(d, p, ev, W, H);
        majDirect(d, d.mode === 'calib' ? 'Image prise' : 'Vue prise', 'ok');
        dessinerDome(d, ev.direction);
      } else {
        majDirect(d, ev.consigne, ev.niveau);
      }
      if (fini(d)) { d.action = 'terminer'; break; }
    }
  } finally {
    d.actif = false;
    d.flux.getTracks().forEach((tr) => tr.stop());
    d.video.srcObject = null;
    try { if (d.verrou) await d.verrou.release(); } catch (e) { /* deja libere */ }
    $('direct').hidden = true;
    document.body.classList.remove('direct-ouvert');
  }
  await finaliserDirect(d);
}

async function finaliserDirect(d) {
  try {
    if (d.mode === 'calib') {
      if (d.action !== 'terminer') { message('Calibration en direct annulée.', ''); return; }
      await calculerCalibration(d.taille, d.prises, [], 'direct');
      return;
    }
    const barre = $('photos-progres');
    if (d.enCours) {
      barre.hidden = false;
      const total = d.enCours;
      while (d.enCours) {
        barre.value = 1 - d.enCours / total;
        message(`Détourage des dernières vues : ${d.enCours} restantes...`);
        await Promise.race([d.file, pause(250)]);
      }
      barre.hidden = true;
    }
    await d.file;
    const actives = etat.vues.filter((v) => v.etat === 'ok' && v.actif).length;
    if (d.action !== 'terminer') {
      message(`Prise de vue arrêtée : ${actives} vues exploitables. Reprendre la prise en direct pour compléter, ou lancer la reconstruction.`, '');
      return;
    }
    if (actives < MIN_VUES) {
      message(`Seulement ${actives} vues exploitables (${MIN_VUES} minimum) : reprendre la prise en direct.`, 'alerte');
      return;
    }
    occuper(false);
    await reconstruire();
  } catch (err) {
    message((d.mode === 'calib' ? 'Calibration : ' : 'Prise en direct : ') + err.message, 'erreur');
  } finally {
    direct = null;
    majCalib();
    occuper(false);
    majResume();
  }
}

/* ---------- Reconstruction ---------- */

function suivreRecon(barre) {
  const noms = { sculpture: 'Sculpture du volume', nettoyage: 'Nettoyage', maillage: 'Maillage' };
  recon.surProgres = (m) => {
    barre.value = m.etape === 'sculpture' ? m.valeur * 0.8 : m.etape === 'maillage' ? 0.8 + m.valeur * 0.2 : 0.8;
    message(`${noms[m.etape] || m.etape}...`);
  };
}

async function reconstruire() {
  const actives = etat.vues.filter((v) => v.etat === 'ok' && v.actif);
  if (actives.length < MIN_VUES) return;
  occuper(true);
  const barre = $('recon-progres');
  barre.hidden = false; barre.value = 0;
  suivreRecon(barre);
  try {
    const r = reglages();
    const f = facteurPlanche();
    const c = etat.calib;
    const { mw, mh } = actives[0];
    const sx = mw / c.largeur, sy = mh / c.hauteur;
    const K = c.K.slice();
    K[0] *= sx; K[1] *= sx; K[2] = (K[2] + 0.5) * sx - 0.5;
    K[4] *= sy; K[5] = (K[5] + 0.5) * sy - 0.5;
    const centre = [(PLANCHE.nx * PLANCHE.carre * f) / 2, (PLANCHE.ny * PLANCHE.carre * f) / 2];
    const t0 = performance.now();
    const res = await recon.appel('reconstruire', {
      vues: actives.map((v) => ({ R: v.R, t: v.t.map((x) => x * f), masque: v.masque })),
      mw, mh, K, dist: c.dist, centre,
      params: { voxel: r.voxel, emprise: r.emprise, hauteur: r.hauteur, tolerance: r.tolerance, lissage: r.lissage },
    });
    const duree = (performance.now() - t0) / 1000;
    etat.modele = { base: res.positions, indices: res.indices, angle: 0, facteurs: [1, 1, 1] };
    $('d-couper').value = '';
    $('resultat').hidden = false;
    await majModele(true);
    let txt = `Reconstruction terminée en ${fmt(duree, 1)} s avec ${actives.length} vues (${res.indices.length / 3} triangles).`;
    if (res.bord) txt += ' Attention : l\'objet touche le bord du volume de travail, augmenter l\'emprise ou la hauteur, ou vérifier les masques.';
    message(txt, res.bord ? 'alerte' : 'ok');
    $('resultat').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (err) {
    message('Reconstruction : ' + err.message, 'erreur');
  } finally {
    barre.hidden = true;
    occuper(false);
  }
}

async function couper() {
  if (!etat.modele) return;
  const v = parseFloat(String($('d-couper').value).replace(',', '.'));
  const z = Number.isFinite(v) && v > 0 ? v : 0;
  occuper(true);
  const barre = $('recon-progres');
  barre.hidden = false; barre.value = 0.8;
  suivreRecon(barre);
  try {
    const res = await recon.appel('remailler', { couperZ: z, lissage: reglages().lissage });
    etat.modele.base = res.positions;
    etat.modele.indices = res.indices;
    etat.modele.facteurs = [1, 1, 1];
    await majModele(false);
    message(z ? `Objet coupé à ${fmt(z, 1)} mm de hauteur.` : 'Coupe supprimée.', 'ok');
  } catch (err) {
    message('Coupe : ' + err.message, 'erreur');
  } finally {
    barre.hidden = true;
    occuper(false);
  }
}

/* ---------- Transformations et dimensions ---------- */

function transformer(base, angle, facteurs) {
  const out = new Float32Array(base.length);
  const c = Math.cos(angle), s = Math.sin(angle);
  for (let i = 0; i < base.length; i += 3) {
    const x = base[i], y = base[i + 1];
    out[i] = (c * x - s * y) * facteurs[0];
    out[i + 1] = (s * x + c * y) * facteurs[1];
    out[i + 2] = base[i + 2] * facteurs[2];
  }
  // Centre en X/Y, base posee sur Z = 0
  const b = bornes(out);
  const dx = (b[0] + b[3]) / 2, dy = (b[1] + b[4]) / 2, dz = b[2];
  for (let i = 0; i < out.length; i += 3) { out[i] -= dx; out[i + 1] -= dy; out[i + 2] -= dz; }
  return out;
}

function bornes(p) {
  const b = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
  for (let i = 0; i < p.length; i += 3) {
    for (let a = 0; a < 3; a++) {
      if (p[i + a] < b[a]) b[a] = p[i + a];
      if (p[i + a] > b[a + 3]) b[a + 3] = p[i + a];
    }
  }
  return b;
}

function dimensions(p) {
  const b = bornes(p);
  return [b[3] - b[0], b[4] - b[1], b[5] - b[2]];
}

/* Angle autour de Z qui minimise l'emprise X/Y (aretes alignees sur les axes). */
function angleAlignement(p) {
  const pts = [];
  const pas = Math.max(1, Math.floor(p.length / 3 / 20000));
  for (let i = 0; i < p.length; i += 3 * pas) pts.push([p[i], p[i + 1]]);
  pts.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const croix = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const bas = [], haut = [];
  for (const q of pts) { while (bas.length >= 2 && croix(bas[bas.length - 2], bas[bas.length - 1], q) <= 0) bas.pop(); bas.push(q); }
  for (let i = pts.length - 1; i >= 0; i--) { const q = pts[i]; while (haut.length >= 2 && croix(haut[haut.length - 2], haut[haut.length - 1], q) <= 0) haut.pop(); haut.push(q); }
  const env = bas.slice(0, -1).concat(haut.slice(0, -1));
  let meilleur = 0, aireMin = Infinity;
  for (let deg = 0; deg < 90; deg += 0.25) {
    const a = (deg * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a);
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const [x, y] of env) {
      const u = c * x - s * y, v = s * x + c * y;
      if (u < x0) x0 = u; if (u > x1) x1 = u; if (v < y0) y0 = v; if (v > y1) y1 = v;
    }
    const aire = (x1 - x0) * (y1 - y0);
    if (aire < aireMin - 1e-9) { aireMin = aire; meilleur = a; }
  }
  return meilleur;
}

async function majModele(recadrer) {
  const m = etat.modele;
  m.courant = transformer(m.base, m.angle, m.facteurs);
  m.dims = dimensions(m.courant);
  const brut = dimensions(transformer(m.base, m.angle, [1, 1, 1]));
  AXES.forEach((a, i) => {
    $('m-' + a).textContent = fmt(brut[i]);
    $('d-' + a).value = m.dims[i].toFixed(2);
  });
  const modifie = m.facteurs.some((f) => Math.abs(f - 1) > 1e-6);
  $('d-facteurs').textContent = modifie
    ? 'Facteurs appliqués : ' + AXES.map((a, i) => `${a.toUpperCase()} x${fmt(m.facteurs[i], 4)}`).join(', ')
    : 'Dimensions du scan, sans correction.';
  await afficher3D(m.courant, m.indices, recadrer);
}

function saisie(axe) {
  if (!$('d-lier').checked || !etat.modele) return;
  const i = AXES.indexOf(axe);
  const v = parseFloat(String($('d-' + axe).value).replace(',', '.'));
  if (!(v > 0)) return;
  const r = v / etat.modele.dims[i];
  AXES.forEach((a, j) => { if (j !== i) $('d-' + a).value = (etat.modele.dims[j] * r).toFixed(2); });
}

async function appliquerCotes() {
  const m = etat.modele;
  if (!m) return;
  const cibles = AXES.map((a) => parseFloat(String($('d-' + a).value).replace(',', '.')));
  if (cibles.some((v) => !(v > 0))) { message('Saisir des dimensions positives en mm.', 'erreur'); return; }
  m.facteurs = m.facteurs.map((f, i) => f * (cibles[i] / m.dims[i]));
  await majModele(false);
  message('Dimensions appliquées.', 'ok');
}

async function reinitialiser() {
  if (!etat.modele) return;
  etat.modele.angle = 0;
  etat.modele.facteurs = [1, 1, 1];
  await majModele(false);
  message('Dimensions du scan restaurées.', 'ok');
}

async function aligner() {
  const m = etat.modele;
  if (!m) return;
  m.angle = angleAlignement(m.base);
  m.facteurs = [1, 1, 1];
  await majModele(true);
  message(`Objet aligné sur les axes (rotation de ${fmt((m.angle * 180) / Math.PI, 1)} degrés).`, 'ok');
}

/* ---------- Export STL ---------- */

function stlBinaire(pos, tri) {
  const nt = tri.length / 3;
  const buf = new ArrayBuffer(84 + 50 * nt);
  const dv = new DataView(buf);
  const entete = 'Scan 3D - unites : mm';
  for (let i = 0; i < entete.length; i++) dv.setUint8(i, entete.charCodeAt(i));
  dv.setUint32(80, nt, true);
  let o = 84;
  for (let t = 0; t < tri.length; t += 3) {
    const a = 3 * tri[t], b = 3 * tri[t + 1], c = 3 * tri[t + 2];
    const ux = pos[b] - pos[a], uy = pos[b + 1] - pos[a + 1], uz = pos[b + 2] - pos[a + 2];
    const vx = pos[c] - pos[a], vy = pos[c + 1] - pos[a + 1], vz = pos[c + 2] - pos[a + 2];
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const l = Math.hypot(nx, ny, nz) || 1;
    nx /= l; ny /= l; nz /= l;
    for (const v of [nx, ny, nz]) { dv.setFloat32(o, v, true); o += 4; }
    for (const s of [a, b, c]) for (let k = 0; k < 3; k++) { dv.setFloat32(o, pos[s + k], true); o += 4; }
    dv.setUint16(o, 0, true); o += 2;
  }
  return new Blob([buf], { type: 'model/stl' });
}

function telecharger(blob, nom) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = nom;
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

function exporterSTL() {
  const m = etat.modele;
  if (!m) return;
  const d = new Date();
  const p = (n) => String(n).padStart(2, '0');
  const nom = `scan-${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}.stl`;
  telecharger(stlBinaire(m.courant, m.indices), nom);
}

/* ---------- Vue 3D ---------- */

let v3d = null;

async function initVue3D() {
  const THREE = await import('three');
  const { OrbitControls } = await import('three/addons/controls/OrbitControls.js');
  const conteneur = $('vue3d');
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  conteneur.append(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 5000);
  camera.up.set(0, 0, 1);
  const controles = new OrbitControls(camera, renderer.domElement);
  controles.enableDamping = true;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8a80, 1.6));
  const lumiere = new THREE.DirectionalLight(0xffffff, 1.6);
  scene.add(lumiere);
  const grille = new THREE.GridHelper(200, 20, 0x888880, 0xbbbbb4);
  grille.rotation.x = Math.PI / 2;
  grille.material.transparent = true;
  grille.material.opacity = 0.5;
  scene.add(grille);
  const materiau = new THREE.MeshStandardMaterial({ color: 0xc9a45c, roughness: 0.6, metalness: 0.05 });
  const geometrie = new THREE.BufferGeometry();
  const maillage = new THREE.Mesh(geometrie, materiau);
  scene.add(maillage);
  const boite = new THREE.Box3Helper(new THREE.Box3(), 0x2f6f5e);
  scene.add(boite);
  const redim = () => {
    const w = conteneur.clientWidth, h = conteneur.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(redim).observe(conteneur);
  redim();
  const boucle = () => {
    controles.update();
    lumiere.position.copy(camera.position);
    renderer.render(scene, camera);
    requestAnimationFrame(boucle);
  };
  boucle();
  v3d = { THREE, camera, controles, geometrie, boite };
}

async function afficher3D(pos, indices, recadrer) {
  try {
    if (!v3d) await initVue3D();
  } catch (err) {
    $('vue3d').textContent = 'Aperçu 3D indisponible (bibliothèque non chargée). L\'export STL fonctionne quand même.';
    return;
  }
  const { THREE, camera, controles, geometrie, boite } = v3d;
  geometrie.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geometrie.setIndex(new THREE.BufferAttribute(indices, 1));
  geometrie.computeVertexNormals();
  geometrie.computeBoundingBox();
  geometrie.computeBoundingSphere();
  boite.box.copy(geometrie.boundingBox);
  if (recadrer) {
    const s = geometrie.boundingSphere;
    const d = (s.radius / Math.sin((camera.fov * Math.PI) / 360)) * 1.15;
    controles.target.copy(s.center);
    camera.position.set(s.center.x + d * 0.6, s.center.y - d * 0.75, s.center.z + d * 0.45);
    camera.near = d / 100; camera.far = d * 20;
    camera.updateProjectionMatrix();
  }
}

/* ---------- Branchements ---------- */

function brancherFichiers(id, action) {
  $(id).addEventListener('change', (e) => {
    const fichiers = Array.from(e.target.files || []);
    e.target.value = '';
    if (fichiers.length && !etat.occupe) action(fichiers);
  });
}

function demarrer() {
  chargerReglages();
  majCalib();
  majResume();
  brancherFichiers('calib-fichiers', calibrer);
  brancherFichiers('photos-fichiers', ajouterPhotos);
  brancherFichiers('calib-importer', (f) => importerCalib(f[0]));
  $('calib-exporter').addEventListener('click', exporterCalib);
  $('calib-effacer').addEventListener('click', () => {
    etat.calib = null;
    ecrire(CLE_CALIB, null);
    majCalib();
    majBoutons();
  });
  $('photos-effacer').addEventListener('click', viderPhotos);
  $('calib-direct').addEventListener('click', () => ouvrirDirect('calib'));
  $('photos-direct').addEventListener('click', () => ouvrirDirect('scan'));
  $('direct-terminer').addEventListener('click', () => { if (direct) { direct.action = 'terminer'; direct.actif = false; } });
  $('direct-arreter').addEventListener('click', () => { if (direct) { direct.action = 'arreter'; direct.actif = false; } });
  $('recon-lancer').addEventListener('click', reconstruire);
  AXES.forEach((a) => $('d-' + a).addEventListener('input', () => saisie(a)));
  $('d-appliquer').addEventListener('click', appliquerCotes);
  $('d-reinit').addEventListener('click', reinitialiser);
  $('d-aligner').addEventListener('click', aligner);
  $('d-couper-btn').addEventListener('click', couper);
  $('stl-telecharger').addEventListener('click', exporterSTL);
  message(etat.calib ? 'Prêt. Lancer la prise en direct ou charger des photos de l\'objet.' : 'Commencer par imprimer la planche et calibrer le téléphone.');
}

demarrer();
