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
    el.textContent = `Calibré : ${c.images} photos, erreur ${fmt(c.rms, 3)} px, images ${c.largeur} x ${c.hauteur}.`;
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
    if (ok < 8) throw new Error(`Seulement ${ok} photos exploitables (8 minimum, 15 à 25 conseillées).`);
    message('Calcul de la calibration...');
    const r = await vision.appel('calibrer', { w: taille[0], h: taille[1] });
    etat.calib = {
      largeur: taille[0], hauteur: taille[1], K: r.K, dist: r.dist, rms: r.rms,
      images: ok, date: new Date().toISOString(),
    };
    ecrire(CLE_CALIB, etat.calib);
    let txt = `Calibration terminée : erreur ${fmt(r.rms, 3)} px sur ${ok} photos.`;
    if (r.rms > 1) txt += ' Erreur élevée : vérifier la netteté et la planéité de la feuille.';
    if (rejets.length) txt += ` Ignorées : ${rejets.join(', ')}.`;
    message(txt, r.rms > 1 ? 'alerte' : 'ok');
    if (etat.vues.length) {
      message(txt + ' Les photos de l\'objet déjà chargées doivent être rechargées.', 'alerte');
      viderPhotos();
    }
  } catch (err) {
    message('Calibration : ' + err.message, 'erreur');
  } finally {
    barre.hidden = true;
    majCalib();
    occuper(false);
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
  $('photos-effacer').disabled = etat.occupe || !etat.vues.length;
}

function viderPhotos() {
  etat.vues = [];
  $('photos-grille').replaceChildren();
  majResume();
}

async function ajouterPhotos(fichiers) {
  if (!etat.calib) { message('Faire la calibration du téléphone avant de charger les photos de l\'objet.', 'alerte'); return; }
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
  $('recon-lancer').addEventListener('click', reconstruire);
  AXES.forEach((a) => $('d-' + a).addEventListener('input', () => saisie(a)));
  $('d-appliquer').addEventListener('click', appliquerCotes);
  $('d-reinit').addEventListener('click', reinitialiser);
  $('d-aligner').addEventListener('click', aligner);
  $('d-couper-btn').addEventListener('click', couper);
  $('stl-telecharger').addEventListener('click', exporterSTL);
  message(etat.calib ? 'Prêt. Charger les photos de l\'objet.' : 'Commencer par imprimer la planche et calibrer le téléphone.');
}

demarrer();
