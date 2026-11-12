'use strict';
/* Detourage de l'objet (modele RMBG-1.4 via ONNX Runtime Web, calcul local). */

const ORT_BASE = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
const CACHE = 'scan3d-modeles';

let session = null;
let urlModele = null;

function progres(etape, valeur, texte) {
  self.postMessage({ type: 'progres', etape, valeur, texte });
}

async function telecharger(url) {
  let cache = null;
  try { cache = await caches.open(CACHE); } catch (e) { cache = null; }
  if (cache) {
    const r = await cache.match(url);
    if (r) return new Uint8Array(await r.arrayBuffer());
  }
  const rep = await fetch(url);
  if (!rep.ok) throw new Error(`Téléchargement du modèle impossible (HTTP ${rep.status})`);
  const total = Number(rep.headers.get('content-length')) || 0;
  let octets;
  if (rep.body && rep.body.getReader) {
    const lecteur = rep.body.getReader();
    const morceaux = [];
    let recu = 0;
    for (;;) {
      const { done, value } = await lecteur.read();
      if (done) break;
      morceaux.push(value);
      recu += value.length;
      progres('modele', total ? recu / total : 0, `${(recu / 1e6).toFixed(0)} Mo`);
    }
    octets = new Uint8Array(recu);
    let pos = 0;
    for (const m of morceaux) { octets.set(m, pos); pos += m.length; }
  } else {
    octets = new Uint8Array(await rep.arrayBuffer());
  }
  if (cache) {
    try {
      await cache.put(url, new Response(octets, { headers: { 'content-type': 'application/octet-stream' } }));
    } catch (e) { /* stockage plein ou indisponible : le modele sera retelecharge */ }
  }
  return octets;
}

async function init(url) {
  if (session && urlModele === url) return;
  if (!self.ort) importScripts(ORT_BASE + 'ort.wasm.min.js');
  const ort = self.ort;
  ort.env.wasm.wasmPaths = ORT_BASE;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  const octets = await telecharger(url);
  progres('modele', 1, 'Initialisation');
  session = await ort.InferenceSession.create(octets, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  urlModele = url;
}

/* rgba : image S x S (etiree), sortie : masque mw x mh, 1 = objet. */
async function segmenter(rgba, S, mw, mh) {
  const ort = self.ort;
  const n = S * S;
  const entree = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    entree[i] = rgba[4 * i] / 255 - 0.5;
    entree[n + i] = rgba[4 * i + 1] / 255 - 0.5;
    entree[2 * n + i] = rgba[4 * i + 2] / 255 - 0.5;
  }
  const tenseur = new ort.Tensor('float32', entree, [1, 3, S, S]);
  const sortie = await session.run({ [session.inputNames[0]]: tenseur });
  const o = sortie[session.outputNames[0]].data;
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < n; i++) { if (o[i] < mn) mn = o[i]; if (o[i] > mx) mx = o[i]; }
  const masque = new Uint8Array(mw * mh);
  if (mx - mn < 0.2) return masque; // aucun objet net : masque vide, la photo sera rejetee
  const seuil = mn + 0.5 * (mx - mn);
  const sx = S / mw, sy = S / mh;
  for (let y = 0; y < mh; y++) {
    const fy = Math.min(S - 1, Math.max(0, (y + 0.5) * sy - 0.5));
    const y0 = Math.floor(fy), y1 = Math.min(S - 1, y0 + 1), ay = fy - y0;
    for (let x = 0; x < mw; x++) {
      const fx = Math.min(S - 1, Math.max(0, (x + 0.5) * sx - 0.5));
      const x0 = Math.floor(fx), x1 = Math.min(S - 1, x0 + 1), ax = fx - x0;
      const v = (o[y0 * S + x0] * (1 - ax) + o[y0 * S + x1] * ax) * (1 - ay)
              + (o[y1 * S + x0] * (1 - ax) + o[y1 * S + x1] * ax) * ay;
      masque[y * mw + x] = v > seuil ? 1 : 0;
    }
  }
  tenseur.dispose && tenseur.dispose();
  return masque;
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      await init(m.url);
      self.postMessage({ id: m.id });
    } else if (m.type === 'segmenter') {
      const masque = await segmenter(m.rgba, m.S, m.mw, m.mh);
      self.postMessage({ id: m.id, masque }, [masque.buffer]);
    } else {
      throw new Error('Message inconnu : ' + m.type);
    }
  } catch (err) {
    self.postMessage({ id: m.id, erreur: err && err.message ? err.message : String(err) });
  }
};
