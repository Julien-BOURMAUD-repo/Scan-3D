'use strict';
/* Sculpture de silhouettes (visual hull), maillage par surface nets et lissage de Taubin. */

let volume = null; // dernier volume reconstruit, garde pour remailler (coupe en hauteur)

function progres(etape, valeur) {
  self.postMessage({ type: 'progres', etape, valeur });
}

function sculpter(m) {
  const { vues, mw, mh, K, dist, params, centre } = m;
  const vx = params.voxel, tol = params.tolerance;
  const nx = Math.max(4, Math.floor(params.emprise / vx));
  const ny = nx;
  const nz = Math.max(4, Math.floor(params.hauteur / vx));
  const x0 = centre[0] - (nx * vx) / 2 + vx / 2;
  const y0 = centre[1] - (ny * vx) / 2 + vx / 2;

  // Cote de la planche ou se trouvent les cameras (= dessus de l'objet)
  let sz = 0;
  for (const v of vues) {
    const R = v.R, t = v.t;
    sz += -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]);
  }
  const signe = sz > 0 ? 1 : -1;

  const nv = vues.length;
  const P = new Float64Array(nv * 12);
  vues.forEach((v, i) => { P.set(v.R, i * 12); P.set(v.t, i * 12 + 9); });
  const masques = vues.map((v) => v.masque);
  const fx = K[0], sk = K[1], cx = K[2], fy = K[4], cy = K[5];
  const [k1, k2, p1, p2, k3] = [dist[0] || 0, dist[1] || 0, dist[2] || 0, dist[3] || 0, dist[4] || 0];
  let rmax2 = 0;
  for (const [u, v] of [[0, 0], [mw, 0], [0, mh], [mw, mh]]) {
    rmax2 = Math.max(rmax2, ((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2);
  }
  rmax2 *= 1.5;
  const umax = mw - 0.5, vmax = mh - 0.5;

  const garde = new Uint8Array(nx * ny * nz);
  for (let k = 0; k < nz; k++) {
    const Z = signe * (k + 0.5) * vx;
    for (let j = 0; j < ny; j++) {
      const Y = y0 + j * vx;
      for (let i = 0; i < nx; i++) {
        const X = x0 + i * vx;
        let dehors = 0, vu = 0;
        for (let v = 0; v < nv; v++) {
          const o = v * 12;
          const zc = P[o + 6] * X + P[o + 7] * Y + P[o + 8] * Z + P[o + 11];
          if (zc <= 1) continue;
          const xn = (P[o] * X + P[o + 1] * Y + P[o + 2] * Z + P[o + 9]) / zc;
          const yn = (P[o + 3] * X + P[o + 4] * Y + P[o + 5] * Z + P[o + 10]) / zc;
          const r2 = xn * xn + yn * yn;
          if (r2 > rmax2) continue;
          const rad = 1 + r2 * (k1 + r2 * (k2 + r2 * k3));
          const xd = xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn);
          const yd = yn * rad + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn;
          const u = fx * xd + sk * yd + cx;
          const w = fy * yd + cy;
          if (u < -0.5 || w < -0.5 || u >= umax || w >= vmax) continue;
          vu++;
          if (masques[v][((w + 0.5) | 0) * mw + ((u + 0.5) | 0)] === 0 && ++dehors > tol) break;
        }
        if (vu >= 3 && dehors <= tol) garde[(k * ny + j) * nx + i] = 1;
      }
    }
    if ((k & 3) === 0) progres('sculpture', k / nz);
  }
  return { garde, nx, ny, nz, x0, y0, signe };
}

/* Recadre, bouche les cavites fermees et garde la plus grande composante connexe. */
function nettoyer(s) {
  const { garde, nx, ny, nz } = s;
  let i0 = nx, i1 = -1, j0 = ny, j1 = -1, k0 = nz, k1 = -1;
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    if (garde[(k * ny + j) * nx + i]) {
      if (i < i0) i0 = i; if (i > i1) i1 = i;
      if (j < j0) j0 = j; if (j > j1) j1 = j;
      if (k < k0) k0 = k; if (k > k1) k1 = k;
    }
  }
  if (i1 < 0) throw new Error('Volume vide : vérifier les masques (vignettes) ou augmenter la tolérance.');
  i0 = Math.max(0, i0 - 1); j0 = Math.max(0, j0 - 1); k0 = Math.max(0, k0 - 1);
  i1 = Math.min(nx - 1, i1 + 1); j1 = Math.min(ny - 1, j1 + 1); k1 = Math.min(nz - 1, k1 + 1);
  const a = i1 - i0 + 1, b = j1 - j0 + 1, c = k1 - k0 + 1;
  const n = a * b * c;
  const vol = new Uint8Array(n);
  for (let k = 0; k < c; k++) for (let j = 0; j < b; j++) {
    const src = ((k + k0) * ny + (j + j0)) * nx + i0;
    vol.set(garde.subarray(src, src + a), (k * b + j) * a);
  }

  const file = new Int32Array(n);
  const voisins = (idx, cb) => {
    const i = idx % a, j = ((idx / a) | 0) % b, k = (idx / (a * b)) | 0;
    if (i > 0) cb(idx - 1); if (i < a - 1) cb(idx + 1);
    if (j > 0) cb(idx - a); if (j < b - 1) cb(idx + a);
    if (k > 0) cb(idx - a * b); if (k < c - 1) cb(idx + a * b);
  };

  // Exterieur : remplissage depuis les bords sur les voxels vides
  const ext = new Uint8Array(n);
  let tete = 0, queue = 0;
  const pousserExt = (idx) => { if (!vol[idx] && !ext[idx]) { ext[idx] = 1; file[queue++] = idx; } };
  for (let k = 0; k < c; k++) for (let j = 0; j < b; j++) for (let i = 0; i < a; i++) {
    if (i === 0 || j === 0 || k === 0 || i === a - 1 || j === b - 1 || k === c - 1) pousserExt((k * b + j) * a + i);
  }
  while (tete < queue) voisins(file[tete++], pousserExt);
  for (let idx = 0; idx < n; idx++) if (!ext[idx]) vol[idx] = 1;

  // Plus grande composante connexe (6-connexite)
  const etiq = new Int32Array(n);
  let meilleure = 0, tailleMax = 0, courante = 0, taille = 0;
  const pousserComp = (idx) => { if (vol[idx] && !etiq[idx]) { etiq[idx] = courante; file[queue++] = idx; taille++; } };
  for (let idx = 0; idx < n; idx++) {
    if (!vol[idx] || etiq[idx]) continue;
    courante++; taille = 0; tete = 0; queue = 0;
    pousserComp(idx);
    while (tete < queue) voisins(file[tete++], pousserComp);
    if (taille > tailleMax) { tailleMax = taille; meilleure = courante; }
  }
  let bord = false;
  for (let k = 0; k < c; k++) for (let j = 0; j < b; j++) for (let i = 0; i < a; i++) {
    const idx = (k * b + j) * a + i;
    vol[idx] = etiq[idx] === meilleure ? 1 : 0;
    if (vol[idx] && (i + i0 === 0 || i + i0 === nx - 1 || j + j0 === 0 || j + j0 === ny - 1 || k + k0 === nz - 1)) bord = true;
  }
  return { vol, a, b, c, i0, j0, k0, bord, voxels: tailleMax };
}

function flouGaussien(f, A, B, C, sigma) {
  const r = 2;
  const w = [];
  let s = 0;
  for (let d = -r; d <= r; d++) { const v = Math.exp(-(d * d) / (2 * sigma * sigma)); w.push(v); s += v; }
  for (let i = 0; i < w.length; i++) w[i] /= s;
  const tmp = new Float32Array(f.length);
  const pas = [1, A, A * B];
  const tailles = [A, B, C];
  let src = f, dst = tmp;
  for (let axe = 0; axe < 3; axe++) {
    const st = pas[axe], L = tailles[axe];
    for (let k = 0; k < C; k++) for (let j = 0; j < B; j++) for (let i = 0; i < A; i++) {
      const idx = (k * B + j) * A + i;
      const pos = axe === 0 ? i : axe === 1 ? j : k;
      let v = 0;
      for (let d = -r; d <= r; d++) {
        const q = pos + d;
        if (q >= 0 && q < L) v += w[d + r] * src[idx + d * st];
      }
      dst[idx] = v;
    }
    const t = src; src = dst; dst = t;
  }
  if (src !== f) f.set(src);
}

/* Surface nets : un sommet par cellule traversee, un quadrilatere par arete traversee. */
function surfaceNets(f, A, B, C, iso) {
  const ca = A - 1, cb = B - 1, cc = C - 1;
  const cellule = new Int32Array(ca * cb * cc).fill(-1);
  let pos = new Float32Array(1 << 16), nv = 0;
  let tri = new Uint32Array(1 << 17), nt = 0;
  const coins = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
  const aretes = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
  const val = new Float32Array(8);
  for (let k = 0; k < cc; k++) for (let j = 0; j < cb; j++) for (let i = 0; i < ca; i++) {
    let masque = 0;
    for (let q = 0; q < 8; q++) {
      const c = coins[q];
      val[q] = f[((k + c[2]) * B + (j + c[1])) * A + (i + c[0])];
      if (val[q] > iso) masque |= 1 << q;
    }
    if (masque === 0 || masque === 255) continue;
    let sx = 0, sy = 0, sz = 0, m = 0;
    for (const [p, q] of aretes) {
      const dp = val[p] > iso, dq = val[q] > iso;
      if (dp === dq) continue;
      const t = (iso - val[p]) / (val[q] - val[p]);
      const cp = coins[p], cq = coins[q];
      sx += cp[0] + t * (cq[0] - cp[0]);
      sy += cp[1] + t * (cq[1] - cp[1]);
      sz += cp[2] + t * (cq[2] - cp[2]);
      m++;
    }
    if (3 * nv + 3 > pos.length) { const n2 = new Float32Array(pos.length * 2); n2.set(pos); pos = n2; }
    pos[3 * nv] = i + sx / m; pos[3 * nv + 1] = j + sy / m; pos[3 * nv + 2] = k + sz / m;
    cellule[(k * cb + j) * ca + i] = nv++;
  }
  const cel = (i, j, k) => cellule[(k * cb + j) * ca + i];
  const quad = (a, b, c, d, inverse) => {
    if (a < 0 || b < 0 || c < 0 || d < 0) return;
    if (nt * 3 + 6 > tri.length) { const n2 = new Uint32Array(tri.length * 2); n2.set(tri); tri = n2; }
    if (!inverse) { tri.set([a, b, c, a, c, d], nt * 3); } else { tri.set([a, c, b, a, d, c], nt * 3); }
    nt += 2;
  };
  for (let k = 1; k < C - 1; k++) for (let j = 1; j < B - 1; j++) for (let i = 1; i < A - 1; i++) {
    const idx = (k * B + j) * A + i;
    const v0 = f[idx] > iso;
    if ((f[idx + 1] > iso) !== v0) quad(cel(i, j - 1, k - 1), cel(i, j, k - 1), cel(i, j, k), cel(i, j - 1, k), !v0);
    if ((f[idx + A] > iso) !== v0) quad(cel(i - 1, j, k - 1), cel(i - 1, j, k), cel(i, j, k), cel(i, j, k - 1), !v0);
    if ((f[idx + A * B] > iso) !== v0) quad(cel(i - 1, j - 1, k), cel(i, j - 1, k), cel(i, j, k), cel(i - 1, j, k), !v0);
  }
  return { pos: pos.slice(0, 3 * nv), tri: tri.slice(0, 3 * nt) };
}

function taubin(pos, tri, iterations) {
  if (!iterations) return;
  const nv = pos.length / 3;
  const deg = new Uint32Array(nv + 1);
  for (let t = 0; t < tri.length; t += 3) {
    deg[tri[t]] += 2; deg[tri[t + 1]] += 2; deg[tri[t + 2]] += 2;
  }
  const debut = new Uint32Array(nv + 1);
  for (let v = 0; v < nv; v++) debut[v + 1] = debut[v] + deg[v];
  const adj = new Uint32Array(debut[nv]);
  const rempli = debut.slice(0, nv);
  const lier = (a, b) => { adj[rempli[a]++] = b; adj[rempli[b]++] = a; };
  for (let t = 0; t < tri.length; t += 3) {
    lier(tri[t], tri[t + 1]); lier(tri[t + 1], tri[t + 2]); lier(tri[t + 2], tri[t]);
  }
  const tmp = new Float32Array(pos.length);
  for (let it = 0; it < iterations; it++) {
    for (const lambda of [0.5, -0.53]) {
      for (let v = 0; v < nv; v++) {
        const d0 = debut[v], d1 = debut[v + 1];
        if (d1 === d0) { tmp[3 * v] = pos[3 * v]; tmp[3 * v + 1] = pos[3 * v + 1]; tmp[3 * v + 2] = pos[3 * v + 2]; continue; }
        let sx = 0, sy = 0, sz = 0;
        for (let q = d0; q < d1; q++) { const n = adj[q]; sx += pos[3 * n]; sy += pos[3 * n + 1]; sz += pos[3 * n + 2]; }
        const inv = 1 / (d1 - d0);
        tmp[3 * v] = pos[3 * v] + lambda * (sx * inv - pos[3 * v]);
        tmp[3 * v + 1] = pos[3 * v + 1] + lambda * (sy * inv - pos[3 * v + 1]);
        tmp[3 * v + 2] = pos[3 * v + 2] + lambda * (sz * inv - pos[3 * v + 2]);
      }
      pos.set(tmp);
    }
  }
}

function mailler(couperZ, lissage) {
  if (!volume) throw new Error('Aucun volume reconstruit.');
  const { vol, a, b, c, i0, j0, k0, vx, x0, y0, centre, signe } = volume;
  const pad = 2;
  const A = a + 2 * pad, B = b + 2 * pad, C = c + 2 * pad;
  const f = new Float32Array(A * B * C);
  for (let k = 0; k < c; k++) {
    const basVoxel = (k0 + k) * vx;
    if (couperZ > 0 && basVoxel >= couperZ) continue;
    for (let j = 0; j < b; j++) for (let i = 0; i < a; i++) {
      if (vol[(k * b + j) * a + i]) f[((k + pad) * B + (j + pad)) * A + (i + pad)] = 1;
    }
  }
  progres('maillage', 0.2);
  flouGaussien(f, A, B, C, 0.8);
  progres('maillage', 0.4);
  const { pos, tri } = surfaceNets(f, A, B, C, 0.5);
  if (tri.length === 0) throw new Error('Maillage vide.');
  const n = pos.length / 3;
  for (let v = 0; v < n; v++) {
    const x = x0 + (pos[3 * v] - pad + i0) * vx;
    const y = y0 + (pos[3 * v + 1] - pad + j0) * vx;
    pos[3 * v] = x - centre[0];
    pos[3 * v + 1] = signe * (y - centre[1]);
    pos[3 * v + 2] = (pos[3 * v + 2] - pad + k0 + 0.5) * vx;
  }
  // Orientation des faces vers l'exterieur (volume signe positif)
  let vs = 0;
  for (let t = 0; t < tri.length; t += 3) {
    const p = 3 * tri[t], q = 3 * tri[t + 1], r = 3 * tri[t + 2];
    vs += pos[p] * (pos[q + 1] * pos[r + 2] - pos[q + 2] * pos[r + 1])
        - pos[p + 1] * (pos[q] * pos[r + 2] - pos[q + 2] * pos[r])
        + pos[p + 2] * (pos[q] * pos[r + 1] - pos[q + 1] * pos[r]);
  }
  if (vs < 0) for (let t = 0; t < tri.length; t += 3) { const s = tri[t + 1]; tri[t + 1] = tri[t + 2]; tri[t + 2] = s; }
  progres('maillage', 0.7);
  taubin(pos, tri, lissage);
  if (couperZ > 0) for (let v = 0; v < n; v++) if (pos[3 * v + 2] > couperZ) pos[3 * v + 2] = couperZ;
  progres('maillage', 1);
  return { positions: pos, indices: tri };
}

self.onmessage = (e) => {
  const m = e.data;
  try {
    if (m.type === 'reconstruire') {
      const s = sculpter(m);
      progres('nettoyage', 0);
      const nt = nettoyer(s);
      volume = { ...nt, vx: m.params.voxel, x0: s.x0, y0: s.y0, centre: m.centre, signe: s.signe };
      const r = mailler(m.couperZ || 0, m.params.lissage);
      self.postMessage({ id: m.id, ...r, bord: nt.bord, voxels: nt.voxels }, [r.positions.buffer, r.indices.buffer]);
    } else if (m.type === 'remailler') {
      const r = mailler(m.couperZ || 0, m.lissage);
      self.postMessage({ id: m.id, ...r }, [r.positions.buffer, r.indices.buffer]);
    } else {
      throw new Error('Message inconnu : ' + m.type);
    }
  } catch (err) {
    self.postMessage({ id: m.id, erreur: err && err.message ? err.message : String(err) });
  }
};
