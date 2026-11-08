"""Exporta el Random Forest a C (ESP32-S3): árboles aplanados + calibración por máquina.
1) barrido de tamaño (árboles x profundidad) para elegir el RF más chico sin perder desempeño
2) exporta el elegido a vib_rf.h y valida que el C reproduce a scikit-learn
Uso: python export_rf.py -> vib_rf.h, vectores_prueba_rf.csv, resultados_export_rf.txt"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import sys
sys.path.append("../1st version")
from detector_vibracion import build, calibrate, NAMES, N_CAL

def mk(n, d): return RandomForestClassifier(n, max_depth=d, random_state=0, n_jobs=-1)
def nodes(rf): return sum(e.tree_.node_count for e in rf.estimators_)
def leaves(rf): return sum(int((e.tree_.children_left == -1).sum()) for e in rf.estimators_)
def size_bytes(rf): return nodes(rf)*9 + leaves(rf)*3 + len(rf.estimators_)*2   # feat(1)+thr(4)+izq(2)+der(2) | 3 prob por hoja | raíz

def fl(x):
    t = f"{x:.9g}"; return t + ("" if any(c in t for c in ".en") else ".0") + "f"
def arr(name, v, typ, f=str, per=12):
    return f"static const {typ} {name}[{len(v)}] = {{\n  " + ",".join(f(x) + ("\n  " if (i+1) % per == 0 else "") for i, x in enumerate(v)) + "\n};\n"

if __name__ == "__main__":
    X, y, g, rpm, cal = build(); Xn = calibrate(X, g, cal); L = []
    def log(s): print(s); L.append(s)
    log("== Barrido de tamaño del RF (5 folds por máquina, features calibrados) ==")
    log(f"{'árboles':>8s}{'prof.':>7s}{'F1 macro':>10s}{'nodos':>9s}{'bytes aprox.':>14s}")
    res = []
    for n, d in [(150, 12), (50, 10), (30, 8), (20, 8), (10, 8), (30, 6)]:
        f1s = [f1_score(y[te], mk(n, d).fit(Xn[tr], y[tr]).predict(Xn[te]), average="macro") for tr, te in GroupKFold(5).split(Xn, y, g)]
        full = mk(n, d).fit(Xn, y); res.append((n, d, np.mean(f1s), size_bytes(full)))
        log(f"{n:8d}{d:7d}{np.mean(f1s):10.3f}{nodes(full):9d}{size_bytes(full):14d}")
    base = res[0][2]; ok = [r for r in res if r[2] >= base - 0.005]; n, d, f1c, sz = min(ok, key=lambda r: r[3])
    log(f"\nElegido: {n} árboles, profundidad {d} (F1 {f1c:.3f} vs {base:.3f} del RF completo; {sz/1024:.1f} KB en flash)")
    rf = mk(n, d).fit(Xn, y); T = rf.estimators_
    feat, thr, lft, rgt, root, leaf = [], [], [], [], [], []
    for e in T:
        t = e.tree_; off = len(feat); root.append(off); lid = {}
        for i in range(t.node_count):
            if t.children_left[i] == -1:
                v = t.value[i, 0]; p = v/v.sum(); lid[i] = len(leaf)//3; leaf += [int(round(q*255)) for q in p]
        for i in range(t.node_count):
            if t.children_left[i] == -1: feat.append(-1); thr.append(0.0); lft.append(lid[i]); rgt.append(0)
            else: feat.append(int(t.feature[i])); thr.append(float(t.threshold[i])); lft.append(off + int(t.children_left[i])); rgt.append(off + int(t.children_right[i]))
    assert len(feat) < 65536
    nf = X.shape[1]
    h = f"""/* Random Forest de detección de vibración (AI-SED 2026) - generado por export_rf.py
 * Entrada: {nf} features por ventana, en este orden:
 *   {", ".join(NAMES)}
 * Flujo: calibración con las primeras {N_CAL} ventanas SANAS de la máquina -> normalizar -> {n} árboles -> voto por probabilidad.
 * Clases: 0 = normal, 1 = perturbación externa, 2 = anomalía del mecanismo.
 * Uso:  vib_calib_t c; vib_calib_init(&c);  for(i<{N_CAL}) vib_calib_add(&c, feat);   // ventanas sanas
 *       float p[3]; int clase = vib_predict(&c, feat, p);   // luego, cada ventana
 */
#ifndef VIB_RF_H
#define VIB_RF_H
#include <stdint.h>
#include <math.h>
#define VIB_N_FEAT {nf}
#define VIB_NCLS 3
#define VIB_N_CAL {N_CAL}
#define VIB_N_TREES {len(T)}
""" + arr("VIB_ROOT", root, "uint16_t") + arr("VIB_FEAT", feat, "int8_t") + arr("VIB_THR", thr, "float", fl, 8) + \
        arr("VIB_LEFT", lft, "uint16_t") + arr("VIB_RIGHT", rgt, "uint16_t") + arr("VIB_LEAF", leaf, "uint8_t", str, 15) + """
typedef struct { int n, ready; float mean[VIB_N_FEAT], m2[VIB_N_FEAT], mu[VIB_N_FEAT], sd[VIB_N_FEAT]; } vib_calib_t;
static inline void vib_calib_init(vib_calib_t *c) { c->n = 0; c->ready = 0; }
/* Acumula una ventana sana (Welford). Al llegar a VIB_N_CAL queda calibrada. */
static inline void vib_calib_add(vib_calib_t *c, const float *f) {
  if (c->ready) return;
  if (c->n == 0) for (int i = 0; i < VIB_N_FEAT; i++) { c->mean[i] = 0; c->m2[i] = 0; }
  c->n++;
  for (int i = 0; i < VIB_N_FEAT; i++) { float d = f[i] - c->mean[i]; c->mean[i] += d / c->n; c->m2[i] += d * (f[i] - c->mean[i]); }
  if (c->n >= VIB_N_CAL) { for (int i = 0; i < VIB_N_FEAT; i++) { c->mu[i] = c->mean[i]; c->sd[i] = sqrtf(c->m2[i] / c->n) + 1e-9f; } c->ready = 1; }
}
/* Devuelve la clase (0,1,2) y, si probs != NULL, las probabilidades. Sin calibración lista devuelve -1. */
static inline int vib_predict(const vib_calib_t *c, const float *feat, float *probs) {
  if (!c->ready) return -1;
  float x[VIB_N_FEAT]; uint32_t acc[VIB_NCLS] = {0, 0, 0};
  for (int i = 0; i < VIB_N_FEAT; i++) x[i] = (feat[i] - c->mu[i]) / c->sd[i];
  for (int t = 0; t < VIB_N_TREES; t++) {
    uint16_t n = VIB_ROOT[t];
    while (VIB_FEAT[n] >= 0) n = (x[VIB_FEAT[n]] <= VIB_THR[n]) ? VIB_LEFT[n] : VIB_RIGHT[n];
    const uint8_t *lp = &VIB_LEAF[VIB_LEFT[n] * VIB_NCLS];
    for (int k = 0; k < VIB_NCLS; k++) acc[k] += lp[k];
  }
  int best = 0; for (int k = 1; k < VIB_NCLS; k++) if (acc[k] > acc[best]) best = k;
  if (probs) for (int k = 0; k < VIB_NCLS; k++) probs[k] = acc[k] / (255.0f * VIB_N_TREES);
  return best;
}
#endif
"""
    open("vib_rf.h", "w").write(h)
    tr, te = g < 32, g >= 32
    ps = mk(n, d).fit(Xn[tr], y[tr]); log(f"F1 macro en 8 máquinas no vistas (RF elegido, sklearn): {f1_score(y[te], ps.predict(Xn[te]), average='macro'):.3f}")
    with open("vectores_prueba_rf.csv", "w") as fh:                   # 6 máquinas; predicción de sklearn con la calibración idéntica
        for s in np.unique(g)[:6]:
            m = np.where(g == s)[0]; b = X[m][cal[m]]; mu, sd = b.mean(0), b.std(0) + 1e-9
            pr = rf.predict((X[m] - mu)/sd)
            for k, i in enumerate(m): fh.write(",".join([str(s), str(int(cal[i]))] + [f"{v:.9g}" for v in X[i]] + [str(pr[k])]) + "\n")
    open("resultados_export_rf.txt", "w").write("\n".join(L))
