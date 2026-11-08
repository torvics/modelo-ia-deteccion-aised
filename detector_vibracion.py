"""
Detector de vibración (AI-SED 2026): generador sintético + modelo ligero.
Cada parámetro está marcado [FUENTE] (verificado) o [SUPUESTO] (se aleatoriza en un rango amplio
y se declara como tal; el modelo no depende de un valor exacto).
Clases: 0 = normal | 1 = perturbación externa | 2 = anomalía del mecanismo (impactos periódicos)
Uso:  python detector_vibracion.py        -> resultados en out_det/
"""
# Importa NumPy para cálculos numéricos y csv para escribir el conjunto de datos.
import numpy as np, csv
# Importa Path para construir rutas de archivos de forma portable.
from pathlib import Path
# Importa filtros y transformadas usadas para generar y analizar vibraciones.
from scipy.signal import lfilter, butter, sosfilt, hilbert
# Importa la curtosis, una medida de la presencia de valores extremos.
from scipy.stats import kurtosis
# Importa el clasificador basado en muchos árboles de decisión.
from sklearn.ensemble import RandomForestClassifier
# Importa el perceptrón multicapa para comparar otro modelo.
from sklearn.neural_network import MLPClassifier
# Importa el estandarizador de variables.
from sklearn.preprocessing import StandardScaler
# Importa la función que encadena transformaciones y modelo.
from sklearn.pipeline import make_pipeline
# Importa la validación cruzada que separa grupos, aquí máquinas simuladas.
from sklearn.model_selection import GroupKFold
# Importa las métricas usadas para evaluar las predicciones.
from sklearn.metrics import f1_score, confusion_matrix
# Importa Matplotlib, selecciona un backend sin interfaz y carga el módulo de gráficas.
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# ---------------- Parámetros ----------------
# Define la frecuencia de muestreo y el tamaño de ventana para la señal rápida.
FS_H, N_H = 40_000, 2048          # [FUENTE] datasheet IIS3DWB10IS: ODR 40 kHz; ventana 51.2 ms
# Define la frecuencia de muestreo y el tamaño de ventana para la señal lenta.
FS_L, N_L = 500, 1000             # flujo decimado para la banda baja (ventana de 2 s)
# Define la densidad de ruido nominal del sensor.
ND = 35e-6                        # [FUENTE] datasheet: ruido 35 ug/rtHz (X,Y)
# Define el rango de velocidades de giro en revoluciones por minuto.
RPM = (3000, 40000)               # [FUENTE] ficha DISCO DAD3220: husillo 3,000-40,000 rpm
# Define el rango de amplitudes de vibración ambiental.
AMBIENT = (30e-6, 120e-6)         # [FUENTE] VC-C = 12.5 um/s ~ 80 ug a 10 Hz (Gordon 1991); rango alrededor
# Define el rango de amplitudes para eventos externos.
EXT_EVENT = (0.3e-3, 5e-3)        # [FUENTE parcial] entre ~VC-C y criterio ISO taller (800 um/s ~ 5 mg a 10 Hz)
# Define la probabilidad de que una ventana corresponda a una etapa de corte.
P_CUT = 0.7                       # [SUPUESTO] fracción del tiempo cortando vs indexando
# Define cuántas ventanas normales se usan para calibrar cada máquina.
N_CAL = 8                         # ventanas sanas de calibración por máquina (supuesto de diseño)
# Aleatorizados por máquina [SUPUESTO] salvo donde se indica: amplitud del giro 3-50 mg, modos estructurales
# 30-150 Hz (base) y 300-800 Hz (cabezal) [valores de la investigación, referencias por verificar],
# resonancia local 2-8 kHz, Q 10-40, orden de falla k = 2.5-8 x rpm (incluye BPFO 5.45 y BPFI 7.55 del SKF 7006).

def resonator(x, f, Q, fs):
    # Calcula los coeficientes de un resonador de segundo orden.
    th, r = 2*np.pi*f/fs, np.exp(-np.pi*f/(Q*fs))
    # Filtra la señal de entrada para simular una resonancia mecánica.
    return lfilter([1.0], [1, -2*r*np.cos(th), r*r], x)

# Ajusta la desviación estándar de una señal al valor RMS solicitado.
def scale(x, rms): return x * rms / (x.std() + 1e-12)

# Genera un valor aleatorio con distribución uniforme en escala logarítmica.
def logU(rng, a, b): return float(np.exp(rng.uniform(np.log(a), np.log(b))))

# Genera ruido blanco limitado a una banda de frecuencias y lo escala.
def bandnoise(rng, n, fs, f1, f2, rms):
    # Genera el ruido y lo transforma al dominio de la frecuencia.
    X = np.fft.rfft(rng.normal(0, 1, n)); f = np.fft.rfftfreq(n, 1/fs); X[(f < f1) | (f > f2)] = 0
    # Regresa la señal al dominio temporal con la amplitud indicada.
    return scale(np.fft.irfft(X, n), rms)

# Crea los parámetros aleatorios de una máquina simulada.
def new_session(rng):                      # una "máquina" distinta por sesión
    # Devuelve las frecuencias, amplitudes y factores propios de la máquina.
    return dict(f_r=rng.uniform(*RPM)/60, a_spin=logU(rng, 3e-3, 5e-2), a_struct=logU(rng, 1e-3, 1e-2),
                a_cut=logU(rng, 1e-3, 8e-3), f_head=rng.uniform(300, 800), f_base=rng.uniform(30, 150),
                Q=rng.uniform(10, 40), f_hf=rng.uniform(2000, 8000), Q_hf=rng.uniform(10, 30),
                a_base=logU(rng, 2e-4, 2e-3), ambient=rng.uniform(*AMBIENT), k=rng.uniform(2.5, 8))

# Genera una ventana de vibración para una clase determinada.
def window(cls, s, rng, nm=1.0):
    # Crea los ejes temporales de alta y baja frecuencia.
    th, tl = np.arange(N_H)/FS_H, np.arange(N_L)/FS_L
    # Decide aleatoriamente si la máquina está cortando material.
    cutting = rng.random() < P_CUT
    # --- alta tasa (kHz): giro + modo del cabezal + ruido de corte
    # Suma tres armónicos de la velocidad de giro con fases aleatorias.
    xh = sum(s["a_spin"]/k*np.sin(2*np.pi*k*s["f_r"]*th + rng.uniform(0, 6.28)) for k in (1, 2, 3))
    # Añade la resonancia estructural del cabezal.
    xh += scale(resonator(rng.normal(0, 1, N_H), s["f_head"], s["Q"], FS_H), s["a_struct"])
    # Añade ruido de corte únicamente cuando la máquina está cortando.
    if cutting: xh += rng.normal(0, s["a_cut"], N_H)
    # --- baja tasa (1-150 Hz): modo de la base + piso ambiental + transitorio de indexación
    # Genera la resonancia de la base en la señal de baja frecuencia.
    xl = scale(resonator(rng.normal(0, 1, N_L), s["f_base"], s["Q"], FS_L), s["a_base"])
    # Añade ruido ambiental limitado a la banda de baja frecuencia.
    xl += bandnoise(rng, N_L, FS_L, 1, 80, s["ambient"])
    # Si no está cortando, simula el transitorio producido al indexar.
    if not cutting:
        # Elige el instante de inicio del transitorio.
        t0 = rng.uniform(0.2, 1.5)
        # Añade una sinusoide amortiguada desde el instante elegido.
        xl += logU(rng, 1e-3, 1e-2)*np.exp(-np.clip(tl-t0, 0, None)*40)*(tl >= t0)*np.sin(2*np.pi*s["f_base"]*tl)
    if cls == 1:                           # externa: ráfaga 1-80 Hz + impactos NO periódicos
        # Construye una envolvente gaussiana para una ráfaga externa.
        env = np.exp(-0.5*((tl - rng.uniform(0.4, 1.6))/rng.uniform(0.1, 0.4))**2)
        # Genera el ruido de la ráfaga dentro de la banda baja.
        ev = bandnoise(rng, N_L, FS_L, 1, 80, 1.0)*env
        # Escala el evento a una amplitud externa aleatoria.
        xl += ev*logU(rng, *EXT_EVENT)/(ev.std() + 1e-12)
        # Crea entre uno y tres impactos no periódicos en la señal rápida.
        tr = np.zeros(N_H); tr[rng.integers(0, N_H, rng.integers(1, 4))] = 1
        # Excita las resonancias del cabezal y de alta frecuencia con esos impactos.
        xh += (resonator(tr, s["f_head"], s["Q"], FS_H) + resonator(tr, s["f_hf"], s["Q_hf"], FS_H))*logU(rng, 1e-3, 3e-2)
    elif cls == 2:                         # anomalía: impactos PERIÓDICOS a k*f_r que excitan la resonancia local
        # Calcula la separación temporal entre impactos periódicos.
        step = FS_H/(s["k"]*s["f_r"]); pos = np.arange(rng.uniform(0, step), N_H, step)
        # Desplaza ligeramente cada impacto y limita sus índices a la ventana.
        idx = np.clip((pos + rng.normal(0, 0.01*step, pos.size)).astype(int), 0, N_H-1)
        # Construye un tren de impactos con amplitud modulada por el giro.
        tr = np.zeros(N_H); tr[idx] = 1 + 0.5*np.sin(2*np.pi*s["f_r"]*idx/FS_H)
        # Pasa los impactos por la resonancia de alta frecuencia.
        xh += resonator(tr, s["f_hf"], s["Q_hf"], FS_H)*rng.uniform(0.1, 1.0)*3e-2
    # Añade el ruido propio del sensor a la señal rápida.
    xh += rng.normal(0, ND*np.sqrt(FS_H/2)*nm, N_H)       # ruido del sensor: constante, no depende del estado
    # Añade el ruido propio del sensor a la señal lenta.
    xl += rng.normal(0, ND*np.sqrt(FS_L/2)*nm, N_L)
    # Devuelve las señales rápida y lenta de la ventana.
    return xh, xl

# Define los límites de las bandas espectrales para ambas señales.
EH = np.geomspace(10, 15000, 17); EL = [1, 4, 10, 20, 40, 80]
# Crea un filtro pasa-altas para extraer la envolvente de la señal rápida.
SOS = butter(4, 1500, 'hp', fs=FS_H, output='sos')
# Define los nombres de las características que se guardarán.
NAMES = (["rms_h", "crest_h", "kurt_h", "centroide_h", "periodicidad_h"] + [f"banda_h{i}" for i in range(16)] +
         [f"banda_l{i}" for i in range(5)] + ["rms_l", "kurt_l"])

# Calcula las características espectrales y estadísticas de una ventana.
def feats(xh, xl):
    # Calcula la potencia espectral y las frecuencias de la señal rápida.
    sp = np.abs(np.fft.rfft(xh*np.hanning(N_H)))**2; fq = np.fft.rfftfreq(N_H, 1/FS_H); rms = xh.std()
    # Obtiene la envolvente de la señal filtrada.
    env = np.abs(hilbert(sosfilt(SOS, xh))); env -= env.mean()
    # Calcula la autocorrelación de la envolvente mediante FFT.
    ac = np.fft.irfft(np.abs(np.fft.rfft(env, 2*N_H))**2)[:N_H]
    # Calcula características temporales, espectrales y de periodicidad.
    hi = [rms, np.abs(xh).max()/rms, kurtosis(xh), (fq*sp).sum()/sp.sum(), ac[8:400].max()/(ac[0] + 1e-12)]
    # Calcula la energía logarítmica de cada banda de alta frecuencia.
    fh = [np.log10(sp[(fq >= a) & (fq < b)].sum() + 1e-12) for a, b in zip(EH[:-1], EH[1:])]
    # Calcula la potencia espectral de la señal lenta y sus frecuencias.
    sl = np.abs(np.fft.rfft(xl*np.hanning(N_L)))**2; fl = np.fft.rfftfreq(N_L, 1/FS_L)
    # Calcula la energía de baja frecuencia, RMS y curtosis.
    lo = [np.log10(sl[(fl >= a) & (fl < b)].sum() + 1e-12) for a, b in zip(EL[:-1], EL[1:])] + [xl.std(), kurtosis(xl)]
    # Concatena todas las características en un solo vector.
    return np.r_[hi, fh, lo]

# Construye el conjunto de datos sintético para varias máquinas.
def build(n_sess=40, n_win=40, seed=0, nm=1.0):
    # Inicializa el generador y las listas de datos y metadatos.
    rng = np.random.default_rng(seed); X, y, g, rpm, cal = [], [], [], [], []
    # Recorre las máquinas simuladas.
    for si in range(n_sess):
        # Genera los parámetros propios de la máquina actual.
        s = new_session(rng)
        # Recorre las tres clases de vibración.
        for c in (0, 1, 2):
            # Genera las ventanas solicitadas para la clase.
            for w in range(n_win):
                # Extrae las características y guarda etiqueta, grupo, RPM y calibración.
                X.append(feats(*window(c, s, rng, nm))); y.append(c); g.append(si); rpm.append(s["f_r"]*60); cal.append(c == 0 and w < N_CAL)
    # Convierte las listas en arreglos NumPy y los devuelve.
    return np.array(X), np.array(y), np.array(g), np.array(rpm), np.array(cal)

# Normaliza cada máquina usando sus primeras ventanas normales como línea base.
def calibrate(X, g, cal):                  # línea base propia de cada máquina (primeras ventanas sanas)
    # Copia las características para conservar intactos los datos originales.
    Xn = X.copy()
    # Procesa cada máquina de forma independiente.
    for s in np.unique(g):
        # Selecciona la máquina y calcula su media y desviación de calibración.
        m = g == s; b = X[m & cal]; Xn[m] = (X[m] - b.mean(0))/(b.std(0) + 1e-9)
    # Devuelve las características calibradas.
    return Xn

# Define el generador del clasificador Random Forest.
RF = lambda: RandomForestClassifier(150, max_depth=12, random_state=0, n_jobs=-1)
# Define el generador de una red MLP con estandarización previa.
MLP = lambda: make_pipeline(StandardScaler(), MLPClassifier((32, 16), max_iter=600, random_state=0))

# Evalúa un modelo separando las máquinas entre entrenamiento y prueba.
def cv(make, X, y, g):
    # Inicializa las colecciones de resultados de todos los folds.
    yt, yp, f1s = [], [], []
    # Crea cinco particiones sin separar una misma máquina entre folds.
    for tr, te in GroupKFold(5).split(X, y, g):
        # Entrena el modelo del fold y predice las muestras de prueba.
        p = make().fit(X[tr], y[tr]).predict(X[te]); yt += list(y[te]); yp += list(p); f1s.append(f1_score(y[te], p, average="macro"))
    # Devuelve F1 medio, dispersión y matriz de confusión global.
    return np.mean(f1s), np.std(f1s), confusion_matrix(yt, yp)

# Ejecuta el experimento solamente al ejecutar este archivo directamente.
if __name__ == "__main__":
    # Crea la carpeta de resultados y la lista de mensajes.
    out = Path("out_det"); out.mkdir(exist_ok=True); L = []
    # Imprime y almacena simultáneamente cada mensaje del experimento.
    def log(s): print(s); L.append(s)
    # Genera el conjunto de datos y su versión calibrada.
    X, y, g, rpm, cal = build(); Xn = calibrate(X, g, cal)
    # Informa las dimensiones del conjunto y la cantidad de ejemplos por clase.
    log(f"Dataset: {X.shape[0]} ventanas, {len(np.unique(g))} máquinas simuladas, {X.shape[1]} features, clases {np.bincount(y)}")
    # Abre el archivo CSV de características.
    with open(out/"dataset.csv", "w", newline="") as fh:
        # Crea el escritor y escribe los nombres de las columnas.
        w = csv.writer(fh); w.writerow(NAMES + ["sesion", "rpm", "clase"])
        # Escribe las características y metadatos de cada ventana.
        for r, gi, rp, yi in zip(X, g, rpm, y): w.writerow(list(np.round(r, 6)) + [gi, round(rp), yi])
    # Anuncia el inicio de la validación cruzada separada por máquina.
    log("\n== Validación cruzada separada por máquina (5 folds) ==")
    # Compara dos variantes de Random Forest y una red neuronal.
    for n, mk, XX in [("RandomForest sin calibrar", RF, X), ("RandomForest con calibración", RF, Xn), ("MLP 32-16 con calibración", MLP, Xn)]:
        # Evalúa el modelo y registra su F1 macro.
        m, s_, cm = cv(mk, XX, y, g); log(f"{n:32s} F1 macro = {m:.3f} +- {s_:.3f}")
        # Conserva la matriz del Random Forest calibrado para las gráficas.
        if "RandomForest con" in n: cm_rf = cm
    # Registra la matriz de confusión del modelo seleccionado.
    log("Matriz de confusión (RF con calibración; filas = real, columnas = predicho):\n" + str(cm_rf))
    # Anuncia la prueba de generalización a velocidades no vistas.
    log("\n== Generalización a rpm no vistas (entrena rpm < 22,000, prueba >= 22,000) ==")
    # Separa las velocidades de entrenamiento y de prueba.
    tr, te = rpm < 22000, rpm >= 22000
    # Evalúa el Random Forest con y sin calibración en esa separación.
    for n, XX in [("sin calibrar", X), ("con calibración", Xn)]:
        # Entrena con las velocidades bajas, predice las altas y registra F1.
        p = RF().fit(XX[tr], y[tr]).predict(XX[te]); log(f"RF {n:16s} F1 macro = {f1_score(y[te], p, average='macro'):.3f}")
    # Anuncia la prueba con un nivel de ruido del sensor tres veces mayor.
    log("\n== Robustez al ruido del sensor (entrena con ruido nominal, prueba con ruido x3, máquinas nuevas) ==")
    # Genera máquinas nuevas con ruido aumentado y calibra sus características.
    X3, y3, g3, _, c3 = build(n_sess=15, seed=7, nm=3.0); X3n = calibrate(X3, g3, c3)
    # Compara la robustez de las versiones calibrada y sin calibrar.
    for n, XA, XB in [("sin calibrar", X, X3), ("con calibración", Xn, X3n)]:
        # Entrena con los datos originales y evalúa en las máquinas nuevas.
        p = RF().fit(XA, y).predict(XB); log(f"RF {n:16s} F1 macro = {f1_score(y3, p, average='macro'):.3f}")
    # Entrena el Random Forest calibrado y ordena sus características más importantes.
    rf = RF().fit(Xn, y); imp = np.argsort(rf.feature_importances_)[::-1][:8]
    # Registra las ocho características con mayor importancia.
    log("\nFeatures más importantes: " + ", ".join(f"{NAMES[i]} ({rf.feature_importances_[i]:.2f})" for i in imp))
    # Entrena la MLP y obtiene su último componente, que contiene la red.
    mlp = MLP().fit(Xn, y); nets = mlp[-1]
    # Registra el tamaño de ambos modelos.
    log(f"Tamaño: RF = {sum(e.tree_.node_count for e in rf.estimators_)} nodos | MLP = {sum(c.size for c in nets.coefs_) + sum(b.size for b in nets.intercepts_)} parámetros")
    # Guarda todos los mensajes del experimento en un archivo de texto.
    (out/"resultados.txt").write_text("\n".join(L), encoding="utf-8")
    # Crea la figura de la matriz de confusión para el documento.
    fig, ax = plt.subplots(figsize=(4.2, 3.6)); ax.imshow(cm_rf, cmap="Blues")
    # Escribe el valor de cada celda dentro de la matriz.
    for i in range(3):
        # Recorre las columnas de la matriz.
        for j in range(3): ax.text(j, i, cm_rf[i, j], ha="center", va="center")
    # Configura las marcas, etiquetas y nombres de las clases.
    ax.set_xticks(range(3)); ax.set_yticks(range(3)); lab = ["Normal", "Externa", "Anomalía"]; ax.set_xticklabels(lab); ax.set_yticklabels(lab)
    # Configura los ejes, ajusta el diseño, guarda y cierra la figura.
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real"); ax.set_title("Matriz de confusión (RF, por máquina)"); plt.tight_layout(); plt.savefig(out/"matriz_confusion.png", dpi=200); plt.close()
    # Crea una máquina reproducible y fija su velocidad de giro en 30,000 RPM.
    rng = np.random.default_rng(3); s = new_session(rng); s["f_r"] = 500.0; fq = np.fft.rfftfreq(N_H, 1/FS_H)
    # Crea la figura donde se dibujarán los espectros de las tres clases.
    plt.figure(figsize=(8, 4))
    # Genera y dibuja el espectro de cada clase.
    for c, n in enumerate(lab):
        # Genera una ventana, calcula su magnitud espectral y la dibuja.
        xh, _ = window(c, s, rng); plt.plot(fq, 20*np.log10(np.abs(np.fft.rfft(xh*np.hanning(N_H))) + 1e-12), label=n, alpha=.75)
    # Limita el eje, etiqueta la gráfica, guarda la figura y la cierra.
    plt.xlim(0, 10000); plt.xlabel("Frecuencia (Hz)"); plt.ylabel("Magnitud (dB, relativa)"); plt.legend(); plt.grid(alpha=.3)
    plt.title("Espectros simulados (misma máquina simulada, 30,000 rpm)"); plt.tight_layout(); plt.savefig(out/"espectros_clases.png", dpi=200); plt.close()
    # Indica que todos los archivos de salida fueron generados.
    print("\nListo: out_detector/")
