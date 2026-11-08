"""Generador sintético de vibración (motion stage) + features + baseline.
Clases: 0 = mecanismo normal, 1 = perturbación externa (impactos aleatorios / piso / tono vecino),
        2 = anomalía del mecanismo (impactos PERIÓDICOS tipo rodamiento/holgura, severidad variable)
Parámetros del sensor tomados del datasheet IIS3DWB10IS: ODR 40 kHz, ruido 35 ug/rtHz, BW > 10 kHz."""

# Importa NumPy para crear arreglos, generar valores numéricos y hacer cálculos vectorizados.
import numpy as np
# Importa filtros y la transformada de Hilbert para procesar las señales de vibración.
from scipy.signal import lfilter, butter, sosfilt, hilbert
# Importa medidas estadísticas usadas para describir cada señal.
from scipy.stats import kurtosis, skew
# Importa el clasificador de bosque aleatorio.
from sklearn.ensemble import RandomForestClassifier
# Importa la validación cruzada que separa los datos por grupos o máquinas.
from sklearn.model_selection import GroupKFold
# Importa las métricas F1 y matriz de confusión para evaluar el clasificador.
from sklearn.metrics import f1_score, confusion_matrix

# Define la frecuencia de muestreo: 40 000 muestras por segundo; N es el tamaño de ventana.
FS, N = 40_000, 2048
# Calcula el ruido RMS del sensor a partir de su densidad de ruido y el ancho de banda.
NOISE_G = 35e-6 * np.sqrt(10_000)
# Crea el eje temporal compartido por todas las ventanas de señal.
t = np.arange(N) / FS

# Define un resonador de segundo orden que simula un modo estructural de la máquina.
def resonator(x, fr, Q):
    # Calcula el ángulo digital de la frecuencia resonante y el factor de amortiguamiento.
    th, r = 2*np.pi*fr/FS, np.exp(-np.pi*fr/(Q*FS))
    # Filtra la entrada con la ecuación del resonador y devuelve la señal resonante.
    return lfilter([1.0], [1, -2*r*np.cos(th), r*r], x)

# Genera los parámetros aleatorios de una sesión, que representa una máquina o condición.
def session_params(rng):
    # Devuelve la frecuencia de giro, resonancia, calidad, amplitud, frecuencia de falla y ruido.
    return dict(f0=rng.uniform(20, 200), fr=rng.uniform(900, 4000), Q=rng.uniform(10, 40),
                a1=rng.uniform(0.02, 0.15), fb=rng.uniform(60, 400), noise=rng.uniform(0.8, 1.5))

# Genera la vibración normal con armónicos y ruido excitando el modo estructural.
def base_mechanism(p, rng):
    # Suma ocho armónicos de la frecuencia de accionamiento con fase aleatoria.
    x = sum(p["a1"]/k**1.2 * np.sin(2*np.pi*k*p["f0"]*t + rng.uniform(0, 6.28)) for k in range(1, 9))
    # Añade ruido blanco pasado por el resonador para simular excitación estructural.
    x += resonator(rng.normal(0, 1, N), p["fr"], p["Q"]) * p["a1"] * 0.05
    # Devuelve la señal base de un mecanismo normal.
    return x

# Genera una ventana de señal para la clase indicada y los parámetros de una sesión.
def window(cls, p, rng):
    # Comienza con la vibración normal del mecanismo.
    x = base_mechanism(p, rng)
    # La clase 1 representa una perturbación externa.
    if cls == 1:
        # Selecciona aleatoriamente uno de tres tipos de perturbación externa.
        k = rng.integers(3)
        # El tipo 0 agrega impactos en posiciones aleatorias.
        if k == 0:
            # Crea un tren de impulsos y asigna amplitudes aleatorias a varios instantes.
            tr = np.zeros(N); tr[rng.integers(0, N, rng.integers(1, 4))] = rng.uniform(0.2, 2.0)
            # Hace pasar los impulsos por un resonador con frecuencia y calidad aleatorias.
            x += resonator(tr, rng.uniform(300, 4000), rng.uniform(8, 40)) * rng.uniform(0.05, 0.5)
        # El tipo 1 agrega vibración de piso de baja frecuencia.
        elif k == 1:
            # Diseña un filtro pasa banda y lo aplica a ruido para simular el movimiento del piso.
            x += sosfilt(butter(2, [5, 80], 'bp', fs=FS, output='sos'), rng.normal(0, 1, N)) * rng.uniform(0.05, 0.4)
        # El tipo 2 agrega un tono producido por una máquina vecina.
        else:
            # Añade una sinusoide cuya frecuencia es un múltiplo aleatorio de 60 Hz.
            x += rng.uniform(0.03, 0.2) * np.sin(2*np.pi*60*rng.integers(1, 6)*t + rng.uniform(0, 6.28))
    # La clase 2 representa una anomalía interna con impactos periódicos.
    elif cls == 2:
        # Elige la severidad, crea el tren de impulsos y calcula el período entre impactos.
        sev = rng.uniform(0.2, 1.0); tr = np.zeros(N); step = FS / p["fb"]
        # Genera las posiciones ideales de los impactos dentro de la ventana.
        pos = np.arange(rng.uniform(0, step), N, step)
        # Añade pequeñas variaciones aleatorias y convierte las posiciones a índices enteros.
        idx = (pos + rng.normal(0, 0.005*step, pos.size)).astype(int)
        # Conserva solo los índices válidos y marca esos instantes como impulsos.
        tr[idx[(idx >= 0) & (idx < N)]] = 1
        # Resuena los impactos y los escala según la severidad de la anomalía.
        x += resonator(tr, p["fr"], p["Q"]) * sev * 0.4
    # Añade el ruido del sensor ajustado por la condición de la sesión y devuelve la ventana.
    return x + rng.normal(0, NOISE_G * p["noise"], N)

# Convierte una señal de vibración en características numéricas para el modelo.
def features(x):
    # Calcula el espectro de potencia de la señal con ventana Hann y sus frecuencias.
    sp = np.abs(np.fft.rfft(x * np.hanning(N)))**2; fq = np.fft.rfftfreq(N, 1/FS)
    # Crea 17 bordes espaciados logarítmicamente para formar 16 bandas de frecuencia.
    edges = np.geomspace(10, 15_000, 17)
    # Calcula el logaritmo de la potencia acumulada en cada banda.
    bands = [np.log10(sp[(fq >= a) & (fq < b)].sum() + 1e-12) for a, b in zip(edges[:-1], edges[1:])]
    # Calcula el valor RMS, que representa la energía promedio de la señal.
    rms = np.sqrt(np.mean(x**2))
    # Filtra las frecuencias altas para analizar la envolvente de impactos.
    hp = sosfilt(butter(4, 500, 'hp', fs=FS, output='sos'), x)
    # Obtiene la envolvente mediante Hilbert y elimina su valor medio.
    env = np.abs(hilbert(hp)); env -= env.mean()
    # Calcula el espectro de la envolvente, sus frecuencias y limita el análisis a 40-600 Hz.
    es = np.abs(np.fft.rfft(env))**2; ef = np.fft.rfftfreq(N, 1/FS); m = (ef > 40) & (ef < 600)
    # Mide cuánto destaca el pico periódico de impactos frente a la potencia media.
    per = es[m].max() / (es[m].mean() + 1e-12)
    # Devuelve RMS, factor de cresta, curtosis, asimetría, frecuencia media, periodicidad y bandas.
    return np.r_[rms, np.abs(x).max()/rms, kurtosis(x), skew(x), (fq*sp).sum()/sp.sum(), per, bands]

# Construye el conjunto de datos sintético para varias sesiones, clases y ventanas.
def build(n_sess=40, n_win=50, seed=0):
    # Crea el generador reproducible y las listas de características, etiquetas y grupos.
    rng = np.random.default_rng(seed); X, y, g = [], [], []
    # Recorre todas las sesiones o máquinas simuladas.
    for s in range(n_sess):
        # Genera parámetros propios para la sesión actual.
        p = session_params(rng)
        # Genera las tres clases: normal, perturbación externa y anomalía interna.
        for c in (0, 1, 2):
            # Genera el número solicitado de ventanas por clase.
            for _ in range(n_win):
                # Extrae características, guarda la clase y asocia la muestra con su sesión.
                X.append(features(window(c, p, rng))); y.append(c); g.append(s)
    # Convierte las listas a arreglos NumPy y devuelve características, etiquetas y grupos.
    return np.array(X), np.array(y), np.array(g)

# Ejecuta este bloque solo cuando el archivo se inicia directamente.
if __name__ == "__main__":
    # Genera todas las muestras sintéticas, etiquetas y grupos de sesión.
    X, y, g = build()
    # Muestra el tamaño de la matriz de características y el número de muestras por clase.
    print("X:", X.shape, "| clases:", np.bincount(y))
    # Inicializa las listas que acumularán etiquetas reales y predicciones.
    yt, yp = [], []
    # Divide por sesión para que ninguna máquina aparezca a la vez en entrenamiento y prueba.
    for tr, te in GroupKFold(5).split(X, y, g):
        # Entrena un bosque aleatorio de 200 árboles con las muestras de entrenamiento.
        m = RandomForestClassifier(200, random_state=0).fit(X[tr], y[tr])
        # Guarda las etiquetas reales y las predicciones de las muestras de prueba.
        yt += list(y[te]); yp += list(m.predict(X[te]))
    # Calcula e imprime el F1 macro promedio entre las tres clases.
    print("F1 macro (GroupKFold):", round(f1_score(yt, yp, average='macro'), 3))
    # Imprime la matriz de confusión para observar aciertos y errores por clase.
    print(confusion_matrix(yt, yp))
