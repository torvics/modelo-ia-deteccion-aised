# Importa NumPy para copiar arreglos, buscar grupos y calcular estadísticas.
import numpy as np
# Importa la función que genera las señales y características sintéticas.
from gen_vib import build
# Importa el clasificador de bosque aleatorio.
from sklearn.ensemble import RandomForestClassifier
# Importa la validación cruzada agrupada por máquina o sesión.
from sklearn.model_selection import GroupKFold
# Importa las métricas de evaluación del modelo.
from sklearn.metrics import f1_score, confusion_matrix

# Genera las características X, las etiquetas y y los identificadores de sesión g.
X, y, g = build()
# Crea una copia para normalizarla sin modificar el conjunto original.
Xn = X.copy()
# Recorre cada máquina o sesión presente en los datos.
for s in np.unique(g):
    # Selecciona las muestras de la máquina actual y toma sus primeras 10 ventanas normales.
    m = g == s; base = X[m & (y == 0)][:10]
    # Normaliza cada característica usando la media y desviación estándar de la línea base.
    Xn[m] = (X[m] - base.mean(0)) / (base.std(0) + 1e-9)
# Inicializa las listas donde se guardarán etiquetas reales y predicciones.
yt, yp = [], []
# Divide los datos por sesión para evaluar en máquinas no vistas durante el entrenamiento.
for tr, te in GroupKFold(5).split(Xn, y, g):
    # Entrena un bosque aleatorio de 200 árboles con las características normalizadas.
    mdl = RandomForestClassifier(200, random_state=0).fit(Xn[tr], y[tr])
    # Acumula las etiquetas reales y las predicciones del bloque de prueba.
    yt += list(y[te]); yp += list(mdl.predict(Xn[te]))
# Calcula e imprime el F1 macro del modelo con normalización por máquina.
print("F1 macro con línea base por máquina:", round(f1_score(yt, yp, average='macro'), 3))
# Imprime la matriz de confusión para revisar los resultados por clase.
print(confusion_matrix(yt, yp))
