# -*- coding: utf-8 -*-
"""
Complemento semi-supervisado del TFG "Detección de transacciones bancarias
fraudulentas con aprendizaje no supervisado".

TFG - Grado en Ingeniería Informática, Universidad de Almería.

Este módulo EXTIENDE el trabajo original (clase PipelineFraude, aprendizaje
NO supervisado) con un enfoque SEMI-supervisado:

    1. Se reutilizan los resultados del TFG (columna 'consenso') para generar
       unas pocas etiquetas semilla fiables.
    2. A partir de esas etiquetas + el resto de transacciones sin etiquetar,
       se aplican algoritmos semi-supervisados (self-training, label
       propagation, ...) que propagan el conocimiento a todo el conjunto.
    3. Se compara el resultado con el del TFG en un pequeño estudio.

El módulo se construye por fases (una función/bloque por fase del plan) para
poder revisarlo paso a paso. Comparte estilo, semilla (random_state=42) y
matriz de características con `pipeline_fraude.py`.

ADVERTENCIA METODOLÓGICA (a tener presente en todo el estudio):
Las etiquetas provienen del consenso de los métodos no supervisados del TFG.
No existe una verdad de fraude real. Por tanto NO se deben interpretar las
métricas como aciertos absolutos; el estudio se enfoca en "¿qué aporta el
enfoque semi-supervisado frente al no supervisado?" (capacidad de generalizar,
transacciones nuevas que descubre, sensibilidad a la semilla).
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import (SelfTrainingClassifier, LabelPropagation,
                                     LabelSpreading)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from pipeline_fraude import PipelineFraude


# Variables clave del patrón de fraude del TFG (importe alto respecto al saldo,
# a menudo con varios intentos de acceso). Se usan para perfilar las anomalías.
COLUMNAS_PERFIL = ['TransactionAmount', 'AccountBalance',
                   'AmountBalanceRatio', 'LoginAttempts']


# =====================================================================
# FASE 0 - PREPARACIÓN DE DATOS
# =====================================================================
def preparar_datos(ruta_csv='bank_transactions_data_2.csv',
                   ruta_resultados='resultados_pipeline.csv',
                   random_state=42, verbose=True):
    """Reutiliza el preprocesamiento del TFG y cruza con sus resultados.

    Devuelve un diccionario con todos los objetos que necesitarán las fases
    posteriores, garantizando que la matriz de características es EXACTAMENTE
    la misma que empleó el pipeline no supervisado.

    Devuelve
    --------
    datos : dict con las claves
        'X'             : np.ndarray (n, 16)   matriz escalada de características
        'feature_names' : list[str]            nombres de las 16 características
        'pca_2d'        : np.ndarray (n, 2)    proyección PCA (para visualizar)
        'df'            : pd.DataFrame          features SIN escalar + TransactionID
                                                + consenso (para perfilar/interpretar)
        'transaction_id': pd.Series            identificador por fila (orden de X)
        'consenso'      : pd.Series            nº de detectores del TFG (0-4) por fila
    """
    if verbose:
        print("=" * 70)
        print("FASE 0 - PREPARACIÓN DE DATOS")
        print("=" * 70)

    # --- 1. Reutilizar el pipeline del TFG para obtener la matriz escalada ---
    # Ejecutamos únicamente carga + preprocesado (no el EDA, que solo genera
    # figuras). Así X_scaled, feature_names y pca_2d son idénticos al TFG.
    pipeline = PipelineFraude(ruta_csv, random_state=random_state)
    pipeline.cargar_datos()
    pipeline.preprocesar()

    X = pipeline.X_scaled
    feature_names = pipeline.feature_names
    pca_2d = pipeline.pca_2d

    # --- 2. TransactionID en el MISMO orden de filas que X ---
    # preprocesar() no reordena ni elimina filas: la fila i de X se corresponde
    # con la fila i de df_raw. Reseteamos el índice para alinear por posición.
    transaction_id = pipeline.df_raw['TransactionID'].reset_index(drop=True)

    # --- 3. Traer la columna 'consenso' desde los resultados del TFG ---
    # Cruzamos por TransactionID (merge explícito) en lugar de fiarnos solo del
    # orden: es más robusto y deja la trazabilidad clara para la memoria.
    resultados_tfg = pd.read_csv(ruta_resultados)
    cruce = transaction_id.to_frame().merge(
        resultados_tfg[['TransactionID', 'consenso']],
        on='TransactionID', how='left')
    consenso = cruce['consenso']

    # --- 4. Verificaciones de integridad ---
    n = X.shape[0]
    assert len(transaction_id) == n, "Desalineación entre X y TransactionID"
    assert consenso.isnull().sum() == 0, "Hay TransactionID sin consenso (cruce roto)"
    assert transaction_id.is_unique, "TransactionID duplicados: el cruce sería ambiguo"

    # DataFrame interpretable: características SIN escalar + id + consenso.
    # Útil en fases posteriores para perfilar las transacciones detectadas.
    df = pipeline.df_features.reset_index(drop=True).copy()
    df.insert(0, 'TransactionID', transaction_id.values)
    df['consenso'] = consenso.values

    if verbose:
        print("\n--- Resumen de la preparación ---")
        print(f"Transacciones                 : {n}")
        print(f"Características (escaladas)     : {X.shape[1]}")
        print(f"Nombres de características      : {feature_names}")
        print(f"Proyección PCA para visualizar : {pca_2d.shape}")
        print("\n--- Distribución del consenso (nº de detectores del TFG, 0-4) ---")
        print(consenso.value_counts().sort_index().to_string())
        print("\nIntegridad OK: sin nulos, sin duplicados, X y consenso alineados.")

    return {
        'X': X,
        'feature_names': feature_names,
        'pca_2d': pca_2d,
        'df': df,
        'df_raw': pipeline.df_raw.reset_index(drop=True),  # datos originales (ejemplos)
        'transaction_id': transaction_id,
        'consenso': consenso,
    }


# =====================================================================
# FASE 1 - ETIQUETAS SEMILLA
# =====================================================================
def construir_etiquetas(datos, umbral_consenso_positivo=4, ratio_neg_pos=10,
                        n_positivos=None, n_negativos=None,
                        random_state=42, verbose=True):
    """Construye el vector de etiquetas semilla `y` para semi-supervisado.

    Convención de scikit-learn:
        y = 1  -> transacción ANÓMALA (positivo), consenso alto del TFG
        y = 0  -> transacción NORMAL fiable (negativo), muestra de consenso 0
        y = -1 -> SIN ETIQUETAR (la inmensa mayoría del conjunto)

    Nunca se etiquetan como negativas TODAS las de consenso 0: solo una muestra
    reducida (ratio_neg_pos por cada positivo). Así se evita un desbalanceo
    extremo en el conjunto etiquetado y se asume que "no anómalo" != "seguro".

    Parámetros
    ----------
    umbral_consenso_positivo : int
        Se etiqueta como positivo toda transacción con consenso >= este valor.
        (4 -> las 5 de máxima sospecha; 3 -> las 10 marcadas por >=3 métodos.)
    ratio_neg_pos : int
        Nº de negativos por cada positivo (1:ratio). Solo se usa si
        n_negativos es None.
    n_positivos : int o None
        Tope de positivos. Si None, se usan todos los que superan el umbral.
        Si es menor que los disponibles, se toma una muestra aleatoria.
    n_negativos : int o None
        Nº explícito de negativos. Si None, se calcula como n_pos * ratio_neg_pos.
    random_state : int
        Semilla para el muestreo (reproducibilidad).

    Devuelve
    --------
    info : dict con
        'y'          : np.ndarray (n,)  vector de etiquetas (-1/0/1)
        'idx_pos'    : np.ndarray       índices de los positivos
        'idx_neg'    : np.ndarray       índices de los negativos
        'n_pos'      : int              nº de positivos etiquetados
        'n_neg'      : int              nº de negativos etiquetados
        'n_sin_etiq' : int              nº de transacciones sin etiquetar
        'params'     : dict             configuración usada (para la memoria)
    """
    consenso = datos['consenso'].values
    n = len(consenso)
    rng = np.random.RandomState(random_state)
    y = np.full(n, -1, dtype=int)

    # --- Positivos: consenso alto ---
    idx_pos_disponibles = np.where(consenso >= umbral_consenso_positivo)[0]
    if n_positivos is not None and n_positivos < len(idx_pos_disponibles):
        idx_pos = np.sort(rng.choice(idx_pos_disponibles, size=n_positivos,
                                     replace=False))
    else:
        idx_pos = idx_pos_disponibles
    n_pos = len(idx_pos)

    # --- Negativos: muestra reducida de las normales fiables (consenso 0) ---
    if n_negativos is None:
        n_negativos = n_pos * ratio_neg_pos
    idx_neg_disponibles = np.where(consenso == 0)[0]
    n_negativos = min(n_negativos, len(idx_neg_disponibles))
    idx_neg = np.sort(rng.choice(idx_neg_disponibles, size=n_negativos,
                                 replace=False))

    y[idx_pos] = 1
    y[idx_neg] = 0
    n_sin_etiq = int((y == -1).sum())

    params = {
        'umbral_consenso_positivo': umbral_consenso_positivo,
        'ratio_neg_pos': ratio_neg_pos,
        'n_positivos': n_positivos,
        'n_negativos': n_negativos,
        'random_state': random_state,
    }

    if verbose:
        print(f"  Positivos (y=1, consenso>={umbral_consenso_positivo}): {n_pos}")
        print(f"  Negativos (y=0, muestra de consenso 0)              : {len(idx_neg)}")
        print(f"  Sin etiquetar (y=-1)                                : {n_sin_etiq}")
        print(f"  Etiquetados en total                                : {n_pos + len(idx_neg)}"
              f" ({(n_pos + len(idx_neg)) / n:.2%} del conjunto)")

    return {
        'y': y,
        'idx_pos': idx_pos,
        'idx_neg': idx_neg,
        'n_pos': n_pos,
        'n_neg': len(idx_neg),
        'n_sin_etiq': n_sin_etiq,
        'params': params,
    }


# Configuraciones de semilla que se compararán en el estudio (Fase 5).
# Varían el umbral de consenso (cuántos positivos) y el ratio de negativos,
# que son las dos decisiones clave del enfoque semi-supervisado.
CONFIGURACIONES_SEMILLA = {
    'A_consenso4_1a10': dict(umbral_consenso_positivo=4, ratio_neg_pos=10),
    'B_consenso3_1a10': dict(umbral_consenso_positivo=3, ratio_neg_pos=10),
    'C_consenso3_1a5':  dict(umbral_consenso_positivo=3, ratio_neg_pos=5),
    'D_consenso3_1a20': dict(umbral_consenso_positivo=3, ratio_neg_pos=20),
}


def resumen_configuraciones(datos, configuraciones=None, random_state=42):
    """Construye todas las configuraciones de semilla y las resume en una
    tabla comparativa (nº de positivos, negativos y sin etiquetar de cada una).
    Sirve para justificar en la memoria el efecto de la semilla."""
    if configuraciones is None:
        configuraciones = CONFIGURACIONES_SEMILLA

    print("\n" + "=" * 70)
    print("FASE 1 - CONFIGURACIONES DE SEMILLA")
    print("=" * 70)

    filas = []
    etiquetas = {}
    for nombre, kwargs in configuraciones.items():
        print(f"\n[{nombre}] {kwargs}")
        info = construir_etiquetas(datos, random_state=random_state, **kwargs)
        etiquetas[nombre] = info
        filas.append({
            'configuracion': nombre,
            'umbral': info['params']['umbral_consenso_positivo'],
            'ratio_neg_pos': info['params']['ratio_neg_pos'],
            'positivos': info['n_pos'],
            'negativos': info['n_neg'],
            'etiquetados': info['n_pos'] + info['n_neg'],
            'sin_etiquetar': info['n_sin_etiq'],
        })

    tabla = pd.DataFrame(filas).set_index('configuracion')
    print("\n--- Tabla comparativa de configuraciones de semilla ---")
    print(tabla.to_string())
    return tabla, etiquetas


# =====================================================================
# FASE 2 - SELF-TRAINING
# =====================================================================
def perfil_anomalias(datos, mascara, columnas=COLUMNAS_PERFIL):
    """Compara la media de las variables clave en las transacciones marcadas
    como anómalas frente a la media global. ratio > 1 indica que la variable
    está sobre-representada en las anomalías (patrón de fraude del TFG)."""
    df = datos['df']
    perfil = pd.DataFrame({
        'media_global': df[columnas].mean(),
        'media_anomalias': df.loc[mascara, columnas].mean(),
    })
    perfil['ratio'] = perfil['media_anomalias'] / perfil['media_global']
    return perfil.round(3)


def aplicar_self_training(datos, info_etiquetas, threshold=0.75,
                          random_state=42, verbose=True):
    """Entrena un SelfTrainingClassifier (base: RandomForest balanceado) sobre
    las etiquetas semilla y propaga a los puntos de alta confianza.

    Devuelve un dict con las etiquetas propagadas, la máscara de anomalías
    predicha sobre todo el conjunto y las métricas de propagación.
    """
    X = datos['X']
    y = info_etiquetas['y']

    base = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                  random_state=random_state, n_jobs=-1)
    modelo = SelfTrainingClassifier(estimator=base, threshold=threshold)
    modelo.fit(X, y)

    # --- Etiquetas propagadas por el self-training ---
    # transduction_: etiqueta asignada a cada muestra (-1 si nunca se etiquetó).
    # labeled_iter_: iteración en que se etiquetó (0 = semilla original,
    #                -1 = nunca etiquetada, >=1 = propagada por el algoritmo).
    transduccion = modelo.transduction_
    iteracion = modelo.labeled_iter_
    propagadas = iteracion >= 1
    nuevas_pos = int(((transduccion == 1) & propagadas).sum())
    nuevas_neg = int(((transduccion == 0) & propagadas).sum())
    sin_etiquetar_final = int((transduccion == -1).sum())

    # --- Veredicto final del modelo sobre TODO el conjunto ---
    # Aquí está el valor del semi-supervisado: puede marcar como anómalas
    # transacciones que ninguna semilla cubría (capacidad de generalizar).
    pred = modelo.predict(X)
    idx_clase_pos = list(modelo.classes_).index(1)
    proba_anom = modelo.predict_proba(X)[:, idx_clase_pos]
    mascara_anom = pred == 1
    n_anom = int(mascara_anom.sum())
    conf_media = float(proba_anom[mascara_anom].mean()) if n_anom > 0 else np.nan

    resultado = {
        'modelo': modelo,
        'pred': pred,
        'proba_anom': proba_anom,
        'mascara_anom': mascara_anom,
        'transduccion': transduccion,
        'nuevas_pos': nuevas_pos,
        'nuevas_neg': nuevas_neg,
        'nuevas_total': nuevas_pos + nuevas_neg,
        'sin_etiquetar_final': sin_etiquetar_final,
        'n_anom_pred': n_anom,
        'conf_media_anom': conf_media,
    }

    if verbose:
        print(f"  Etiquetas nuevas propagadas : {resultado['nuevas_total']}"
              f"  (anómalas: {nuevas_pos} | normales: {nuevas_neg})")
        print(f"  Sin etiquetar tras propagar : {sin_etiquetar_final}")
        print(f"  Anomalías (predicción final): {n_anom} ({n_anom / len(pred):.2%})"
              f"  | confianza media: {conf_media:.3f}")

    return resultado


def estudio_self_training(datos, configuraciones=None, random_state=42):
    """Aplica self-training a las 4 configuraciones de semilla y compara:
    etiquetas propagadas, anomalías predichas y perfil de las anomalías."""
    if configuraciones is None:
        configuraciones = CONFIGURACIONES_SEMILLA

    print("\n" + "=" * 70)
    print("FASE 2 - SELF-TRAINING (SelfTrainingClassifier + RandomForest)")
    print("=" * 70)

    filas, resultados = [], {}
    for nombre, kwargs in configuraciones.items():
        print(f"\n[{nombre}] {kwargs}")
        info = construir_etiquetas(datos, random_state=random_state,
                                   verbose=False, **kwargs)
        res = aplicar_self_training(datos, info, random_state=random_state)
        resultados[nombre] = res
        filas.append({
            'configuracion': nombre,
            'semillas_pos': info['n_pos'],
            'semillas_neg': info['n_neg'],
            'nuevas_anom': res['nuevas_pos'],
            'nuevas_norm': res['nuevas_neg'],
            'sin_etiq_final': res['sin_etiquetar_final'],
            'anom_pred': res['n_anom_pred'],
            'conf_media': round(res['conf_media_anom'], 3),
        })

    tabla = pd.DataFrame(filas).set_index('configuracion')
    print("\n--- Tabla comparativa del self-training ---")
    print(tabla.to_string())

    print("\n--- Perfil de las anomalías predichas por cada configuración ---")
    print("(ratio > 1: la variable es mayor de lo normal en las anomalías)")
    for nombre, res in resultados.items():
        print(f"\n[{nombre}] {res['n_anom_pred']} anomalías predichas:")
        print(perfil_anomalias(datos, res['mascara_anom']).to_string())

    return tabla, resultados


def sensibilidad_muestreo_negativos(datos, config_kwargs, semillas=(42, 7, 123),
                                    random_state_modelo=42):
    """Comprobación SECUNDARIA y acotada: repite una configuración cambiando
    solo la semilla del muestreo de negativos, para ver si el nº de anomalías
    y el perfil se mantienen estables (robustez a QUÉ negativos caen)."""
    print("\n" + "=" * 70)
    print("FASE 2 (secundario) - SENSIBILIDAD AL MUESTREO DE NEGATIVOS")
    print(f"Configuración: {config_kwargs} | semillas: {semillas}")
    print("=" * 70)
    filas = []
    for s in semillas:
        info = construir_etiquetas(datos, random_state=s, verbose=False,
                                   **config_kwargs)
        res = aplicar_self_training(datos, info, random_state=random_state_modelo,
                                    verbose=False)
        perfil = perfil_anomalias(datos, res['mascara_anom'])
        filas.append({
            'semilla_neg': s,
            'anom_pred': res['n_anom_pred'],
            'nuevas_anom': res['nuevas_pos'],
            'ratio_importe': perfil.loc['TransactionAmount', 'ratio'],
            'ratio_imp_saldo': perfil.loc['AmountBalanceRatio', 'ratio'],
            'ratio_intentos': perfil.loc['LoginAttempts', 'ratio'],
        })
    tabla = pd.DataFrame(filas).set_index('semilla_neg')
    print(tabla.to_string())
    return tabla


# =====================================================================
# FASE 3 - PROPAGACIÓN POR VECINDAD (Label Propagation / Label Spreading)
# =====================================================================
# Métodos de propagación por vecindad que se comparan. A diferencia del
# self-training (propaga según la confianza de un clasificador), estos propagan
# por la estructura del grafo de vecinos de los datos. Se prueban los dos
# algoritmos de sklearn con los dos kernels, usando sus parámetros por defecto
# (knn: n_neighbors=7; rbf: gamma=20).
METODOS_VECINDAD = {
    'LP-knn': dict(algoritmo='propagation', kernel='knn'),
    'LP-rbf': dict(algoritmo='propagation', kernel='rbf'),
    'LS-knn': dict(algoritmo='spreading', kernel='knn'),
    'LS-rbf': dict(algoritmo='spreading', kernel='rbf'),
}


def aplicar_propagacion_vecindad(datos, info_etiquetas, algoritmo='spreading',
                                 kernel='knn', n_neighbors=7, gamma=20,
                                 max_iter=2000, verbose=True):
    """Aplica LabelPropagation o LabelSpreading y devuelve las anomalías.

    Estos modelos asignan una etiqueta (0/1) a TODAS las transacciones a partir
    del grafo de vecindad; no dejan puntos sin etiquetar. La anomalía es la
    clase positiva propagada (transduction_ == 1).
    """
    X = datos['X']
    y = info_etiquetas['y']

    Clase = LabelSpreading if algoritmo == 'spreading' else LabelPropagation
    kwargs = dict(kernel=kernel, max_iter=max_iter)
    if kernel == 'knn':
        kwargs['n_neighbors'] = n_neighbors
    else:
        kwargs['gamma'] = gamma

    modelo = Clase(**kwargs)
    # Estos modelos avisan de no-convergencia con semillas tan escasas; el
    # resultado transducido es igualmente válido, así que silenciamos el aviso.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        modelo.fit(X, y)

    transduccion = modelo.transduction_
    idx_clase_pos = list(modelo.classes_).index(1)
    # label_distributions_: distribución de probabilidad sobre las clases.
    proba_anom = modelo.label_distributions_[:, idx_clase_pos]
    mascara_anom = transduccion == 1
    n_anom = int(mascara_anom.sum())

    resultado = {
        'modelo': modelo,
        'transduccion': transduccion,
        'proba_anom': proba_anom,
        'mascara_anom': mascara_anom,
        'n_anom': n_anom,
    }
    if verbose:
        etiqueta = f"{algoritmo}/{kernel}"
        print(f"  {etiqueta:16s}: anomalías -> {n_anom} ({n_anom / len(X):.2%})")
    return resultado


def solapamiento(mascara_a, mascara_b):
    """Comparación de dos conjuntos de anomalías: cuántas comparten, cuántas
    son exclusivas de cada uno y el índice de Jaccard."""
    a = set(np.where(mascara_a)[0])
    b = set(np.where(mascara_b)[0])
    interseccion = len(a & b)
    union = len(a | b)
    return {
        'comparten': interseccion,
        'solo_A': len(a - b),
        'solo_B': len(b - a),
        'jaccard': round(interseccion / union, 3) if union else np.nan,
    }


def estudio_propagacion(datos, configuraciones=None,
                        config_detalle='B_consenso3_1a10', random_state=42):
    """Aplica los 4 métodos de vecindad a las 4 configuraciones y los compara
    con el self-training: recuentos, solapamiento y perfil de las anomalías."""
    if configuraciones is None:
        configuraciones = CONFIGURACIONES_SEMILLA

    print("\n" + "=" * 70)
    print("FASE 3 - PROPAGACIÓN POR VECINDAD (LabelPropagation / LabelSpreading)")
    print("=" * 70)

    # Para cada configuración guardamos la máscara del self-training y las de
    # los 4 métodos de vecindad, de modo que podamos comparar conjuntos.
    mascaras = {}   # nombre_config -> {'ST': mask, 'LP-knn': mask, ...}
    filas_conteo = []
    for nombre, kwargs in configuraciones.items():
        info = construir_etiquetas(datos, random_state=random_state,
                                   verbose=False, **kwargs)
        # Self-training (referencia de la Fase 2)
        res_st = aplicar_self_training(datos, info, random_state=random_state,
                                       verbose=False)
        mascaras[nombre] = {'ST': res_st['mascara_anom']}
        fila = {'configuracion': nombre, 'self_training': res_st['n_anom_pred']}
        # 4 métodos de vecindad
        for metodo, params in METODOS_VECINDAD.items():
            res = aplicar_propagacion_vecindad(datos, info, verbose=False, **params)
            mascaras[nombre][metodo] = res['mascara_anom']
            fila[metodo] = res['n_anom']
        filas_conteo.append(fila)

    conteo = pd.DataFrame(filas_conteo).set_index('configuracion')
    print("\n--- Nº de anomalías por método y configuración ---")
    print(conteo.to_string())

    # --- Solapamiento self-training vs cada método de vecindad ---
    print("\n--- Solapamiento self-training (A) vs vecindad (B) por configuración ---")
    filas_solap = []
    for nombre in configuraciones:
        for metodo in METODOS_VECINDAD:
            s = solapamiento(mascaras[nombre]['ST'], mascaras[nombre][metodo])
            filas_solap.append({'configuracion': nombre, 'vecindad': metodo, **s})
    solap = pd.DataFrame(filas_solap).set_index(['configuracion', 'vecindad'])
    print(solap.to_string())

    # --- Perfil de las anomalías de cada método (configuración de detalle) ---
    print(f"\n--- Perfil de las anomalías por método [config {config_detalle}] ---")
    print("(ratio > 1: la variable es mayor de lo normal en las anomalías)")
    for metodo, mascara in mascaras[config_detalle].items():
        n = int(mascara.sum())
        etiqueta = 'self-training' if metodo == 'ST' else metodo
        print(f"\n[{etiqueta}] {n} anomalías:")
        if n > 0:
            print(perfil_anomalias(datos, mascara).to_string())

    return conteo, solap, mascaras


# =====================================================================
# FASE 4 - PU-LEARNING (Elkan & Noto, 2008)
# =====================================================================
# Positive-Unlabeled learning: solo hay POSITIVOS fiables (las semillas de
# consenso alto) y el RESTO es NO ETIQUETADO (mezcla desconocida de normales
# y anomalías no descubiertas). NO se muestrean negativos.
#
# IMPORTANTE - PU NO usa negativos, luego el ratio neg:pos es irrelevante y
# las configuraciones B, C y D (todas umbral>=3, 10 positivos) COLAPSAN en el
# mismo escenario. Por eso PU se corre en solo DOS escenarios, por umbral:
#   - umbral>=4 :  5 positivos  (c ~ 0.05)
#   - umbral>=3 : 10 positivos  (c ~ 0.15)
#
# Método (Elkan & Noto 2008, supuesto SCAR "etiquetado completamente al azar"):
#   1. Se entrena g(x) = P(s=1|x) distinguiendo positivos (s=1) de no
#      etiquetados (s=0).
#   2. Se estima c = P(s=1|y=1) con el estimador e1: media de g(x) sobre los
#      positivos, calculada OUT-OF-FOLD por validación cruzada (insesgada: los
#      positivos se puntúan con un modelo que no los vio en su fold).
#   3. Puntuación ajustada w(x) = g(x)/c, con g del modelo reajustado sobre
#      todo (full-fit). Anomalía si w >= 0.5.
#
# AVISO (escala de w): con tan pocos positivos, w = g/c es una PUNTUACIÓN /
# RANKING de sospecha, NO una probabilidad calibrada. Su escala absoluta no es
# interpretable como P(y=1|x); de hecho w puede superar 1. El umbral 0.5 se
# justifica OPERATIVAMENTE (no probabilísticamente): recupera todas las
# semillas positivas y produce un recuento comparable al de los otros
# detectores del estudio.
#
# LIMITACIÓN (inconsistencia OOF-c / full-fit-g): c se estima con g OUT-OF-FOLD
# pero w se puntúa con g FULL-FIT. Ambas g no están en la misma escala (el
# full-fit "ha visto" las semillas y sube g), así que w arrastra un sesgo. Se
# asume a conciencia: el full-fit garantiza que las semillas quedan marcadas y
# es coherente con cómo puntúan in-sample el resto de métodos del estudio
# (self-training, label spreading). Alternativa (puntuar también OOF) se
# descartó porque desmarca algunas semillas (w<0.5) por el ruido de tener
# poquísimos positivos.
def aplicar_pu_learning(datos, umbral_consenso_positivo=3, n_folds=5,
                        umbral_w=0.5, random_state=42, verbose=True):
    """Aplica PU-learning de Elkan & Noto sobre las semillas de consenso alto.

    Devuelve un dict con la c estimada, la puntuación w, la máscara de
    anomalías (w >= umbral_w) y metadatos del escenario.
    """
    X = datos['X']
    consenso = datos['consenso'].values
    # s = 1 para los positivos (semillas); 0 para TODO lo demás (no etiquetado).
    s = (consenso >= umbral_consenso_positivo).astype(int)
    idx_pos = np.where(s == 1)[0]
    n_pos = len(idx_pos)
    k = min(n_folds, n_pos)   # no puede haber más folds que positivos

    def nuevo_clf():
        return RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                      random_state=random_state, n_jobs=-1)

    # --- Paso 2: c por validación cruzada (estimador e1, OOF) ---
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    g_oof = cross_val_predict(nuevo_clf(), X, s, cv=skf,
                              method='predict_proba', n_jobs=-1)[:, 1]
    c = float(g_oof[idx_pos].mean())

    # --- Pasos 1 y 3: g full-fit y puntuación ajustada w = g/c ---
    clf = nuevo_clf()
    clf.fit(X, s)
    g = clf.predict_proba(X)[:, 1]
    w = g / c                       # PUNTUACIÓN de sospecha (no probabilidad)
    mascara_anom = w >= umbral_w
    n_anom = int(mascara_anom.sum())
    todas_semillas = bool(mascara_anom[idx_pos].all())

    resultado = {
        'umbral': umbral_consenso_positivo,
        'n_pos': n_pos,
        'folds': k,
        'c': c,
        'g': g,
        'w': w,
        'mascara_anom': mascara_anom,
        'n_anom': n_anom,
        'todas_semillas_marcadas': todas_semillas,
    }
    if verbose:
        print(f"  c estimada (CV, e1) : {c:.4f} | anomalías (w>=0.5): "
              f"{n_anom} ({n_anom / len(X):.2%}) | semillas marcadas: "
              f"{todas_semillas}")
    return resultado


def estudio_pu_learning(datos, umbrales=(4, 3), random_state=42):
    """Corre PU-learning en los DOS escenarios no redundantes (umbral≥4 y
    umbral≥3) y reporta, PERFIL-PRIMERO, el perfil de las anomalías y luego
    el recuento."""
    print("\n" + "=" * 70)
    print("FASE 4 - PU-LEARNING (Elkan & Noto, 2008)")
    print("=" * 70)
    print("Nota: PU no muestrea negativos -> B=C=D colapsan; solo 2 escenarios.")

    resultados = {}
    for umbral in umbrales:
        res = aplicar_pu_learning(datos, umbral_consenso_positivo=umbral,
                                  random_state=random_state, verbose=False)
        resultados[umbral] = res
        print(f"\n--- ESCENARIO umbral consenso>={umbral} "
              f"({res['n_pos']} positivos, c={res['c']:.4f}) ---")
        # PERFIL PRIMERO (ratio vs media global)
        print("Perfil de las anomalías (ratio > 1: mayor de lo normal):")
        print(perfil_anomalias(datos, res['mascara_anom']).to_string())
        # Luego el recuento
        print(f"Nº de anomalías (w>=0.5): {res['n_anom']} "
              f"({res['n_anom'] / len(datos['X']):.2%}) | "
              f"todas las semillas marcadas: {res['todas_semillas_marcadas']}")

    return resultados


# =====================================================================
# FASE 5 - ESTUDIO COMPARATIVO CON EL TFG
# =====================================================================
# Nombres legibles de cada método semi-supervisado (para tablas y figuras).
ETIQUETAS_METODO = {
    'ST': 'self-training',
    'LP-knn': 'LabelProp-knn (contraejemplo)',   # clamping duro + kernel local
    'LP-rbf': 'LabelProp-rbf',
    'LS-knn': 'LabelSpread-knn',
    'LS-rbf': 'LabelSpread-rbf',
    'PU': 'PU-learning (Elkan-Noto)',
}

# Columnas originales legibles para mostrar ejemplos concretos de exclusivas.
COLUMNAS_EJEMPLO = ['TransactionID', 'TransactionAmount', 'AccountBalance',
                    'TransactionType', 'Channel', 'CustomerAge',
                    'CustomerOccupation', 'LoginAttempts']


def _calcular_todas_mascaras(datos, configuraciones, random_state=42):
    """Calcula, en silencio, la máscara de anomalías de cada método
    (self-training + 4 de vecindad) para cada configuración de semilla."""
    mascaras = {}
    for nombre, kwargs in configuraciones.items():
        info = construir_etiquetas(datos, random_state=random_state,
                                   verbose=False, **kwargs)
        res_st = aplicar_self_training(datos, info, random_state=random_state,
                                       verbose=False)
        mascaras[nombre] = {'ST': res_st['mascara_anom']}
        for metodo, params in METODOS_VECINDAD.items():
            r = aplicar_propagacion_vecindad(datos, info, verbose=False, **params)
            mascaras[nombre][metodo] = r['mascara_anom']
    return mascaras


def _guardar_fig(nombre, dir_figuras):
    ruta = os.path.join(dir_figuras, nombre)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close('all')
    print(f"   [Figura guardada: {ruta}]")


def _fig_pca_solapamiento(datos, mascara_semi, mascara_tfg, titulo, nombre,
                          dir_figuras):
    """PCA del TFG con 4 categorías: normales, solo TFG, ambos, y exclusivas
    del semi-supervisado (lo que descubre de nuevo)."""
    pca = datos['pca_2d']
    comparten = mascara_semi & mascara_tfg
    solo_tfg = mascara_tfg & ~mascara_semi
    solo_semi = mascara_semi & ~mascara_tfg
    normales = ~(mascara_semi | mascara_tfg)

    plt.figure(figsize=(9, 7))
    plt.scatter(pca[normales, 0], pca[normales, 1], c='lightgrey', s=8,
                alpha=0.4, label=f'Normales [{int(normales.sum())}]')
    plt.scatter(pca[solo_tfg, 0], pca[solo_tfg, 1], c='tab:blue', s=35,
                marker='o', edgecolor='k', linewidth=0.3,
                label=f'Solo TFG (consenso≥1) [{int(solo_tfg.sum())}]')
    plt.scatter(pca[comparten, 0], pca[comparten, 1], c='tab:orange', s=55,
                marker='o', edgecolor='k', linewidth=0.3,
                label=f'Ambos [{int(comparten.sum())}]')
    plt.scatter(pca[solo_semi, 0], pca[solo_semi, 1], c='red', s=70,
                marker='X', edgecolor='k', linewidth=0.4,
                label=f'Solo semi-sup. (nuevas) [{int(solo_semi.sum())}]')
    plt.title(titulo)
    plt.xlabel('Componente principal 1')
    plt.ylabel('Componente principal 2')
    plt.legend(loc='best', fontsize=9)
    _guardar_fig(nombre, dir_figuras)


def _fig_pca_metodos(datos, mascaras_config, mascara_tfg, nombre, dir_figuras):
    """Small multiples: anomalías de cada método sobre la PCA, con las
    anomalías del TFG (consenso≥1) de fondo para comparar la dispersión."""
    pca = datos['pca_2d']
    metodos = list(mascaras_config.keys())
    _, ejes = plt.subplots(2, 3, figsize=(16, 9))
    for eje, metodo in zip(ejes.flat, metodos):
        m = mascaras_config[metodo]
        eje.scatter(pca[:, 0], pca[:, 1], c='lightgrey', s=6, alpha=0.3)
        eje.scatter(pca[mascara_tfg, 0], pca[mascara_tfg, 1], c='tab:blue',
                    s=20, alpha=0.6, label='TFG (consenso≥1)')
        eje.scatter(pca[m, 0], pca[m, 1], c='red', marker='X', s=40,
                    edgecolor='k', linewidth=0.3, label='Semi-sup.')
        eje.set_title(f'{ETIQUETAS_METODO[metodo]}  ({int(m.sum())})', fontsize=10)
        eje.set_xlabel('CP1'); eje.set_ylabel('CP2')
        eje.legend(fontsize=7, loc='best')
    # Sobra un panel (5 métodos en una rejilla 2x3): lo ocultamos.
    for eje in ejes.flat[len(metodos):]:
        eje.axis('off')
    plt.suptitle('Anomalías semi-supervisadas sobre la proyección PCA del TFG',
                 fontsize=13)
    _guardar_fig(nombre, dir_figuras)


def estudio_comparativo_tfg(datos, configuraciones=None,
                            config_principal='B_consenso3_1a10',
                            metodo_principal='ST', random_state=42,
                            dir_figuras='figuras'):
    """Estudio comparativo del complemento semi-supervisado frente al TFG:
    (1) visualizaciones PCA, (2) solapamiento global con las anomalías del TFG
    (consenso≥1), (3) transacciones exclusivas nuevas (consenso 0/1)."""
    if configuraciones is None:
        configuraciones = CONFIGURACIONES_SEMILLA

    print("\n" + "=" * 70)
    print("FASE 5 - ESTUDIO COMPARATIVO CON EL TFG")
    print("=" * 70)

    consenso = datos['consenso'].values
    mascara_tfg = consenso >= 1          # anomalías del TFG (consenso≥1)
    n_tfg = int(mascara_tfg.sum())
    print(f"\nAnomalías del TFG (consenso≥1): {n_tfg}")
    print(f"Configuración principal: {config_principal} | método principal: "
          f"{ETIQUETAS_METODO[metodo_principal]}")

    mascaras = _calcular_todas_mascaras(datos, configuraciones, random_state)

    # -----------------------------------------------------------------
    # (2) Solapamiento global semi-supervisado vs TFG (todas las configs)
    # -----------------------------------------------------------------
    print("\n--- (2) Solapamiento semi-supervisado (A) vs TFG consenso≥1 (B) ---")
    print("comparten = confirma al TFG | solo_A = NUEVAS | solo_B = TFG no captadas")
    filas = []
    for nombre in configuraciones:
        for metodo, m in mascaras[nombre].items():
            s = solapamiento(m, mascara_tfg)
            filas.append({
                'configuracion': nombre,
                'metodo': metodo,
                'n_semi': int(m.sum()),
                'confirma_TFG': s['comparten'],
                'nuevas': s['solo_A'],
                'TFG_no_captadas': s['solo_B'],
                'jaccard': s['jaccard'],
            })
    tabla_solap = pd.DataFrame(filas).set_index(['configuracion', 'metodo'])
    print(tabla_solap.to_string())

    # -----------------------------------------------------------------
    # (1) Visualizaciones PCA para la configuración principal
    # -----------------------------------------------------------------
    print("\n--- (1) Visualizaciones sobre la proyección PCA del TFG ---")
    mascara_princ = mascaras[config_principal][metodo_principal]
    _fig_pca_solapamiento(
        datos, mascara_princ, mascara_tfg,
        f'Semi-supervisado ({ETIQUETAS_METODO[metodo_principal]}) vs TFG '
        f'[{config_principal}]',
        '13_pca_semi_vs_tfg.png', dir_figuras)
    _fig_pca_metodos(datos, mascaras[config_principal], mascara_tfg,
                     '14_pca_metodos_semi.png', dir_figuras)

    # -----------------------------------------------------------------
    # (3) Exclusivas NUEVAS (consenso 0 o 1) del método principal
    # -----------------------------------------------------------------
    print("\n--- (3) Transacciones exclusivas NUEVAS (consenso 0 o 1) ---")
    # Recalculamos el método principal para recuperar la confianza (proba),
    # que usamos para ordenar y elegir los ejemplos más claros.
    info = construir_etiquetas(datos, random_state=random_state, verbose=False,
                               **configuraciones[config_principal])
    res = aplicar_self_training(datos, info, random_state=random_state,
                                verbose=False)
    mascara_semi = res['mascara_anom']
    proba = res['proba_anom']

    nuevas = mascara_semi & (consenso <= 1)       # descubrimientos del semi-sup.
    n_nuevas = int(nuevas.sum())
    n_c0 = int((mascara_semi & (consenso == 0)).sum())
    n_c1 = int((mascara_semi & (consenso == 1)).sum())
    print(f"Anomalías del método principal : {int(mascara_semi.sum())}")
    print(f"De ellas, NUEVAS (consenso 0/1): {n_nuevas}  "
          f"(consenso 0: {n_c0} | consenso 1: {n_c1})")

    if n_nuevas > 0:
        print("\nPerfil de las exclusivas nuevas frente a la media global:")
        print(perfil_anomalias(datos, nuevas).to_string())

        # 2-3 ejemplos concretos: los de mayor confianza del clasificador.
        idx_nuevas = np.where(nuevas)[0]
        idx_ordenados = idx_nuevas[np.argsort(proba[idx_nuevas])[::-1]]
        idx_ejemplos = idx_ordenados[:3]
        ejemplos = datos['df_raw'].loc[idx_ejemplos, COLUMNAS_EJEMPLO].copy()
        ejemplos['AmountBalanceRatio'] = datos['df'].loc[idx_ejemplos,
                                                         'AmountBalanceRatio'].round(3)
        ejemplos['consenso_TFG'] = consenso[idx_ejemplos]
        ejemplos['confianza_semi'] = proba[idx_ejemplos].round(3)
        print("\n3 ejemplos concretos de transacciones NUEVAS (mayor confianza):")
        print(ejemplos.to_string(index=False))

    # -----------------------------------------------------------------
    # Sensibilidad: nuevas descubiertas por configuración (método principal)
    # -----------------------------------------------------------------
    print("\n--- Sensibilidad: NUEVAS (consenso 0/1) por configuración "
          f"[{ETIQUETAS_METODO[metodo_principal]}] ---")
    filas_sens = []
    for nombre in configuraciones:
        m = mascaras[nombre][metodo_principal]
        filas_sens.append({
            'configuracion': nombre,
            'n_semi': int(m.sum()),
            'nuevas_c0': int((m & (consenso == 0)).sum()),
            'nuevas_c1': int((m & (consenso == 1)).sum()),
            'confirma_TFG': int((m & (consenso >= 2)).sum()),
        })
    print(pd.DataFrame(filas_sens).set_index('configuracion').to_string())

    return tabla_solap, mascaras


# =====================================================================
# 4.3 - INTEGRACIÓN DE PU-LEARNING EN EL ESTUDIO COMPARATIVO
# =====================================================================
def integrar_pu_en_estudio(datos, config_principal='B_consenso3_1a10',
                           random_state=42, dir_figuras='figuras'):
    """Integra PU-learning en la comparativa (marco perfil-primero):
    (1) solapamiento PU vs self-training (config B) y PU vs TFG (consenso≥1);
    (2) exclusivas nuevas de PU (consenso 0/1): perfil, nº y ejemplos;
    (3) comparación de los dos escenarios PU (≥4 vs ≥3);
    y añade PU al small-multiples de la PCA (figura 14).

    Escenario PU principal: umbral≥3 (10 positivos) = equivalente a la config B
    con la que se compara el resto de métodos semi-supervisados.
    """
    print("\n" + "=" * 70)
    print("4.3 - INTEGRACIÓN DE PU-LEARNING EN EL ESTUDIO COMPARATIVO")
    print("=" * 70)

    consenso = datos['consenso'].values
    mascara_tfg = consenso >= 1                      # anomalías del TFG (97)

    # Máscaras de los otros métodos para la config principal (y para la figura).
    mascaras_B = _calcular_todas_mascaras(
        datos, {config_principal: CONFIGURACIONES_SEMILLA[config_principal]},
        random_state)[config_principal]
    mascara_st = mascaras_B['ST']                    # self-training config B

    # PU: dos escenarios (≥3 principal, ≥4 secundario).
    pu3 = aplicar_pu_learning(datos, umbral_consenso_positivo=3,
                              random_state=random_state, verbose=False)
    pu4 = aplicar_pu_learning(datos, umbral_consenso_positivo=4,
                              random_state=random_state, verbose=False)
    mascara_pu = pu3['mascara_anom']
    print(f"\nEscenario PU principal: umbral≥3 ({pu3['n_pos']} positivos, "
          f"c={pu3['c']:.4f}) -> {pu3['n_anom']} anomalías")

    # --- PERFIL PRIMERO: recordatorio del perfil de PU principal ---
    print("\n--- Perfil de las anomalías de PU (perfil-primero) ---")
    print(perfil_anomalias(datos, mascara_pu).to_string())

    # -----------------------------------------------------------------
    # (1) Solapamiento
    # -----------------------------------------------------------------
    print("\n--- (1) Solapamiento (A = PU) ---")
    s_st = solapamiento(mascara_pu, mascara_st)
    s_tfg = solapamiento(mascara_pu, mascara_tfg)
    tabla = pd.DataFrame([
        {'comparacion': 'PU vs self-training (B)', 'n_PU': int(mascara_pu.sum()),
         'n_B': int(mascara_st.sum()), 'comparten': s_st['comparten'],
         'solo_PU': s_st['solo_A'], 'solo_otro': s_st['solo_B'],
         'jaccard': s_st['jaccard']},
        {'comparacion': 'PU vs TFG (consenso≥1)', 'n_PU': int(mascara_pu.sum()),
         'n_B': int(mascara_tfg.sum()), 'comparten': s_tfg['comparten'],
         'solo_PU': s_tfg['solo_A'], 'solo_otro': s_tfg['solo_B'],
         'jaccard': s_tfg['jaccard']},
    ]).set_index('comparacion')
    print(tabla.to_string())

    # -----------------------------------------------------------------
    # (3) Comparación de los dos escenarios PU (≥4 vs ≥3)
    # -----------------------------------------------------------------
    print("\n--- (3) Escenarios PU: umbral≥4 vs umbral≥3 ---")
    m4, m3 = pu4['mascara_anom'], pu3['mascara_anom']
    s43 = solapamiento(m4, m3)
    es_superconjunto = int((m4 & ~m3).sum()) == 0   # ¿todo ≥4 está en ≥3?
    print(f"  ≥4: {int(m4.sum())} anomalías | ≥3: {int(m3.sum())} anomalías")
    print(f"  comparten: {s43['comparten']} | solo ≥4: {s43['solo_A']} | "
          f"solo ≥3: {s43['solo_B']} | Jaccard: {s43['jaccard']}")
    print(f"  ¿≥3 es SUPERCONJUNTO de ≥4? {es_superconjunto} "
          f"(anomalías de ≥4 fuera de ≥3: {int((m4 & ~m3).sum())})")

    # -----------------------------------------------------------------
    # (2) Exclusivas NUEVAS de PU (consenso 0/1) - perfil primero
    # -----------------------------------------------------------------
    print("\n--- (2) Exclusivas NUEVAS de PU (consenso 0 o 1) ---")
    nuevas = mascara_pu & (consenso <= 1)
    n_nuevas = int(nuevas.sum())
    n_c0 = int((mascara_pu & (consenso == 0)).sum())
    n_c1 = int((mascara_pu & (consenso == 1)).sum())
    if n_nuevas > 0:
        print("Perfil de las exclusivas nuevas (perfil-primero):")
        print(perfil_anomalias(datos, nuevas).to_string())
    print(f"\nNuevas (consenso 0/1): {n_nuevas}  (consenso 0: {n_c0} | "
          f"consenso 1: {n_c1})")
    if n_nuevas > 0:
        # Ejemplos: mayor puntuación w de PU.
        w = pu3['w']
        idx_nuevas = np.where(nuevas)[0]
        idx_ej = idx_nuevas[np.argsort(w[idx_nuevas])[::-1]][:3]
        ejemplos = datos['df_raw'].loc[idx_ej, COLUMNAS_EJEMPLO].copy()
        ejemplos['AmountBalanceRatio'] = datos['df'].loc[
            idx_ej, 'AmountBalanceRatio'].round(3)
        ejemplos['consenso_TFG'] = consenso[idx_ej]
        ejemplos['w_PU'] = w[idx_ej].round(3)
        print("\n3 ejemplos concretos (mayor w de PU):")
        print(ejemplos.to_string(index=False))

    # -----------------------------------------------------------------
    # Figura: small-multiples de la PCA INCLUYENDO PU (actualiza fig. 14)
    # -----------------------------------------------------------------
    print("\n--- Figura: small-multiples PCA con PU incluido ---")
    mascaras_fig = dict(mascaras_B)
    mascaras_fig['PU'] = mascara_pu
    _fig_pca_metodos(datos, mascaras_fig, mascara_tfg,
                     '14_pca_metodos_semi.png', dir_figuras)

    return {'pu3': pu3, 'pu4': pu4, 'solap_st': s_st, 'solap_tfg': s_tfg,
            'nuevas': nuevas}


# =====================================================================
# EXPORTACIÓN DE RESULTADOS DEL COMPLEMENTO
# =====================================================================
def exportar_resultados_complemento(datos, config='B_consenso3_1a10',
                                    random_state=42,
                                    ruta='resultados_complemento.csv'):
    """Exporta, por transacción, las etiquetas del complemento semi-supervisado
    (configuración principal, self-training) al estilo de resultados_pipeline.csv.

    Columnas:
        TransactionID           identificador
        consenso_TFG            nº de detectores del TFG que la marcaron (0-4)
        etiqueta_semilla        etiqueta semilla usada (-1 sin etiquetar / 0 / 1)
        prediccion_selftraining veredicto final del self-training (0/1)
        nueva                   1 si es NUEVA (predicha anómala con consenso 0/1)
        prediccion_pu           1 si PU-learning la marca anómala (w>=0.5), 0 si no
        w_pu                    puntuación continua w(x)=g(x)/c de PU (ranking)
    """
    consenso = datos['consenso'].values
    info = construir_etiquetas(datos, random_state=random_state, verbose=False,
                               **CONFIGURACIONES_SEMILLA[config])
    res = aplicar_self_training(datos, info, random_state=random_state,
                                verbose=False)
    pred = res['mascara_anom'].astype(int)
    nueva = ((res['mascara_anom']) & (consenso <= 1)).astype(int)

    salida = pd.DataFrame({
        'TransactionID': datos['transaction_id'].values,
        'consenso_TFG': consenso,
        'etiqueta_semilla': info['y'],
        'prediccion_selftraining': pred,
        'nueva': nueva,
    })

    # --- Columnas de PU-learning (escenario principal: umbral>=3, 10 positivos,
    # el mismo que en el 4.3). Se añaden al FINAL mediante cruce por
    # TransactionID, sin tocar ni reordenar las columnas anteriores. ---
    pu = aplicar_pu_learning(datos, umbral_consenso_positivo=3,
                             random_state=random_state, verbose=False)
    pu_df = pd.DataFrame({
        'TransactionID': datos['transaction_id'].values,
        'prediccion_pu': pu['mascara_anom'].astype(int),
        'w_pu': np.round(pu['w'], 3),
    })
    salida = salida.merge(pu_df, on='TransactionID', how='left')

    salida.to_csv(ruta, index=False)
    print(f"Resultados del complemento exportados a '{ruta}' "
          f"(config {config}, {len(salida)} transacciones, "
          f"{int(pred.sum())} anómalas ST, {int(nueva.sum())} nuevas ST, "
          f"{int(pu['mascara_anom'].sum())} anómalas PU)")
    return salida


if __name__ == '__main__':
    datos = preparar_datos()
    print("\nFase 0 completada. Estructura de datos lista para la Fase 1.")

    tabla_semilla, etiquetas = resumen_configuraciones(datos)
    print("\nFase 1 completada. Etiquetas semilla listas para la Fase 2.")

    tabla_st, resultados_st = estudio_self_training(datos)
    sensibilidad_muestreo_negativos(datos, CONFIGURACIONES_SEMILLA['B_consenso3_1a10'])
    print("\nFase 2 completada.")

    conteo, solap, mascaras = estudio_propagacion(datos)
    print("\nFase 3 completada.")

    resultados_pu = estudio_pu_learning(datos)
    print("\nFase 4 completada.")

    tabla_solap, mascaras_todas = estudio_comparativo_tfg(datos)
    integrar_pu_en_estudio(datos)
    print("\nFase 5 completada.")

    exportar_resultados_complemento(datos)
    print("\nComplemento finalizado.")
