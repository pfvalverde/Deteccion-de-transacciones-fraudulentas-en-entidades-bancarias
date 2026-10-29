# -*- coding: utf-8 -*-
"""
Pipeline de clustering y detección de anomalías para la detección de
transacciones fraudulentas en entidades bancarias.

TFG - Grado en Ingeniería Informática, Universidad de Almería.

La clase PipelineFraude implementa el flujo completo del trabajo siguiendo
la estructura del índice de la memoria:

    Capítulo 3 (Metodología):
        - cargar_datos()              -> 3.1  Explicación del dataset
        - analisis_exploratorio()     -> 3.2.1 Análisis exploratorio (EDA)
        - preprocesar()               -> 3.2.2 Limpieza e ingeniería de características
                                         3.2.3 Codificación y escalado

    Capítulo 4 (Desarrollo experimental):
        - seleccionar_k() + aplicar_kmeans()      -> 4.1 Segmentación con K-Means
        - estimar_eps() + aplicar_dbscan()        -> 4.2 Segmentación con DBSCAN
        - aplicar_isolation_forest()              -> 4.3 Isolation Forest
        - aplicar_lof()                           -> 4.4 Local Outlier Factor
        - aplicar_zscore()                        -> 4.5 Z-Score

    Capítulo 5 (Comparativa y validación):
        - comparar_metodos()          -> 5.1 / 5.2 Comparativa y discusión

Todas las gráficas se guardan automáticamente en el directorio `figuras/`
con numeración correlativa, listas para insertarlas en la memoria.

Referencia metodológica principal:
    Müller, A. C. & Guido, S. (2017). "Introduction to Machine Learning
    with Python". O'Reilly. (Capítulo 3: Unsupervised Learning).
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


class PipelineFraude:
    """Pipeline no supervisado para la detección de transacciones anómalas.

    Parámetros
    ----------
    ruta_csv : str
        Ruta al fichero CSV con las transacciones.
    dir_figuras : str
        Directorio donde se guardan las figuras generadas.
    random_state : int
        Semilla para garantizar la reproducibilidad de los experimentos.
    mostrar_graficas : bool
        Si True, además de guardar cada figura la muestra en pantalla.
    """

    def __init__(self, ruta_csv, dir_figuras='figuras', random_state=42,
                 mostrar_graficas=False):
        self.ruta_csv = ruta_csv
        self.dir_figuras = dir_figuras
        self.random_state = random_state
        self.mostrar_graficas = mostrar_graficas
        os.makedirs(dir_figuras, exist_ok=True)

        # Estado interno que se va rellenando a lo largo del pipeline
        self.df_raw = None            # datos originales sin tocar
        self.df_features = None       # características finales SIN escalar (para perfilado)
        self.X_scaled = None          # matriz escalada usada por los algoritmos
        self.feature_names = None
        self.columnas_continuas = None  # subconjunto de features continuas (para Z-Score y perfiles)
        self.resultados = None        # DataFrame con las etiquetas/flags de cada método
        self.pca_2d = None            # proyección 2D para visualización

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------
    def _guardar_figura(self, nombre):
        """Guarda la figura actual en dir_figuras y la cierra."""
        ruta = os.path.join(self.dir_figuras, nombre)
        plt.tight_layout()
        plt.savefig(ruta, dpi=150)
        if self.mostrar_graficas:
            plt.show()
        plt.close('all')
        print(f"   [Figura guardada: {ruta}]")

    def _perfil_anomalias(self, mascara, nombre_metodo):
        """Compara la media de las features continuas de las anomalías
        frente a la media global. El ratio > 1 indica que la característica
        está sobre-representada en las anomalías (clave para la discusión
        del capítulo 5 de la memoria)."""
        continuas = self.df_features[self.columnas_continuas]
        perfil = pd.DataFrame({
            'media_global': continuas.mean(),
            'media_anomalias': continuas[mascara].mean(),
        })
        perfil['ratio'] = perfil['media_anomalias'] / perfil['media_global']
        print(f"\nPerfil de las {int(mascara.sum())} anomalías detectadas por {nombre_metodo}")
        print("(ratio > 1: la característica es mayor de lo normal en las anomalías)")
        print(perfil.round(3).to_string())
        return perfil

    def _scatter_pca(self, colores, titulo, nombre_fichero, etiqueta_color=None,
                     mascara_anomalias=None):
        """Dibuja la proyección PCA 2D coloreada según `colores`. Si se pasa
        una máscara de anomalías, las resalta con una X roja."""
        plt.figure(figsize=(9, 7))
        sc = plt.scatter(self.pca_2d[:, 0], self.pca_2d[:, 1],
                         c=colores, cmap='viridis', s=12, alpha=0.6)
        if etiqueta_color:
            plt.colorbar(sc, label=etiqueta_color)
        if mascara_anomalias is not None and mascara_anomalias.sum() > 0:
            plt.scatter(self.pca_2d[mascara_anomalias, 0],
                        self.pca_2d[mascara_anomalias, 1],
                        c='red', marker='x', s=60, label='Anomalías')
            plt.legend()
        plt.title(titulo)
        plt.xlabel('Componente principal 1')
        plt.ylabel('Componente principal 2')
        self._guardar_figura(nombre_fichero)

    # ------------------------------------------------------------------
    # 3.1 - Carga del dataset
    # ------------------------------------------------------------------
    def cargar_datos(self):
        print("=" * 70)
        print("3.1 - CARGA DEL DATASET")
        print("=" * 70)
        self.df_raw = pd.read_csv(self.ruta_csv)
        self.resultados = pd.DataFrame(index=self.df_raw.index)
        print(f"Dimensiones: {self.df_raw.shape[0]} transacciones x {self.df_raw.shape[1]} columnas")
        print(f"Columnas: {self.df_raw.columns.tolist()}")
        return self.df_raw

    # ------------------------------------------------------------------
    # 3.2.1 - Análisis exploratorio de datos (EDA)
    # ------------------------------------------------------------------
    def analisis_exploratorio(self):
        print("\n" + "=" * 70)
        print("3.2.1 - ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
        print("=" * 70)
        df = self.df_raw

        print("\n--- Vista previa ---")
        print(df.head().to_string())

        print("\n--- Tipos de datos y valores no nulos ---")
        df.info()

        print("\n--- Calidad de los datos ---")
        print(f"Valores nulos por columna:\n{df.isnull().sum().to_string()}")
        print(f"\nFilas duplicadas: {df.duplicated().sum()}")
        print(f"Cuentas únicas (AccountID): {df['AccountID'].nunique()}")

        print("\n--- Resumen estadístico (variables numéricas) ---")
        print(df.describe().round(2).to_string())

        print("\n--- Resumen de variables categóricas ---")
        print(df.describe(include='object').to_string())

        # Histogramas de las variables numéricas
        numericas = df.select_dtypes(include=np.number).columns
        df[numericas].hist(bins=30, figsize=(14, 9), edgecolor='black')
        plt.suptitle('Distribución de las variables numéricas')
        self._guardar_figura('01_distribuciones_numericas.png')

        # Distribución de las variables categóricas principales
        categoricas = ['TransactionType', 'Channel', 'CustomerOccupation']
        fig, ejes = plt.subplots(1, len(categoricas), figsize=(15, 4))
        for eje, col in zip(ejes, categoricas):
            df[col].value_counts().plot(kind='bar', ax=eje, color='steelblue',
                                        edgecolor='black')
            eje.set_title(col)
            eje.set_ylabel('Nº de transacciones')
        self._guardar_figura('02_distribuciones_categoricas.png')

        # Matriz de correlación de las variables numéricas
        corr = df[numericas].corr()
        plt.figure(figsize=(8, 6))
        im = plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(im, label='Correlación de Pearson')
        plt.xticks(range(len(corr)), corr.columns, rotation=45, ha='right')
        plt.yticks(range(len(corr)), corr.columns)
        for i in range(len(corr)):
            for j in range(len(corr)):
                plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center',
                         va='center', fontsize=8)
        plt.title('Matriz de correlación')
        self._guardar_figura('03_matriz_correlacion.png')

        # Hallazgos del EDA relevantes para el preprocesamiento (sección 3.2.2)
        horas = pd.to_datetime(df['TransactionDate']).dt.hour
        fecha_prev = pd.to_datetime(df['PreviousTransactionDate'])
        print("\n--- Hallazgos relevantes del EDA ---")
        print(f"* Todas las transacciones ocurren entre las {horas.min()}h y las "
              f"{horas.max()}h -> una variable 'IsNight' tendría varianza nula.")
        print(f"* 'PreviousTransactionDate' está concentrada en un único instante "
              f"({fecha_prev.min()} a {fecha_prev.max()}), POSTERIOR a todas las "
              f"transacciones de 2023 -> es un artefacto del dataset sintético y "
              f"cualquier 'TimeDelta' derivado sería ruido sin significado.")

    # ------------------------------------------------------------------
    # 3.2.2 y 3.2.3 - Limpieza, ingeniería, codificación y escalado
    # ------------------------------------------------------------------
    def preprocesar(self):
        print("\n" + "=" * 70)
        print("3.2.2 / 3.2.3 - PREPROCESAMIENTO E INGENIERÍA DE CARACTERÍSTICAS")
        print("=" * 70)
        df = self.df_raw.copy()
        df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])

        # --- Ingeniería de características temporales ---
        print(" - Creando características temporales (Hour, DayOfWeek, IsWeekend)...")
        df['Hour'] = df['TransactionDate'].dt.hour
        df['DayOfWeek'] = df['TransactionDate'].dt.dayofweek  # Lunes=0 ... Domingo=6
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        # NOTA (auditoría): se descartan 'IsNight' (varianza nula: todas las
        # transacciones son de 16-18h) y 'TimeDelta' (PreviousTransactionDate
        # es un artefacto constante del dataset). Ver hallazgos del EDA.

        # --- Características de comportamiento ---
        print(" - Creando características de importe y comportamiento por cuenta...")
        # Importe relativo al saldo: gastar 500EUR con 100EUR de saldo medio es
        # mucho más sospechoso que con 15.000EUR.
        df['AmountBalanceRatio'] = df['TransactionAmount'] / (df['AccountBalance'] + 1)
        # Agregados por cuenta: contextualizan cada transacción dentro del
        # historial de su titular (aprovechamos AccountID antes de eliminarlo).
        df['TxPorCuenta'] = df.groupby('AccountID')['TransactionID'].transform('count')
        importe_medio_cuenta = df.groupby('AccountID')['TransactionAmount'].transform('mean')
        df['RatioImporteMedioCuenta'] = df['TransactionAmount'] / importe_medio_cuenta

        # --- Eliminación de columnas no informativas ---
        columnas_a_eliminar = [
            'TransactionID', 'AccountID', 'DeviceID', 'IP Address', 'MerchantID',
            'TransactionDate', 'PreviousTransactionDate',
            'Location',  # 43 ciudades: one-hot dispararía la dimensionalidad
        ]
        print(f" - Eliminando identificadores y columnas no informativas: {columnas_a_eliminar}")
        df = df.drop(columns=columnas_a_eliminar)

        # Features continuas: se usan para el Z-Score y para los perfiles.
        # (Excluimos las binarias/dummies: un Z-score sobre una variable 0/1
        # no tiene interpretación estadística válida.)
        self.columnas_continuas = [
            'TransactionAmount', 'CustomerAge', 'TransactionDuration',
            'LoginAttempts', 'AccountBalance', 'Hour', 'DayOfWeek',
            'AmountBalanceRatio', 'TxPorCuenta', 'RatioImporteMedioCuenta',
        ]

        # --- Codificación de variables categóricas ---
        print(" - Codificando variables categóricas (one-hot)...")
        df = pd.get_dummies(df, drop_first=True)
        columnas_bool = df.select_dtypes(include='bool').columns
        df[columnas_bool] = df[columnas_bool].astype(int)

        # --- Eliminación de columnas de varianza nula ---
        constantes = df.columns[df.std() == 0].tolist()
        if constantes:
            print(f" - Eliminando columnas constantes (varianza nula): {constantes}")
            df = df.drop(columns=constantes)

        self.df_features = df
        self.feature_names = df.columns.tolist()

        # --- Escalado ---
        print(" - Escalando con StandardScaler (media 0, desviación 1)...")
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(df)

        # --- Proyección PCA 2D para visualización ---
        pca = PCA(n_components=2, random_state=self.random_state)
        self.pca_2d = pca.fit_transform(self.X_scaled)
        var_explicada = pca.explained_variance_ratio_
        print(f" - PCA 2D para visualización: varianza explicada "
              f"{var_explicada[0]:.1%} + {var_explicada[1]:.1%} = {var_explicada.sum():.1%}")

        print(f"\nPreprocesamiento completo. Matriz final: {self.X_scaled.shape}")
        print(f"Características finales: {self.feature_names}")
        return self.X_scaled

    # ------------------------------------------------------------------
    # 4.1 - K-Means: selección de k (codo + silueta) y segmentación
    # ------------------------------------------------------------------
    def seleccionar_k(self, k_min=2, k_max=10):
        """Combina el método del codo (inercia) con el coeficiente de silueta.
        El codo es subjetivo; la silueta da un criterio cuantitativo para
        justificar la elección de k en la memoria."""
        print("\n" + "=" * 70)
        print("4.1a - SELECCIÓN DEL NÚMERO DE CLUSTERS (CODO + SILUETA)")
        print("=" * 70)
        rango_k = range(k_min, k_max + 1)
        inercias, siluetas = [], []
        for k in rango_k:
            km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
            etiquetas = km.fit_predict(self.X_scaled)
            sil = silhouette_score(self.X_scaled, etiquetas)
            inercias.append(km.inertia_)
            siluetas.append(sil)
            print(f"  k={k}: inercia={km.inertia_:.0f} | silueta={sil:.4f}")

        fig, (eje1, eje2) = plt.subplots(1, 2, figsize=(14, 5))
        eje1.plot(rango_k, inercias, marker='o', linestyle='--')
        eje1.set_title('Método del codo')
        eje1.set_xlabel('Número de clusters (k)')
        eje1.set_ylabel('Inercia (WCSS)')
        eje1.grid(True)
        eje2.plot(rango_k, siluetas, marker='o', linestyle='--', color='green')
        eje2.set_title('Coeficiente de silueta')
        eje2.set_xlabel('Número de clusters (k)')
        eje2.set_ylabel('Silueta media')
        eje2.grid(True)
        self._guardar_figura('04_codo_y_silueta.png')

        k_optimo = list(rango_k)[int(np.argmax(siluetas))]
        print(f"\nk óptimo según el coeficiente de silueta: k={k_optimo}")
        return k_optimo

    def aplicar_kmeans(self, k=None):
        print("\n" + "=" * 70)
        print("4.1b - SEGMENTACIÓN CON K-MEANS")
        print("=" * 70)
        if k is None:
            k = self.seleccionar_k()

        km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
        etiquetas = km.fit_predict(self.X_scaled)
        self.resultados['kmeans_cluster'] = etiquetas

        sil = silhouette_score(self.X_scaled, etiquetas)
        print(f"\nK-Means con k={k} | silueta={sil:.4f}")
        print("\n--- Tamaño de cada cluster ---")
        print(pd.Series(etiquetas).value_counts().sort_index().to_string())

        # Perfilado sobre los datos SIN escalar (interpretables) y SIN
        # contaminar el perfil con columnas de etiquetas de otros métodos.
        print("\n--- Perfil de los clusters (media por característica) ---")
        perfil = self.df_features.groupby(etiquetas).mean().T.round(2)
        perfil.columns = [f'Cluster {c}' for c in perfil.columns]
        print(perfil.to_string())

        self._scatter_pca(etiquetas, f'K-Means (k={k}) sobre proyección PCA 2D',
                          '05_kmeans_pca.png', etiqueta_color='Cluster')
        return etiquetas

    # ------------------------------------------------------------------
    # 4.2 - DBSCAN: estimación de eps (gráfico k-distancias) y segmentación
    # ------------------------------------------------------------------
    def estimar_eps(self, min_samples=10):
        """Gráfico de k-distancias (distancia al k-ésimo vecino, ordenada).
        El 'codo' de la curva es el eps recomendado. Como heurística
        automática se toma el percentil 97.5 de las distancias, que deja
        aproximadamente el 2.5% de los puntos en zonas poco densas."""
        print("\n" + "=" * 70)
        print("4.2a - ESTIMACIÓN DE EPS PARA DBSCAN (K-DISTANCIAS)")
        print("=" * 70)
        vecinos = NearestNeighbors(n_neighbors=min_samples).fit(self.X_scaled)
        distancias, _ = vecinos.kneighbors(self.X_scaled)
        k_distancias = np.sort(distancias[:, -1])

        eps_sugerido = float(np.percentile(k_distancias, 97.5))

        plt.figure(figsize=(10, 6))
        plt.plot(k_distancias)
        plt.axhline(eps_sugerido, color='red', linestyle='--',
                    label=f'eps sugerido = {eps_sugerido:.2f} (percentil 97.5)')
        plt.title(f'Gráfico de k-distancias (k = min_samples = {min_samples})')
        plt.xlabel('Puntos ordenados por distancia')
        plt.ylabel(f'Distancia al vecino nº {min_samples}')
        plt.legend()
        plt.grid(True)
        self._guardar_figura('06_k_distancias_dbscan.png')

        print(f"eps sugerido: {eps_sugerido:.3f}")
        return eps_sugerido

    def aplicar_dbscan(self, eps=None, min_samples=10):
        print("\n" + "=" * 70)
        print("4.2b - SEGMENTACIÓN CON DBSCAN")
        print("=" * 70)
        if eps is None:
            eps = self.estimar_eps(min_samples)

        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        etiquetas = dbscan.fit_predict(self.X_scaled)
        self.resultados['dbscan_cluster'] = etiquetas
        mascara_ruido = etiquetas == -1
        self.resultados['dbscan_anomalia'] = mascara_ruido.astype(int)

        n_clusters = len(set(etiquetas)) - (1 if -1 in etiquetas else 0)
        n_ruido = int(mascara_ruido.sum())
        print(f"\nDBSCAN (eps={eps:.3f}, min_samples={min_samples})")
        print(f"Clusters encontrados: {n_clusters} | Puntos de ruido (anomalías): "
              f"{n_ruido} ({n_ruido / len(etiquetas):.2%})")
        print("\n--- Tamaño de cada cluster (-1 = ruido/anomalías) ---")
        print(pd.Series(etiquetas).value_counts().sort_index().to_string())

        # Silueta calculada solo sobre los puntos asignados a clusters
        if n_clusters > 1:
            sin_ruido = ~mascara_ruido
            sil = silhouette_score(self.X_scaled[sin_ruido], etiquetas[sin_ruido])
            print(f"Silueta (excluyendo ruido): {sil:.4f}")

        if n_ruido > 0:
            self._perfil_anomalias(pd.Series(mascara_ruido, index=self.df_features.index),
                                   'DBSCAN (ruido)')

        self._scatter_pca(etiquetas, 'DBSCAN sobre proyección PCA 2D',
                          '07_dbscan_pca.png', etiqueta_color='Cluster',
                          mascara_anomalias=mascara_ruido)
        return etiquetas

    # ------------------------------------------------------------------
    # 4.3 - Isolation Forest
    # ------------------------------------------------------------------
    def aplicar_isolation_forest(self, contamination=0.02):
        print("\n" + "=" * 70)
        print("4.3 - DETECCIÓN DE ANOMALÍAS CON ISOLATION FOREST")
        print("=" * 70)
        iso = IsolationForest(n_estimators=200, contamination=contamination,
                              random_state=self.random_state, n_jobs=-1)
        predicciones = iso.fit_predict(self.X_scaled)
        mascara = predicciones == -1
        self.resultados['iforest_anomalia'] = mascara.astype(int)
        # Score continuo: cuanto mayor, más anómala es la transacción.
        # Permite RANKING de sospecha, no solo etiqueta binaria.
        self.resultados['iforest_score'] = -iso.decision_function(self.X_scaled)

        print(f"\nIsolation Forest (contamination={contamination})")
        print(f"Anomalías detectadas: {int(mascara.sum())} ({mascara.mean():.2%})")

        plt.figure(figsize=(10, 6))
        plt.hist(self.resultados['iforest_score'], bins=60, color='steelblue',
                 edgecolor='black')
        umbral = self.resultados.loc[mascara, 'iforest_score'].min()
        plt.axvline(umbral, color='red', linestyle='--',
                    label=f'Umbral de anomalía ({umbral:.3f})')
        plt.title('Distribución de la puntuación de anomalía (Isolation Forest)')
        plt.xlabel('Puntuación de anomalía (mayor = más anómala)')
        plt.ylabel('Nº de transacciones')
        plt.legend()
        self._guardar_figura('08_isoforest_scores.png')

        self._perfil_anomalias(pd.Series(mascara, index=self.df_features.index),
                               'Isolation Forest')
        self._scatter_pca(self.resultados['iforest_score'],
                          'Puntuación de Isolation Forest sobre PCA 2D',
                          '09_isoforest_pca.png',
                          etiqueta_color='Puntuación de anomalía',
                          mascara_anomalias=mascara)
        return mascara.astype(int)

    # ------------------------------------------------------------------
    # 4.4 - Local Outlier Factor (LOF)
    # ------------------------------------------------------------------
    def aplicar_lof(self, n_neighbors=20, contamination=0.02):
        print("\n" + "=" * 70)
        print("4.4 - DETECCIÓN DE ANOMALÍAS CON LOCAL OUTLIER FACTOR (LOF)")
        print("=" * 70)
        lof = LocalOutlierFactor(n_neighbors=n_neighbors,
                                 contamination=contamination, n_jobs=-1)
        predicciones = lof.fit_predict(self.X_scaled)
        mascara = predicciones == -1
        self.resultados['lof_anomalia'] = mascara.astype(int)
        self.resultados['lof_score'] = -lof.negative_outlier_factor_

        print(f"\nLOF (n_neighbors={n_neighbors}, contamination={contamination})")
        print(f"Anomalías detectadas: {int(mascara.sum())} ({mascara.mean():.2%})")

        self._perfil_anomalias(pd.Series(mascara, index=self.df_features.index), 'LOF')
        self._scatter_pca(self.resultados['lof_score'],
                          'Factor de outlier local (LOF) sobre PCA 2D',
                          '10_lof_pca.png', etiqueta_color='Factor LOF',
                          mascara_anomalias=mascara)
        return mascara.astype(int)

    # ------------------------------------------------------------------
    # 4.5 - Z-Score
    # ------------------------------------------------------------------
    def aplicar_zscore(self, umbral_z=3.0, min_caracteristicas=2):
        """Marca como anomalía toda transacción con al menos
        `min_caracteristicas` features continuas cuyo |z| > umbral_z.
        Se aplica SOLO sobre las variables continuas: un z-score sobre una
        variable binaria (dummy) carece de interpretación válida."""
        print("\n" + "=" * 70)
        print("4.5 - DETECCIÓN DE ANOMALÍAS MEDIANTE Z-SCORE")
        print("=" * 70)
        continuas = self.df_features[self.columnas_continuas]
        z = (continuas - continuas.mean()) / continuas.std()
        num_features_anomalas = (z.abs() > umbral_z).sum(axis=1)
        mascara = num_features_anomalas >= min_caracteristicas
        self.resultados['zscore_num_features'] = num_features_anomalas
        self.resultados['zscore_anomalia'] = mascara.astype(int)

        print(f"\nZ-Score (|z| > {umbral_z} en al menos {min_caracteristicas} "
              f"características continuas)")
        print(f"Anomalías detectadas: {int(mascara.sum())} ({mascara.mean():.2%})")
        print("\nDistribución del nº de características anómalas por transacción:")
        print(num_features_anomalas.value_counts().sort_index().to_string())

        if mascara.sum() > 0:
            self._perfil_anomalias(mascara, 'Z-Score')
        return mascara.astype(int)

    # ------------------------------------------------------------------
    # 5 - Comparativa entre métodos y consenso
    # ------------------------------------------------------------------
    def comparar_metodos(self):
        print("\n" + "=" * 70)
        print("5 - COMPARATIVA ENTRE MÉTODOS Y ANÁLISIS DE CONSENSO")
        print("=" * 70)
        metodos = {
            'DBSCAN': 'dbscan_anomalia',
            'Isolation Forest': 'iforest_anomalia',
            'LOF': 'lof_anomalia',
            'Z-Score': 'zscore_anomalia',
        }
        flags = self.resultados[list(metodos.values())]
        flags.columns = list(metodos.keys())

        print("\n--- Nº de anomalías detectadas por cada método ---")
        print(flags.sum().to_string())

        # Matriz de similitud de Jaccard entre los conjuntos de anomalías:
        # J(A,B) = |A ∩ B| / |A ∪ B|. Mide cuánto coinciden dos métodos.
        nombres = list(metodos.keys())
        jaccard = pd.DataFrame(index=nombres, columns=nombres, dtype=float)
        for a in nombres:
            for b in nombres:
                interseccion = ((flags[a] == 1) & (flags[b] == 1)).sum()
                union = ((flags[a] == 1) | (flags[b] == 1)).sum()
                jaccard.loc[a, b] = interseccion / union if union > 0 else np.nan
        print("\n--- Matriz de similitud de Jaccard entre métodos ---")
        print(jaccard.round(3).to_string())

        plt.figure(figsize=(7, 6))
        im = plt.imshow(jaccard.values.astype(float), cmap='Blues', vmin=0, vmax=1)
        plt.colorbar(im, label='Índice de Jaccard')
        plt.xticks(range(len(nombres)), nombres, rotation=45, ha='right')
        plt.yticks(range(len(nombres)), nombres)
        for i in range(len(nombres)):
            for j in range(len(nombres)):
                plt.text(j, i, f"{jaccard.iloc[i, j]:.2f}", ha='center', va='center')
        plt.title('Coincidencia entre métodos (Jaccard)')
        self._guardar_figura('11_jaccard_metodos.png')

        # Consenso: en cuántos métodos aparece cada transacción como anómala
        consenso = flags.sum(axis=1)
        self.resultados['consenso'] = consenso
        print("\n--- Distribución del consenso (nº de métodos que marcan cada transacción) ---")
        print(consenso.value_counts().sort_index().to_string())

        # Transacciones de máxima sospecha: marcadas por >= 3 métodos.
        # Se recuperan los datos ORIGINALES para que sean interpretables.
        mascara_consenso = consenso >= 3
        print(f"\n--- Transacciones marcadas por 3 o más métodos: {int(mascara_consenso.sum())} ---")
        if mascara_consenso.sum() > 0:
            columnas_interes = ['TransactionID', 'TransactionAmount', 'TransactionType',
                                'Channel', 'CustomerAge', 'CustomerOccupation',
                                'LoginAttempts', 'AccountBalance']
            sospechosas = self.df_raw.loc[mascara_consenso, columnas_interes].copy()
            sospechosas['MetodosQueLaMarcan'] = consenso[mascara_consenso]
            sospechosas = sospechosas.sort_values('MetodosQueLaMarcan', ascending=False)
            print(sospechosas.to_string(index=False))

        self._scatter_pca(consenso, 'Consenso entre métodos sobre PCA 2D',
                          '12_consenso_pca.png',
                          etiqueta_color='Nº de métodos que marcan la transacción',
                          mascara_anomalias=mascara_consenso.values)
        return flags, jaccard, consenso

    # ------------------------------------------------------------------
    # Ejecución completa del pipeline
    # ------------------------------------------------------------------
    def ejecutar(self, k=None, eps=None, min_samples=10, contamination=0.02,
                 n_neighbors_lof=20, umbral_z=3.0, min_caracteristicas_z=2):
        """Ejecuta todas las fases en el orden del índice de la memoria y
        devuelve el DataFrame de resultados (una fila por transacción con
        las etiquetas y puntuaciones de todos los métodos)."""
        self.cargar_datos()
        self.analisis_exploratorio()
        self.preprocesar()
        self.aplicar_kmeans(k=k)
        self.aplicar_dbscan(eps=eps, min_samples=min_samples)
        self.aplicar_isolation_forest(contamination=contamination)
        self.aplicar_lof(n_neighbors=n_neighbors_lof, contamination=contamination)
        self.aplicar_zscore(umbral_z=umbral_z, min_caracteristicas=min_caracteristicas_z)
        self.comparar_metodos()

        # Exportación de resultados para análisis posterior en la memoria
        salida = pd.concat([self.df_raw[['TransactionID', 'AccountID']],
                            self.resultados], axis=1)
        ruta_salida = 'resultados_pipeline.csv'
        salida.to_csv(ruta_salida, index=False)
        print(f"\nResultados completos exportados a '{ruta_salida}'")
        print("Pipeline finalizado.")
        return salida


if __name__ == '__main__':
    pipeline = PipelineFraude('bank_transactions_data_2.csv')
    resultados = pipeline.ejecutar()
