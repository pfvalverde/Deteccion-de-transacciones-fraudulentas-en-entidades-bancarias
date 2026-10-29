# Datos

Los experimentos utilizan el conjunto de datos público **Bank Transaction Dataset
for Fraud Detection**, de Valakhorasani (Kaggle):

**https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection**

El fichero **no se incluye en este repositorio** (se respeta la fuente original).
Para reproducir los experimentos:

1. Descarga el dataset desde el enlace anterior.
2. Renombra/copia el fichero CSV como **`bank_transactions_data_2.csv`**.
3. Colócalo en la **raíz del repositorio** (al lado de `pipeline_fraude.py`),
   que es donde los scripts lo buscan por defecto.

## Descripción del conjunto

- **2 512** transacciones (filas) × **16** variables (columnas).
- No contiene ninguna etiqueta de fraude: es un escenario **no supervisado**
  (de ahí el enfoque del TFG). El complemento genera "etiquetas semilla"
  a partir del consenso de los detectores del TFG.

| Columna | Descripción |
|---|---|
| `TransactionID` | Identificador de la transacción |
| `AccountID` | Identificador de la cuenta |
| `TransactionAmount` | Importe de la transacción |
| `TransactionDate` | Fecha/hora de la transacción |
| `TransactionType` | Tipo (p. ej. débito/crédito) |
| `Location` | Localización |
| `DeviceID` | Identificador del dispositivo |
| `IP Address` | Dirección IP |
| `MerchantID` | Identificador del comercio |
| `Channel` | Canal (p. ej. online, cajero, sucursal) |
| `CustomerAge` | Edad del titular |
| `CustomerOccupation` | Ocupación del titular |
| `TransactionDuration` | Duración de la transacción |
| `LoginAttempts` | Número de intentos de acceso |
| `AccountBalance` | Saldo de la cuenta |
| `PreviousTransactionDate` | Fecha de la transacción anterior |
