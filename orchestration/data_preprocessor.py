"""
Pré-processamento (Camada 2) com a mesma engenharia de features validada na PoC.

- Detecta o esquema da base (Credit Card, PaySim ou genérico) e a coluna-alvo (Class, isFraud, ...).
- O scaler e as medianas de imputação são ajustados SOMENTE no treino (sem vazamento de dados).
- Na inferência, as colunas são alinhadas às features do treino: colunas faltantes geram erro explícito.
- O estado é serializável em tipos primitivos para ser salvo com segurança no checkpoint do modelo.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

TARGET_CANDIDATES = ("Class", "isFraud", "is_fraud", "fraud", "label", "target")
PAYSIM_TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")
PAYSIM_COLUMNS = {"type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"}
PAYSIM_BALANCES = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
PAYSIM_DROP = ["nameOrig", "nameDest", "isFlaggedFraud"]
CREDITCARD_COLUMNS = {"V1", "V2", "V3", "Amount"}

SCHEMA_LABELS = {
    "creditcard": "Cartão de crédito (V1..V28, Amount)",
    "paysim": "Transferências móveis (PaySim)",
    "generic": "Genérico (colunas numéricas)",
}


def clean_raw_dataframe(df):
    """Corrige CSVs lidos numa única coluna e remove aspas/espaços de nomes e valores."""
    df = df.copy()
    if len(df.columns) == 1:
        column = df.columns[0]
        if df[column].astype(str).str.contains(",", regex=False).any():
            expanded = df[column].astype(str).str.split(",", expand=True)
            names = str(column).split(",")
            if len(names) == expanded.shape[1]:
                expanded.columns = names
                df = expanded
    df.columns = [str(c).strip().replace('"', "").replace("'", "") for c in df.columns]
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].astype(str).str.strip(" \"'")
    return df


def detect_target(df):
    for name in TARGET_CANDIDATES:
        if name in df.columns:
            return name
    return None


def detect_schema(df):
    columns = set(df.columns)
    if PAYSIM_COLUMNS <= columns:
        return "paysim"
    if CREDITCARD_COLUMNS <= columns:
        return "creditcard"
    return "generic"


def _numeric(series):
    return pd.to_numeric(series, errors="coerce")


class DataPreprocessor:
    def __init__(self):
        self.schema = None
        self.target = None
        self.feature_names = None
        self.medians = None
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

    # ----- engenharia de features -----
    def _engineer(self, df):
        if self.schema == "creditcard":
            out = df.drop(columns=[c for c in ["Time", self.target] if c and c in df.columns]).apply(_numeric)
            out["amount_log"] = np.log1p(out.pop("Amount").clip(lower=0))
            return out

        if self.schema == "paysim":
            out = pd.DataFrame(index=df.index)
            if "step" in df.columns:
                out["step"] = _numeric(df["step"])
            balances = {c: _numeric(df[c]) for c in PAYSIM_BALANCES}
            out["orig_balance_zero"] = (balances["oldbalanceOrg"] == 0).astype(float)
            out["dest_balance_zero"] = (balances["oldbalanceDest"] == 0).astype(float)
            for c in PAYSIM_BALANCES:
                out[c + "_log"] = np.log1p(balances[c].clip(lower=0))
            out["errorBalanceOrig"] = balances["newbalanceOrig"] + balances["amount"] - balances["oldbalanceOrg"]
            out["errorBalanceDest"] = balances["oldbalanceDest"] + balances["amount"] - balances["newbalanceDest"]
            # Categorias fixas: as mesmas colunas no treino e na inferência, mesmo que algum tipo não apareça
            tipo = df["type"].astype(str).str.upper()
            for categoria in PAYSIM_TYPES[1:]:
                out[f"type_{categoria}"] = (tipo == categoria).astype(float)
            return out

        out = df.drop(columns=[self.target] if self.target in df.columns else []).apply(_numeric)
        return out.dropna(axis=1, how="all")

    def prepare(self, df, training=True):
        """Limpa e gera as features. No treino, detecta esquema/alvo e descarta linhas sem rótulo."""
        df = clean_raw_dataframe(df)
        if training:
            self.target = detect_target(df)
            self.schema = detect_schema(df)
            if self.target is not None:
                df = df[_numeric(df[self.target]).notna()]

        y = None
        if self.target is not None and self.target in df.columns:
            y = _numeric(df[self.target])
            if training:
                y = y.astype(int).clip(0, 1)

        features = self._engineer(df.drop(columns=[c for c in PAYSIM_DROP if c in df.columns]))
        features = features.replace([np.inf, -np.inf], np.nan)
        return df, features, y

    # ----- ajuste e transformação -----
    def fit(self, features):
        self.feature_names = list(features.columns)
        self.medians = features.median(numeric_only=True).fillna(0.0).to_dict()
        self.scaler.fit(features.fillna(self.medians).values)
        return self

    def transform(self, features):
        faltantes = [c for c in self.feature_names if c not in features.columns]
        if faltantes:
            raise ValueError(
                "O arquivo não tem as colunas usadas no treinamento: " + ", ".join(faltantes[:10])
                + (" ..." if len(faltantes) > 10 else "")
            )
        alinhado = features[self.feature_names].fillna(self.medians)
        return self.scaler.transform(alinhado.values).astype(np.float32)

    # ----- serialização (tipos primitivos, seguros para torch.load(weights_only=True)) -----
    def to_dict(self):
        return {
            "schema": self.schema,
            "target": self.target,
            "feature_names": list(self.feature_names),
            "medians": {k: float(v) for k, v in self.medians.items()},
            "data_min": [float(v) for v in self.scaler.data_min_],
            "data_max": [float(v) for v in self.scaler.data_max_],
        }

    @classmethod
    def from_dict(cls, state):
        pre = cls()
        pre.schema = state["schema"]
        pre.target = state["target"]
        pre.feature_names = list(state["feature_names"])
        pre.medians = dict(state["medians"])
        limites = np.array([state["data_min"], state["data_max"]], dtype=np.float64)
        pre.scaler.fit(limites)
        return pre


def balance_sample(features, y, normals_per_fraud, seed):
    """Mantém todas as fraudes e até normals_per_fraud normais por fraude (amostragem do Teste 4)."""
    if not normals_per_fraud or y is None or y.sum() == 0:
        return features, y
    fraudes = y[y == 1].index
    normais = y[y == 0].index
    n = min(len(fraudes) * normals_per_fraud, len(normais))
    escolhidos = pd.Index(normais.to_series().sample(n=n, random_state=seed)).append(fraudes)
    escolhidos = escolhidos.to_series().sample(frac=1, random_state=seed).index
    return features.loc[escolhidos], y.loc[escolhidos]


def split_train_val_test(features, y, seed):
    """Split 60/20/20, estratificado quando há fraudes suficientes."""
    estratificar = y is not None and y.sum() >= 5 and (y == 0).sum() >= 5
    alvo = y if y is not None else pd.Series(0, index=features.index)
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, alvo, test_size=0.2, random_state=seed, stratify=alvo if estratificar else None
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=seed, stratify=y_temp if estratificar else None
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
