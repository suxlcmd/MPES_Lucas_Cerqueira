"""
Pré-processamento (Camada 2) com a mesma engenharia de features validada na PoC.

- Detecta o esquema da base (Credit Card, PaySim ou genérico) e a coluna que indica fraude, inclusive com
  rótulos em texto (Y/N, Sim/Não, True/False...), para que o usuário não precise preparar a base.
- Esquema genérico: colunas numéricas (log nas de cauda longa), colunas de texto com poucas categorias
  (One-Hot) e descarte automático de identificadores, datas e textos livres.
- O scaler, as medianas e o plano de colunas são ajustados SOMENTE no treino (sem vazamento de dados).
- Na inferência, as colunas são alinhadas às do treino: colunas faltantes geram erro explícito.
- O estado é serializável em tipos primitivos para ser salvo com segurança no checkpoint do modelo.
"""

import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Nomes comuns da coluna de fraude (comparados sem diferenciar maiúsculas, espaços e '_')
TARGET_CANDIDATES = (
    "Class", "isFraud", "is_fraud", "Is_Fraud", "fraud_bool", "fraud_reported", "FraudFound_P", "FraudFound",
    "Is Fraudulent", "fraudulent", "fraude", "fraud", "label", "target",
)
POSITIVE_LABELS = {"1", "1.0", "y", "yes", "s", "sim", "true", "t", "fraud", "fraude", "fraudulent", "fraudulenta"}
NEGATIVE_LABELS = {"0", "0.0", "n", "no", "nao", "não", "false", "f", "legit", "legitimate", "legitima", "legítima",
                   "normal", "ok"}

PAYSIM_TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")
PAYSIM_COLUMNS = {"type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"}
PAYSIM_BALANCES = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
PAYSIM_DROP = ["nameOrig", "nameDest", "isFlaggedFraud"]
CREDITCARD_COLUMNS = {"V1", "V2", "V3", "Amount"}

MAX_CATEGORIES = 30            # texto com mais categorias que isso é tratado como texto livre e descartado
_ID_TOKENS = {"id", "ids", "number", "numero", "número", "num", "nr", "codigo", "código", "code", "cpf", "cnpj",
              "zip", "cep", "uuid", "key"}
_DATE_NAME = re.compile(r"(date|data|dt_|_dt|timestamp)", re.IGNORECASE)

SCHEMA_LABELS = {
    "creditcard": "Cartão de crédito (V1..V28, Amount)",
    "paysim": "Transferências móveis (PaySim)",
    "generic": "Genérico (colunas numéricas e categóricas)",
}


def _normalize_name(name):
    return re.sub(r"[\s_\-]", "", str(name)).lower()


def _id_like_name(name):
    """'policy_number', 'PolicyNumber', 'Transaction ID', 'cpf_cliente' -> True."""
    palavras = re.split(r"[^a-z0-9áéíóúâêôãõç]+", re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name)).lower())
    return any(p in _ID_TOKENS for p in palavras if p)


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
        if not pd.api.types.is_numeric_dtype(df[column]) and not pd.api.types.is_bool_dtype(df[column]):
            df[column] = df[column].astype(str).str.strip(" \"'").replace({"nan": np.nan, "None": np.nan, "": np.nan})
    return df


def detect_target(df):
    """Coluna que indica fraude: primeiro o nome exato, depois sem diferenciar maiúsculas/espaços/'_'."""
    for name in TARGET_CANDIDATES:
        if name in df.columns:
            return name
    normalizados = {_normalize_name(c): c for c in df.columns}
    for name in TARGET_CANDIDATES:
        if _normalize_name(name) in normalizados:
            return normalizados[_normalize_name(name)]
    return None


def encode_target(series):
    """Rótulo -> 1 (fraude), 0 (legítima) ou NaN (desconhecido). Aceita 0/1, Y/N, Sim/Não, True/False..."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    numerico = pd.to_numeric(series, errors="coerce")
    if numerico.notna().sum() >= 0.95 * series.notna().sum() and series.notna().any():
        return numerico.where(numerico.isin([0, 1]))
    texto = series.astype(str).str.strip().str.lower()
    return pd.Series(np.select([texto.isin(POSITIVE_LABELS), texto.isin(NEGATIVE_LABELS)], [1.0, 0.0], np.nan),
                     index=series.index).where(series.notna())


def label_candidates(df):
    """Colunas que podem indicar fraude (exatamente dois valores), para o usuário escolher se preciso."""
    candidatas = []
    for coluna in df.columns:
        valores = df[coluna].dropna()
        if valores.nunique() == 2 and encode_target(valores).notna().all():
            candidatas.append(coluna)
    return candidatas


def detect_schema(df):
    columns = set(df.columns)
    if PAYSIM_COLUMNS <= columns:
        return "paysim"
    if CREDITCARD_COLUMNS <= columns:
        return "creditcard"
    return "generic"


def _numeric(series):
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    return pd.to_numeric(series, errors="coerce")


def _looks_like_date(name, series):
    if _DATE_NAME.search(str(name)):
        return True
    amostra = series.dropna().astype(str).head(200)
    if amostra.empty or not amostra.str.contains(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}").mean() > 0.9:
        return False
    return pd.to_datetime(amostra, errors="coerce", format="mixed").notna().mean() > 0.9


def plan_generic_columns(df, target):
    """
    Decide o uso de cada coluna de uma base genérica (só no treino):
    numérica, numérica com log (cauda longa), categórica (One-Hot) ou descartada (ID, data, texto livre, constante).
    """
    plano = {"numeric": [], "log": [], "categorical": {}, "dropped": {}}
    n = len(df)
    for coluna in df.columns:
        if coluna == target:
            continue
        serie = df[coluna]
        valores = serie.dropna()
        if valores.nunique() <= 1:
            plano["dropped"][coluna] = "constante"
            continue
        numerico = _numeric(serie)
        e_numerica = numerico.notna().sum() >= 0.95 * max(valores.size, 1)
        nome_de_id = _id_like_name(coluna)
        if e_numerica:
            inteiro = np.allclose(numerico.dropna() % 1, 0)
            if nome_de_id and inteiro and numerico.nunique() >= 0.9 * n:
                plano["dropped"][coluna] = "identificador"
            elif valores.nunique() <= 2 and not set(numerico.dropna().unique()) <= {0, 1}:
                plano["categorical"][coluna] = sorted(valores.astype(str).unique().tolist())
            else:
                plano["numeric"].append(coluna)
                positivo = numerico.dropna()
                if (positivo >= 0).all() and positivo.max() > 100 and positivo.skew() > 2:
                    plano["log"].append(coluna)
            continue
        if _looks_like_date(coluna, serie):
            plano["dropped"][coluna] = "data"
        elif valores.nunique() <= MAX_CATEGORIES:
            plano["categorical"][coluna] = sorted(valores.astype(str).unique().tolist())
        elif nome_de_id or valores.nunique() >= 0.5 * n:
            plano["dropped"][coluna] = "identificador ou texto único por linha"
        else:
            plano["dropped"][coluna] = f"texto com muitas categorias ({valores.nunique()})"
    return plano


class DataPreprocessor:
    def __init__(self):
        self.schema = None
        self.target = None
        self.feature_names = None
        self.medians = None
        self.plan = None
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

        if self.plan is None:  # checkpoints antigos (só colunas numéricas)
            out = df.drop(columns=[self.target] if self.target in df.columns else []).apply(_numeric)
            return out.dropna(axis=1, how="all")

        usadas = self.plan["numeric"] + list(self.plan["categorical"])
        faltantes = [c for c in usadas if c not in df.columns]
        if faltantes:
            raise ValueError("O arquivo não tem as colunas usadas no treinamento: " + ", ".join(faltantes[:10])
                             + (" ..." if len(faltantes) > 10 else ""))
        colunas = {}
        for c in self.plan["numeric"]:
            valores = _numeric(df[c])
            colunas[c + "_log" if c in self.plan["log"] else c] = (
                np.log1p(valores.clip(lower=0)) if c in self.plan["log"] else valores)
        for c, categorias in self.plan["categorical"].items():
            texto = df[c].astype(str)
            for categoria in categorias[1:]:  # a primeira categoria é a referência (como drop_first)
                colunas[f"{c}={categoria}"] = (texto == categoria).astype(float)
        return pd.DataFrame(colunas, index=df.index)

    def prepare(self, df, training=True, target=None):
        """
        Limpa e gera as features. No treino, detecta esquema/alvo (ou usa `target`, escolhido pelo usuário;
        target="" indica base sem rótulo) e descarta linhas sem rótulo.
        """
        df = clean_raw_dataframe(df)
        if training:
            self.target = detect_target(df) if target is None else (target or None)
            if self.target is not None and self.target not in df.columns:
                raise ValueError(f"A coluna '{self.target}' não existe no arquivo.")
            self.schema = detect_schema(df)
            if self.target is not None:
                rotulo = encode_target(df[self.target])
                if rotulo.notna().sum() == 0:
                    raise ValueError(f"A coluna '{self.target}' não tem valores reconhecidos como fraude/legítima.")
                df = df[rotulo.notna()]
            if self.schema == "generic":
                self.plan = plan_generic_columns(df, self.target)

        y = None
        if self.target is not None and self.target in df.columns:
            y = encode_target(df[self.target])
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
        estado = {
            "schema": self.schema,
            "target": self.target,
            "feature_names": list(self.feature_names),
            "medians": {k: float(v) for k, v in self.medians.items()},
            "data_min": [float(v) for v in self.scaler.data_min_],
            "data_max": [float(v) for v in self.scaler.data_max_],
        }
        if self.plan is not None:
            estado["plan"] = {
                "numeric": list(self.plan["numeric"]),
                "log": list(self.plan["log"]),
                "categorical": {k: list(v) for k, v in self.plan["categorical"].items()},
                "dropped": {k: str(v) for k, v in self.plan["dropped"].items()},
            }
        return estado

    @classmethod
    def from_dict(cls, state):
        pre = cls()
        pre.schema = state["schema"]
        pre.target = state["target"]
        pre.feature_names = list(state["feature_names"])
        pre.medians = dict(state["medians"])
        pre.plan = state.get("plan")
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


def limit_rows(features, y, max_rows, seed):
    """Limita o tamanho da base a uma amostra estratificada (mantém a taxa de fraude), para treinos rápidos."""
    if not max_rows or len(features) <= max_rows:
        return features, y
    estratificar = y is not None and y.nunique() == 2 and y.value_counts().min() >= 2
    idx, _ = train_test_split(features.index, train_size=max_rows, random_state=seed,
                              stratify=y if estratificar else None)
    return features.loc[idx], (y.loc[idx] if y is not None else None)


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
