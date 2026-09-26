import numpy as np
import pandas as pd
import pytest

from orchestration.data_preprocessor import DataPreprocessor, balance_sample, clean_raw_dataframe


def test_creditcard_engenharia_e_serializacao(creditcard_like):
    pre = DataPreprocessor()
    _, features, y = pre.prepare(creditcard_like, training=True)
    assert pre.schema == "creditcard" and pre.target == "Class"
    assert "Time" not in features and "Amount" not in features and "amount_log" in features
    assert y.sum() == 40

    pre.fit(features)
    X = pre.transform(features)
    assert X.shape == (len(creditcard_like), 29)
    assert np.isclose(X.min(), -1) and np.isclose(X.max(), 1)

    restaurado = DataPreprocessor.from_dict(pre.to_dict())
    assert np.allclose(restaurado.transform(features), X)


def test_paysim_one_hot_fixo(paysim_like):
    pre = DataPreprocessor()
    _, features, _ = pre.prepare(paysim_like, training=True)
    assert pre.schema == "paysim" and pre.target == "isFraud"
    assert {"type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"} <= set(features.columns)
    assert not {"nameOrig", "nameDest", "isFlaggedFraud"} & set(features.columns)
    pre.fit(features)

    # Um lote só com TRANSFER precisa produzir exatamente as mesmas colunas
    lote = paysim_like[paysim_like["type"] == "TRANSFER"].drop(columns=["isFraud"])
    _, feat_lote, y_lote = pre.prepare(lote, training=False)
    assert y_lote is None
    assert pre.transform(feat_lote).shape[1] == len(pre.feature_names)


def test_colunas_faltantes_geram_erro_claro(creditcard_like):
    pre = DataPreprocessor()
    _, features, _ = pre.prepare(creditcard_like, training=True)
    pre.fit(features)
    _, incompleto, _ = pre.prepare(creditcard_like.drop(columns=["V3"]), training=False)
    with pytest.raises(ValueError, match="V3"):
        pre.transform(incompleto)


def test_csv_lido_em_uma_coluna():
    df = pd.DataFrame({'"a,b,Class"': ['"1,2,0"', '"3,4,1"']})
    limpo = clean_raw_dataframe(df)
    assert list(limpo.columns) == ["a", "b", "Class"]


def test_amostragem_mantem_todas_as_fraudes(creditcard_like):
    pre = DataPreprocessor()
    _, features, y = pre.prepare(creditcard_like, training=True)
    f2, y2 = balance_sample(features, y, normals_per_fraud=5, seed=0)
    assert y2.sum() == y.sum() and len(y2) == y.sum() * 6
