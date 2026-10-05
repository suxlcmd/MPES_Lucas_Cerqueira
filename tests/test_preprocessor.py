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


def test_rotulos_em_texto_e_nomes_variados():
    from orchestration.data_preprocessor import detect_target, encode_target
    df = pd.DataFrame({"Is Fraudulent": [1, 0], "valor": [1.0, 2.0]})
    assert detect_target(df) == "Is Fraudulent"
    assert detect_target(pd.DataFrame({"FRAUD_BOOL": [0, 1]})) == "FRAUD_BOOL"
    rotulos = encode_target(pd.Series(["Y", "N", "sim", "Não", "True", "?", None]))
    assert rotulos.tolist()[:5] == [1, 0, 1, 0, 1] and rotulos[5:].isna().all()


def test_plano_generico(generic_like):
    pre = DataPreprocessor()
    _, features, y = pre.prepare(generic_like, training=True)
    assert pre.schema == "generic" and pre.target == "fraud_reported" and y.isin([0, 1]).all()
    assert pre.plan["dropped"]["policy_number"] == "identificador"
    assert pre.plan["dropped"]["incident_date"] == "data"
    assert "total_claim_amount" in pre.plan["log"]
    assert "incident_severity=Minor Damage" in features.columns
    pre.fit(features)
    restaurado = DataPreprocessor.from_dict(pre.to_dict())
    _, f2, _ = restaurado.prepare(generic_like.drop(columns=["fraud_reported"]), training=False)
    assert np.allclose(restaurado.transform(f2), pre.transform(features))
    with pytest.raises(ValueError, match="incident_severity"):
        restaurado.prepare(generic_like.drop(columns=["incident_severity"]), training=False)


def test_candidatas_a_coluna_de_fraude(generic_like):
    from orchestration.data_preprocessor import label_candidates
    assert "fraud_reported" in label_candidates(generic_like)
    assert "age" not in label_candidates(generic_like)
