import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def make_creditcard_like():
    """Base sintética no formato Credit Card (V1..V28, Amount, Class) com fraudes separáveis."""
    rng = np.random.default_rng(0)
    n, n_fraudes = 1500, 40
    dados = rng.normal(size=(n, 28))
    dados[:n_fraudes] += 3.0
    df = pd.DataFrame(dados, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.arange(n, dtype=float))
    df["Amount"] = np.abs(rng.normal(80, 40, size=n))
    df["Class"] = 0
    df.loc[: n_fraudes - 1, "Class"] = 1
    return df.sample(frac=1, random_state=0).reset_index(drop=True)


def make_paysim_like():
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame({
        "step": rng.integers(1, 700, n),
        "type": rng.choice(["CASH_IN", "CASH_OUT", "PAYMENT", "TRANSFER", "DEBIT"], n),
        "amount": rng.exponential(1000, n),
        "nameOrig": [f"C{i}" for i in range(n)],
        "oldbalanceOrg": rng.exponential(2000, n),
        "newbalanceOrig": rng.exponential(2000, n),
        "nameDest": [f"M{i}" for i in range(n)],
        "oldbalanceDest": rng.exponential(5000, n),
        "newbalanceDest": rng.exponential(5000, n),
        "isFraud": (rng.random(n) < 0.05).astype(int),
        "isFlaggedFraud": 0,
    })
    return df


def make_generic_like():
    """Base genérica no estilo de sinistros de seguro: rótulo Y/N, categorias, identificador e data."""
    rng = np.random.default_rng(2)
    n = 900
    fraude = rng.random(n) < 0.2
    severidade = np.where(fraude & (rng.random(n) < 0.8), "Major Damage",
                          rng.choice(["Minor Damage", "Total Loss", "Trivial Damage"], n))
    return pd.DataFrame({
        "policy_number": rng.permutation(np.arange(100000, 100000 + n)),
        "incident_date": pd.date_range("2015-01-01", periods=n, freq="h").strftime("%Y-%m-%d"),
        "age": rng.integers(19, 70, n),
        "total_claim_amount": np.round(rng.lognormal(10, 0.6, n) * np.where(fraude, 1.6, 1.0)),
        "incident_severity": severidade,
        "insured_sex": rng.choice(["MALE", "FEMALE"], n),
        "fraud_reported": np.where(fraude, "Y", "N"),
    })


@pytest.fixture
def generic_like():
    return make_generic_like()


@pytest.fixture
def creditcard_like():
    return make_creditcard_like()


@pytest.fixture
def paysim_like():
    return make_paysim_like()
