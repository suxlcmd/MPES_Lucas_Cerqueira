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


@pytest.fixture
def creditcard_like():
    return make_creditcard_like()


@pytest.fixture
def paysim_like():
    return make_paysim_like()
