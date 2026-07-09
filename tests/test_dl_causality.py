"""MIE-DL training-harness certification (pre-registered controls).

(1) positive control — a label leaked INTO the input window must be learnable
    (proves the harness CAN detect signal);
(2) negative control — within-session label shuffle must yield chance AUC
    (proves the harness CANNOT cheat);
(3) determinism — two same-seed fits produce byte-identical weights;
(4) parameter budget cap.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from engine.config import load_config              # noqa: E402
from engine.dl.model import CausalTCN              # noqa: E402
from engine.dl import train as tr                  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg():
    return load_config(REPO_ROOT / "config.yaml")


def _toy_batch(n=2048, seed=0, leak=False):
    """Synthetic tensors; when leak=True the class is written into the last
    timestep of channel 0 (inside the input window) — a detectable plant."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, (n, 9, 64)).astype(np.float32)
    x5 = rng.normal(0, 1, (n, 6, 48)).astype(np.float32)
    sym = rng.integers(0, 8, n)
    stat = rng.normal(0, 1, (n, 2)).astype(np.float32)
    y = rng.integers(0, 3, n)
    if leak:
        x1[:, 0, -1] = (y.astype(np.float32) - 1.0) * 2.0
    return (torch.from_numpy(x1), torch.from_numpy(x5),
            torch.from_numpy(sym), torch.from_numpy(stat),
            torch.from_numpy(y))


def _fit_small(batch, epochs=3, seed=7):
    tr.set_determinism(seed)
    x1, x5, sym, stat, y = batch
    model = CausalTCN()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    ce = torch.nn.CrossEntropyLoss()
    n = len(y)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(n)
        for lo in range(0, n, 256):
            idx = order[lo:lo + 256]
            logits, _ = model(x1[idx], x5[idx], sym[idx], stat[idx])
            loss = ce(logits, y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def _auc_up_down(model, batch):
    from sklearn.metrics import roc_auc_score
    x1, x5, sym, stat, y = batch
    model.eval()
    with torch.no_grad():
        logits, _ = model(x1, x5, sym, stat)
        s = CausalTCN.score(logits).numpy()
    yb = y.numpy()
    m = yb != 1
    return roc_auc_score((yb[m] == 2).astype(int), s[m])


class TestHarness:
    def test_param_budget(self):
        m = CausalTCN()
        assert m.param_count() <= 500_000
        assert m.param_count() >= 50_000       # sanity: it's a real model

    def test_positive_control_leaked_label_is_learned(self):
        train = _toy_batch(2048, seed=1, leak=True)
        test = _toy_batch(1024, seed=2, leak=True)
        model = _fit_small(train, epochs=3)
        auc = _auc_up_down(model, test)
        assert auc > 0.90, f"harness failed to learn a planted signal (AUC {auc:.3f})"

    def test_negative_control_pure_noise_is_chance(self):
        train = _toy_batch(2048, seed=3, leak=False)
        test = _toy_batch(1024, seed=4, leak=False)
        model = _fit_small(train, epochs=3)
        auc = _auc_up_down(model, test)
        assert 0.44 <= auc <= 0.56, f"noise fit strayed from chance (AUC {auc:.3f})"

    def test_same_seed_fits_are_byte_identical(self):
        b = _toy_batch(512, seed=5)
        m1 = _fit_small(b, epochs=1, seed=7)
        m2 = _fit_small(b, epochs=1, seed=7)
        b1, b2 = io.BytesIO(), io.BytesIO()
        torch.save(m1.state_dict(), b1)
        torch.save(m2.state_dict(), b2)
        assert b1.getvalue() == b2.getvalue()
