"""Training with the pre-registered regime: embargoed session-tail validation,
early stopping, exact determinism, warm-start support for the nightly loop.

The bundle stores everything live inference needs (state_dict, val-fit
isotonic calibrator, s_min, trained_through) so eval and live share one code
path (parity by construction — pinned by tests/test_dl_signals.py).
"""
from __future__ import annotations

import io
import json
import logging
import pickle
import random
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from . import require_torch
from .dataset import Samples, SymbolCache, build_samples, tensor_batch
from ..config import Config

log = logging.getLogger("engine.dl.train")


def set_determinism(seed: int) -> None:
    torch = require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)


@dataclass
class DLBundle:
    state_dict: dict
    calibrator: object | None          # isotonic: score -> P(target-before-stop)
    s_min: float
    trained_through: str
    seed: int
    meta: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        torch = require_torch()
        path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        torch.save(self.state_dict, buf)
        payload = {"state": buf.getvalue(),
                   "calibrator": pickle.dumps(self.calibrator),
                   "s_min": self.s_min, "trained_through": self.trained_through,
                   "seed": self.seed, "meta": self.meta}
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        with open(path.with_suffix(".json"), "w") as fh:
            json.dump({"trained_through": self.trained_through,
                       "s_min": self.s_min, "seed": self.seed,
                       **self.meta}, fh, indent=2)

    @staticmethod
    def load(path: Path) -> "DLBundle":
        torch = require_torch()
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        state = torch.load(io.BytesIO(payload["state"]), weights_only=True)
        return DLBundle(state_dict=state,
                        calibrator=pickle.loads(payload["calibrator"]),
                        s_min=float(payload["s_min"]),
                        trained_through=payload["trained_through"],
                        seed=int(payload["seed"]), meta=payload.get("meta", {}))


def _split_sessions(samples: Samples, val_frac: float, embargo_days: int):
    """Train-core / val row indices: val = last `val_frac` of sessions, with
    `embargo_days` TRADING sessions purged before it (mirrors walk_forward)."""
    sessions = np.unique(samples.session)
    n_val = max(1, int(round(len(sessions) * val_frac)))
    val_sessions = sessions[-n_val:]
    cut = len(sessions) - n_val - embargo_days
    core_sessions = sessions[:max(cut, 1)]
    core = np.where(np.isin(samples.session, core_sessions))[0]
    val = np.where(np.isin(samples.session, val_sessions))[0]
    return core, val


def _model(cfg: Config):
    from .model import CausalTCN
    return CausalTCN()


def fit(cfg: Config, caches: dict[str, SymbolCache], d_from: date, d_to: date,
        *, warm_start: "DLBundle | None" = None,
        horizon_min: int | None = None) -> tuple["DLBundle", Samples]:
    """Fit on all decision samples in [d_from, d_to]. Returns (bundle, samples).
    Calibrator + s_min are fit on the embargoed val slice ONLY (never test).
    """
    torch = require_torch()
    from .model import pinball_loss
    from .signals import fit_calibrator_and_smin

    dl = cfg.raw["dl"]
    seed = int(dl["seed"])
    set_determinism(seed)

    samples = build_samples(cfg, caches, d_from, d_to, horizon_min=horizon_min)
    if samples.y_cls.size < 5000:
        raise RuntimeError(f"too few samples ({samples.y_cls.size}) — widen range")
    core, val = _split_sessions(samples, float(dl["train"]["val_frac"]),
                                int(dl["train"]["embargo_days"]))
    max_n = int(dl["train"]["max_train_samples"])
    if core.size > max_n:                      # budget rule: keep most recent
        core = core[-max_n:]
    log.info("fit: %d train / %d val samples (%s -> %s)",
             core.size, val.size, d_from, d_to)

    model = _model(cfg)
    if warm_start is not None:
        model.load_state_dict(warm_start.state_dict)
    opt = torch.optim.AdamW(model.parameters(), lr=float(dl["train"]["lr"]),
                            weight_decay=float(dl["train"]["weight_decay"]))
    epochs = int(dl["train"]["max_epochs"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    batch = int(dl["train"]["batch"])
    patience = int(dl["train"]["patience"])

    # class weights from TRAIN core only
    cnt = np.bincount(samples.y_cls[core], minlength=3).astype(float)
    w = cnt.sum() / np.maximum(cnt, 1.0)
    w = w / w.mean()
    ce = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))

    y_cls_t = torch.from_numpy(samples.y_cls)
    y_ret_t = torch.from_numpy(samples.y_ret)
    rng = np.random.default_rng(seed)

    def run_val() -> float:
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for lo in range(0, val.size, 4096):
                rows = val[lo:lo + 4096]
                x1, x5, sy, st = tensor_batch(cfg, caches, samples, rows)
                logits, _ = model(x1, x5, sy, st)
                tot += torch.nn.functional.cross_entropy(
                    logits, y_cls_t[rows], reduction="sum").item()
                n += len(rows)
        return tot / max(n, 1)

    best_val, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        order = core.copy()
        rng.shuffle(order)
        ep_loss, nb = 0.0, 0
        for lo in range(0, order.size, batch):
            rows = order[lo:lo + batch]
            x1, x5, sy, st = tensor_batch(cfg, caches, samples, rows)
            logits, q = model(x1, x5, sy, st)
            loss = ce(logits, y_cls_t[rows]) + 0.3 * pinball_loss(q, y_ret_t[rows])
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss.item()); nb += 1
        sched.step()
        v = run_val()
        log.info("epoch %d: train %.4f  val CE %.4f", ep, ep_loss / max(nb, 1), v)
        if v < best_val - 1e-4:
            best_val, bad = v, 0
            buf = io.BytesIO(); torch.save(model.state_dict(), buf)
            best_state = buf.getvalue()
        else:
            bad += 1
            if bad >= patience:
                log.info("early stop at epoch %d (best val %.4f)", ep, best_val)
                break
    if best_state is not None:
        model.load_state_dict(torch.load(io.BytesIO(best_state), weights_only=True))

    # ---- calibrator + s_min on the VAL slice only ----
    model.eval()
    scores = predict_scores(cfg, model, caches, samples, val)
    calibrator, s_min = fit_calibrator_and_smin(cfg, caches, samples, val, scores)

    bundle = DLBundle(state_dict=model.state_dict(), calibrator=calibrator,
                      s_min=s_min, trained_through=d_to.isoformat(), seed=seed,
                      meta={"val_ce": round(best_val, 5),
                            "n_train": int(core.size), "n_val": int(val.size),
                            "params": model.param_count(),
                            "horizon_min": int(horizon_min or dl["horizon_min"])})
    return bundle, samples


def predict_scores(cfg: Config, model, caches, samples: Samples,
                   rows: np.ndarray) -> np.ndarray:
    """s = p_up - p_down for `rows` (deterministic, batched)."""
    torch = require_torch()
    from .model import CausalTCN
    model.eval()
    out = np.empty(rows.size, dtype=np.float64)
    with torch.no_grad():
        for lo in range(0, rows.size, 4096):
            rr = rows[lo:lo + 4096]
            x1, x5, sy, st = tensor_batch(cfg, caches, samples, rr)
            logits, _ = model(x1, x5, sy, st)
            out[lo:lo + rr.size] = CausalTCN.score(logits).numpy()
    return out


def model_from_bundle(cfg: Config, bundle: DLBundle):
    m = _model(cfg)
    m.load_state_dict(bundle.state_dict)
    m.eval()
    return m
