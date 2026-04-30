from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Keep matplotlib fully local and headless.
if "MPLCONFIGDIR" not in os.environ:
    _mpl_dir = Path(__file__).resolve().parent / ".mplconfig"
    _mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_dir)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import svd
from scipy.stats import kurtosis
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0
T_TOTAL = 100.0
DT = 0.001
T_TRANSIENT = 20.0
NUM_DELAYS = 100
RANK = 15
TRAIN_FRACTION = 0.8
INNER_VALIDATION_FRACTION = 0.2
FORCING_ACTIVE_STD = 3.0
SWITCH_WINDOW_SAMPLES = 500
FREE_RUN_HORIZON = 5000
AR_ORDER_CANDIDATES = [10, 20, 30, 40, 50, 75, 100]
RIDGE_ALPHA = 1e-4
LINEAR_RIDGE_ALPHA = 1e-6
LINEAR_TRIM = 5
SEED_SWEEP = list(range(5))
TRAJECTORY_CACHE_FILENAME = "lorenz63_trajectory.npz"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "lines.linewidth": 1.2,
        "figure.figsize": (10, 6),
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
)


def lorenz63(t: float, state: np.ndarray, sigma: float = SIGMA, rho: float = RHO, beta: float = BETA):
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def lorenz_cache_signature() -> np.ndarray:
    return np.asarray([SIGMA, RHO, BETA, T_TOTAL, DT, T_TRANSIENT], dtype=float)



def simulate_lorenz(
    t_total: float = T_TOTAL,
    dt: float = DT,
    t_transient: float = T_TRANSIENT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t_eval = np.arange(0.0, t_total, dt)
    sol = solve_ivp(
        lorenz63,
        (0.0, t_total),
        [1.0, 1.0, 1.0],
        t_eval=t_eval,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"Lorenz simulation failed: {sol.message}")
    start = int(round(t_transient / dt))
    return sol.t[start:], sol.y[0, start:], sol.y[1, start:], sol.y[2, start:]


def load_or_simulate_lorenz(output_root: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dirs = ensure_output_dirs(output_root)
    cache_path = dirs["common"] / TRAJECTORY_CACHE_FILENAME
    signature = lorenz_cache_signature()

    if cache_path.exists():
        with np.load(cache_path) as cached:
            if {"t", "x", "y", "z", "config_signature"}.issubset(cached.files) and np.allclose(
                cached["config_signature"], signature
            ):
                return (
                    np.asarray(cached["t"]),
                    np.asarray(cached["x"]),
                    np.asarray(cached["y"]),
                    np.asarray(cached["z"]),
                )

    t, x, y, z = simulate_lorenz()
    np.savez(cache_path, t=t, x=x, y=y, z=z, config_signature=signature)
    return t, x, y, z



def build_hankel(x: np.ndarray, num_delays: int = NUM_DELAYS) -> np.ndarray:
    n_cols = len(x) - num_delays + 1
    if n_cols <= 0:
        raise ValueError("num_delays must be smaller than the signal length")
    hankel = np.zeros((num_delays, n_cols))
    for row in range(num_delays):
        hankel[row] = x[row : row + n_cols]
    return hankel



def project_onto_havok_basis(hankel: np.ndarray, u_r: np.ndarray, s_r: np.ndarray) -> np.ndarray:
    projected = hankel.T @ u_r
    return projected / s_r[np.newaxis, :]



def havok_decomposition(
    x: np.ndarray,
    num_delays: int = NUM_DELAYS,
    rank: int = RANK,
    train_fraction: float = TRAIN_FRACTION,
):
    hankel_full = build_hankel(x, num_delays)
    train_cols = int(train_fraction * hankel_full.shape[1])
    hankel_train = hankel_full[:, :train_cols]
    u, s, vt_train = svd(hankel_train, full_matrices=False)
    energy = np.cumsum(s**2) / np.sum(s**2)
    u_r = u[:, :rank]
    s_r = s[:rank]
    v_full = project_onto_havok_basis(hankel_full, u_r, s_r)
    return {
        "hankel_full": hankel_full,
        "hankel_train": hankel_train,
        "train_cols": train_cols,
        "u_r": u_r,
        "s_r": s_r,
        "v_full": v_full,
        "s_all": s,
        "energy": energy,
        "vt_train": vt_train,
    }



def compute_time_derivative(states: np.ndarray, dt: float = DT) -> np.ndarray:
    deriv = np.zeros_like(states)
    deriv[1:-1] = (states[2:] - states[:-2]) / (2.0 * dt)
    deriv[0] = (states[1] - states[0]) / dt
    deriv[-1] = (states[-1] - states[-2]) / dt
    return deriv



def fit_havok_linear_model(v_trim_train: np.ndarray, dt: float = DT):
    dvdt = compute_time_derivative(v_trim_train, dt)
    v_active = v_trim_train[:, :-1]
    forcing = v_trim_train[:, -1]
    regressors = np.column_stack([v_active, forcing])
    dv_active = dvdt[:, :-1]

    reg = Ridge(alpha=LINEAR_RIDGE_ALPHA, fit_intercept=False)
    reg.fit(regressors, dv_active)
    coeff = reg.coef_
    a = coeff[:, :-1]
    b = coeff[:, -1:]
    dv_pred = np.einsum("ij,kj->ik", regressors, coeff, optimize=True)
    fit_rmse = float(np.sqrt(np.mean((dv_active - dv_pred) ** 2)))
    return a, b, forcing, v_active, fit_rmse



def build_supervised_block(series: np.ndarray, start_idx: int, end_idx: int, order: int):
    x_rows: List[np.ndarray] = []
    y_rows: List[float] = []
    for target_idx in range(start_idx, end_idx):
        x_rows.append(series[target_idx - order : target_idx][::-1])
        y_rows.append(series[target_idx])
    return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=float)



def selection_split(n: int) -> Tuple[int, int]:
    train_end = int(n * TRAIN_FRACTION)
    inner_train_end = int(train_end * (1.0 - INNER_VALIDATION_FRACTION))
    return inner_train_end, train_end



def fit_ar_regression(series: np.ndarray, order: int, train_end: int) -> Ridge:
    x_train, y_train = build_supervised_block(series, order, train_end, order)
    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    model.fit(x_train, y_train)
    return model



def one_step_predict(model: Ridge, series: np.ndarray, start_idx: int, end_idx: int, order: int) -> np.ndarray:
    x_block, _ = build_supervised_block(series, start_idx, end_idx, order)
    return model.predict(x_block)



def recursive_mean_forecast(model: Ridge, history: np.ndarray, horizon: int, order: int) -> np.ndarray:
    buffer = list(np.asarray(history[-order:], dtype=float))
    preds = np.zeros(horizon, dtype=float)
    for idx in range(horizon):
        features = np.asarray(buffer[-order:][::-1], dtype=float).reshape(1, -1)
        pred = float(model.predict(features)[0])
        preds[idx] = pred
        buffer.append(pred)
    return preds



def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))



def matrix_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))



def choose_ar_order(series: np.ndarray, candidates: List[int] | None = None) -> Tuple[int, List[Dict[str, float]]]:
    if candidates is None:
        candidates = AR_ORDER_CANDIDATES

    inner_train_end, train_end = selection_split(len(series))
    scores: List[Dict[str, float]] = []
    best_order = candidates[0]
    best_rmse = np.inf

    for order in candidates:
        if inner_train_end <= order or train_end <= order:
            continue
        model = fit_ar_regression(series[:train_end], order, inner_train_end)
        val_pred = one_step_predict(model, series[:train_end], inner_train_end, train_end, order)
        val_true = series[inner_train_end:train_end]
        val_rmse = rmse(val_true, val_pred)
        scores.append({"order": int(order), "validation_rmse": float(val_rmse)})
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_order = order

    if not scores:
        raise ValueError("No valid AR order candidates for the available series length")
    return best_order, scores



def forcing_switch_metrics(
    x: np.ndarray,
    forcing: np.ndarray,
    num_delays: int,
    trim: int,
    dt: float,
    threshold_std: float = FORCING_ACTIVE_STD,
    window_samples: int = SWITCH_WINDOW_SAMPLES,
) -> Dict[str, float]:
    start = (num_delays - 1) + trim
    x_aligned = x[start : start + len(forcing)]
    if len(x_aligned) != len(forcing):
        common = min(len(x_aligned), len(forcing))
        x_aligned = x_aligned[:common]
        forcing = forcing[:common]

    sign = np.sign(x_aligned)
    if len(sign) and sign[0] == 0:
        sign[0] = 1.0
    for idx in range(1, len(sign)):
        if sign[idx] == 0:
            sign[idx] = sign[idx - 1]
    switch = np.zeros(len(sign), dtype=bool)
    switch[1:] = sign[1:] != sign[:-1]

    threshold = threshold_std * np.std(forcing)
    active = np.abs(forcing) > threshold

    switch_idx = np.where(switch)[0]
    active_idx = np.where(active)[0]
    if len(switch_idx) == 0 or len(active_idx) == 0:
        return {
            "active_threshold_abs": float(threshold),
            "switch_count": int(len(switch_idx)),
            "active_count": int(len(active_idx)),
            "switch_recall_in_window": 0.0,
            "active_precision_in_window": 0.0,
            "window_time": float(window_samples * dt),
        }

    switch_hits = 0
    for switch_i in switch_idx:
        left = max(0, switch_i - window_samples)
        right = min(len(active), switch_i + window_samples + 1)
        if np.any(active[left:right]):
            switch_hits += 1

    active_hits = 0
    for active_i in active_idx:
        left = max(0, active_i - window_samples)
        right = min(len(switch), active_i + window_samples + 1)
        if np.any(switch[left:right]):
            active_hits += 1

    return {
        "active_threshold_abs": float(threshold),
        "switch_count": int(len(switch_idx)),
        "active_count": int(len(active_idx)),
        "switch_recall_in_window": float(switch_hits / len(switch_idx)),
        "active_precision_in_window": float(active_hits / len(active_idx)),
        "window_time": float(window_samples * dt),
    }



def evaluate_naive_baselines(forcing: np.ndarray, train_end: int, common_horizon: int) -> Dict[str, float]:
    y_true = forcing[train_end:]
    persistence_pred = forcing[train_end - 1 : -1]
    mean_value = float(np.mean(forcing[:train_end]))
    mean_pred = np.full_like(y_true, mean_value)

    persistence_free = np.full(common_horizon, forcing[train_end - 1])
    mean_free = np.full(common_horizon, mean_value)

    return {
        "persistence_one_step_rmse": float(rmse(y_true, persistence_pred)),
        "mean_one_step_rmse": float(rmse(y_true, mean_pred)),
        "persistence_free_run_rmse_common": float(rmse(y_true[:common_horizon], persistence_free)),
        "mean_free_run_rmse_common": float(rmse(y_true[:common_horizon], mean_free)),
        "common_horizon": int(common_horizon),
    }



def reconstruct_scalar_from_coordinates(v_coords: np.ndarray, u_r: np.ndarray, s_r: np.ndarray, row: int = -1) -> np.ndarray:
    weights = u_r[row, :] * s_r
    return v_coords @ weights



def simulate_true_forcing_linear_system(
    a: np.ndarray,
    b: np.ndarray,
    forcing_test: np.ndarray,
    t_test: np.ndarray,
    initial_state: np.ndarray,
) -> np.ndarray:
    if len(t_test) == 0:
        return np.zeros((0, len(initial_state)), dtype=float)

    def rhs(t_query: float, state: np.ndarray) -> np.ndarray:
        forcing_now = float(np.interp(t_query, t_test, forcing_test))
        return a @ state + b[:, 0] * forcing_now

    sol = solve_ivp(
        rhs,
        (float(t_test[0]), float(t_test[-1])),
        initial_state,
        t_eval=t_test,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"linear-system simulation failed: {sol.message}")
    return sol.y.T



def paper_style_havok_validation(
    a: np.ndarray,
    b: np.ndarray,
    v_active_full: np.ndarray,
    forcing_full: np.ndarray,
    t_trimmed: np.ndarray,
    train_end: int,
    u_r: np.ndarray,
    s_r: np.ndarray,
    x_aligned: np.ndarray,
) -> Dict[str, np.ndarray | float | int]:
    v_active_test = v_active_full[train_end:]
    forcing_test = forcing_full[train_end:]
    t_test = t_trimmed[train_end:]
    initial_state = v_active_full[train_end]

    v_active_pred = simulate_true_forcing_linear_system(a, b, forcing_test, t_test, initial_state)
    v_full_pred = np.column_stack([v_active_pred, forcing_test])
    x_pred = reconstruct_scalar_from_coordinates(v_full_pred, u_r, s_r, row=-1)
    x_true = x_aligned[train_end:]
    t_rel = t_test - t_test[0]
    v1_error = np.abs(v_active_test[:, 0] - v_active_pred[:, 0])
    v1_std = float(np.std(v_active_test[:, 0]))

    def first_crossing_time(values: np.ndarray, threshold: float) -> float | None:
        hits = np.where(values > threshold)[0]
        if len(hits) == 0:
            return None
        return float(t_rel[hits[0]])

    return {
        "v_active_pred": v_active_pred,
        "v_active_true": v_active_test,
        "forcing_test": forcing_test,
        "t_test": t_test,
        "x_pred": x_pred,
        "x_true": x_true,
        "v1_rmse": float(rmse(v_active_test[:, 0], v_active_pred[:, 0])),
        "active_rmse": float(matrix_rmse(v_active_test, v_active_pred)),
        "x_rmse": float(rmse(x_true, x_pred)),
        "v1_std_test": v1_std,
        "tracking_time_half_std": first_crossing_time(v1_error, 0.5 * v1_std),
        "tracking_time_abs_0p005": first_crossing_time(v1_error, 0.005),
        "test_points": int(len(t_test)),
    }



def ensure_output_dirs(output_root: Path) -> Dict[str, Path]:
    common_dir = output_root / "common"
    deterministic_dir = output_root / "deterministic"
    probabilistic_dir = output_root / "probabilistic"
    markov_dir = output_root / "markov"
    for folder in [output_root, common_dir, deterministic_dir, probabilistic_dir, markov_dir]:
        folder.mkdir(parents=True, exist_ok=True)
    return {
        "root": output_root,
        "common": common_dir,
        "deterministic": deterministic_dir,
        "probabilistic": probabilistic_dir,
        "markov": markov_dir,
    }



def _sanitize_json_value(value):
    if isinstance(value, dict):
        return {str(key): _sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _sanitize_json_value(value.tolist())
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(_sanitize_json_value(payload), indent=2))



def update_summary(output_root: Path, model_name: str, metrics: Dict) -> Dict:
    summary_path = output_root / "summary_metrics.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        summary = {}
    summary[model_name] = metrics
    write_json(summary_path, summary)
    return summary



def prepare_havok_dataset(output_root: Path) -> Dict:
    dirs = ensure_output_dirs(output_root)
    t, x, y, z = load_or_simulate_lorenz(output_root)
    decomposition = havok_decomposition(x, NUM_DELAYS, RANK, TRAIN_FRACTION)

    v_full = decomposition["v_full"]
    u_r = decomposition["u_r"]
    s_r = decomposition["s_r"]
    s_all = decomposition["s_all"]
    energy = decomposition["energy"]

    v_trimmed = v_full[LINEAR_TRIM:-LINEAR_TRIM]
    forcing = v_trimmed[:, -1]
    v_active = v_trimmed[:, :-1]
    train_end = int(TRAIN_FRACTION * len(v_trimmed))

    a, b, forcing_train, v_active_train, havok_rmse = fit_havok_linear_model(v_trimmed[:train_end], DT)

    start = (NUM_DELAYS - 1) + LINEAR_TRIM
    t_forcing = t[start : start + len(forcing)]
    x_aligned = x[start : start + len(forcing)]
    switch_stats = forcing_switch_metrics(x, forcing, NUM_DELAYS, LINEAR_TRIM, DT)

    skew_symmetry_error = float(np.linalg.norm(a + a.T) / np.linalg.norm(a))
    eigvals = np.linalg.eigvals(a)
    forcing_excess_kurtosis = float(kurtosis(forcing, fisher=True, bias=False))

    paper_validation = paper_style_havok_validation(
        a=a,
        b=b,
        v_active_full=v_active,
        forcing_full=forcing,
        t_trimmed=t_forcing,
        train_end=train_end,
        u_r=u_r,
        s_r=s_r,
        x_aligned=x_aligned,
    )

    common_horizon = min(FREE_RUN_HORIZON, len(forcing) - train_end)
    naive_baselines = evaluate_naive_baselines(forcing, train_end, common_horizon)

    common_metrics = {
        "framing": {
            "project_type": "extension",
            "summary": "The original HAVOK paper treats v_r as a known input; this project extends HAVOK by forecasting v_r so the model can run autonomously without future measurements.",
            "havok_basis_fit": "train_only",
        },
        "config": {
            "sigma": SIGMA,
            "rho": RHO,
            "beta": BETA,
            "t_total": T_TOTAL,
            "dt": DT,
            "t_transient": T_TRANSIENT,
            "num_delays": NUM_DELAYS,
            "rank": RANK,
            "train_fraction": TRAIN_FRACTION,
            "inner_validation_fraction": INNER_VALIDATION_FRACTION,
            "free_run_horizon": FREE_RUN_HORIZON,
            "linear_trim": LINEAR_TRIM,
            "trajectory_cache_file": str(dirs["common"] / TRAJECTORY_CACHE_FILENAME),
        },
        "havok": {
            "A_shape": list(a.shape),
            "B_shape": list(b.shape),
            "forcing_length": int(len(forcing)),
            "linear_rmse_train": float(havok_rmse),
            "energy_at_rank": float(energy[RANK - 1]),
            "forcing_excess_kurtosis": forcing_excess_kurtosis,
            "skew_symmetry_relative_error": skew_symmetry_error,
            "eigenvalue_real_max_abs": float(np.max(np.abs(eigvals.real))),
        },
        "paper_style_validation": {
            "v1_rmse": float(paper_validation["v1_rmse"]),
            "active_rmse": float(paper_validation["active_rmse"]),
            "x_rmse": float(paper_validation["x_rmse"]),
            "v1_std_test": float(paper_validation["v1_std_test"]),
            "tracking_time_half_std": paper_validation["tracking_time_half_std"],
            "tracking_time_abs_0p005": paper_validation["tracking_time_abs_0p005"],
            "test_points": int(paper_validation["test_points"]),
        },
        "forcing_switch": switch_stats,
        "naive_baselines": naive_baselines,
    }
    write_json(dirs["common"] / "common_metrics.json", common_metrics)
    update_summary(output_root, "common", common_metrics)

    plot_common_figures(
        common_dir=dirs["common"],
        t=t,
        x=x,
        y=y,
        z=z,
        s_all=s_all,
        energy=energy,
        forcing=forcing,
        t_forcing=t_forcing,
        a=a,
        eigvals=eigvals,
        switch_stats=switch_stats,
        paper_validation=paper_validation,
    )

    return {
        "t": t,
        "x": x,
        "y": y,
        "z": z,
        "u_r": u_r,
        "s_r": s_r,
        "v_full": v_full,
        "v_trimmed": v_trimmed,
        "v_active": v_active,
        "forcing": forcing,
        "a": a,
        "b": b,
        "train_end": train_end,
        "t_forcing": t_forcing,
        "x_aligned": x_aligned,
        "energy": energy,
        "s_all": s_all,
        "havok_rmse": havok_rmse,
        "switch_stats": switch_stats,
        "paper_validation": paper_validation,
        "naive_baselines": naive_baselines,
        "common_horizon": common_horizon,
    }



def plot_common_figures(
    common_dir: Path,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    s_all: np.ndarray,
    energy: np.ndarray,
    forcing: np.ndarray,
    t_forcing: np.ndarray,
    a: np.ndarray,
    eigvals: np.ndarray,
    switch_stats: Dict[str, float],
    paper_validation: Dict[str, np.ndarray | float | int],
) -> None:
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot(x[::10], y[::10], z[::10], lw=0.3, alpha=0.8, color="#1f77b4")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.set_title("Lorenz-63 attractor")
    ax1.view_init(elev=25, azim=130)

    ax2 = fig.add_subplot(122)
    n_short = min(10000, len(t))
    ax2.plot(t[:n_short], x[:n_short], lw=0.6, color="#1f77b4")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("x(t)")
    ax2.set_title("Observed scalar signal used for HAVOK")
    plt.tight_layout()
    plt.savefig(common_dir / "fig01_lorenz63_overview.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    n_sv = min(50, len(s_all))
    axes[0].semilogy(np.arange(1, n_sv + 1), s_all[:n_sv], "o-", markersize=4, color="#1f77b4")
    axes[0].axvline(RANK, color="red", linestyle="--", alpha=0.7, label=f"r = {RANK}")
    axes[0].set_xlabel("Mode index")
    axes[0].set_ylabel("Singular value")
    axes[0].set_title("Hankel singular values (train basis)")
    axes[0].legend()

    axes[1].plot(np.arange(1, n_sv + 1), 100.0 * energy[:n_sv], "o-", markersize=4, color="#ff7f0e")
    axes[1].axvline(RANK, color="red", linestyle="--", alpha=0.7, label=f"r = {RANK}")
    axes[1].set_xlabel("Mode index")
    axes[1].set_ylabel("Captured energy (%)")
    axes[1].set_title("Cumulative train energy")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(common_dir / "fig02_hankel_spectrum.png")
    plt.close(fig)

    threshold = switch_stats["active_threshold_abs"]
    active = np.abs(forcing) > threshold

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    n_show = min(10000, len(forcing))
    axes[0].plot(t_forcing[:n_show], forcing[:n_show], color="#d62728", lw=0.7)
    axes[0].axhline(threshold, color="black", linestyle="--", alpha=0.6, label="+3 std threshold")
    axes[0].axhline(-threshold, color="black", linestyle="--", alpha=0.6, label="-3 std threshold")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel(f"v_{RANK}(t)")
    axes[0].set_title("HAVOK forcing signal v_r")
    axes[0].legend(loc="upper right")

    axes[1].plot(t_forcing[:n_show], forcing[:n_show], color="gray", lw=0.5, alpha=0.6)
    axes[1].fill_between(
        t_forcing[:n_show],
        0.0,
        forcing[:n_show],
        where=active[:n_show],
        color="#d62728",
        alpha=0.45,
        label="Large forcing bursts",
    )
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel(f"v_{RANK}(t)")
    axes[1].set_title("Bursts in the forcing signal")
    axes[1].legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(common_dir / "fig03_vr_forcing_signal.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].hist(forcing, bins=120, density=True, color="#1f77b4", alpha=0.8, edgecolor="white", lw=0.3)
    axes[0].set_xlabel(f"v_{RANK}")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Heavy-tailed forcing distribution")

    vmax = np.max(np.abs(a))
    im = axes[1].imshow(a, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Row")
    axes[1].set_title("Linear HAVOK A matrix")
    plt.colorbar(im, ax=axes[1])

    axes[2].scatter(eigvals.real, eigvals.imag, color="#d62728", s=22)
    axes[2].axhline(0.0, color="black", lw=0.6)
    axes[2].axvline(0.0, color="black", lw=0.6)
    axes[2].set_xlabel("Real part")
    axes[2].set_ylabel("Imaginary part")
    axes[2].set_title("Eigenvalues of A")
    plt.tight_layout()
    plt.savefig(common_dir / "fig04_forcing_distribution_and_A.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    t_test = np.asarray(paper_validation["t_test"])
    v_active_true = np.asarray(paper_validation["v_active_true"])
    v_active_pred = np.asarray(paper_validation["v_active_pred"])
    x_true = np.asarray(paper_validation["x_true"])
    x_pred = np.asarray(paper_validation["x_pred"])

    n_plot = min(2000, len(t_test))
    axes[0].plot(t_test[:n_plot], v_active_true[:n_plot, 0], color="#1f77b4", lw=0.8, label="True v1")
    axes[0].plot(t_test[:n_plot], v_active_pred[:n_plot, 0], color="#d62728", lw=0.8, label="Driven HAVOK v1")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("v1")
    axes[0].set_title("Paper-style validation with true forcing input")
    axes[0].legend(loc="upper right")

    axes[1].plot(t_test[:n_plot], x_true[:n_plot], color="#1f77b4", lw=0.8, label="True x(t)")
    axes[1].plot(t_test[:n_plot], x_pred[:n_plot], color="#2ca02c", lw=0.8, label="Reconstructed x(t)")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("x(t)")
    axes[1].set_title("Driven linear subsystem with true forcing from test data")
    axes[1].legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(common_dir / "fig05_paper_driven_validation.png")
    plt.close(fig)



def run_deterministic_model(output_root: Path, seed: int = 0) -> Dict:
    dirs = ensure_output_dirs(output_root)
    data = prepare_havok_dataset(output_root)
    forcing = data["forcing"]
    train_end = data["train_end"]

    best_order, order_scores = choose_ar_order(forcing)
    model = fit_ar_regression(forcing[:train_end], best_order, train_end)

    test_pred = one_step_predict(model, forcing, train_end, len(forcing), best_order)
    test_true = forcing[train_end:]
    test_mae = float(mean_absolute_error(test_true, test_pred))
    one_step_rmse = rmse(test_true, test_pred)

    common_horizon = min(FREE_RUN_HORIZON, len(test_true))
    context = forcing[train_end - best_order : train_end]
    free_run_pred = recursive_mean_forecast(model, context, len(test_true), best_order)
    free_run_rmse_full = rmse(test_true, free_run_pred)
    free_run_rmse_common = rmse(test_true[:common_horizon], free_run_pred[:common_horizon])

    metrics = {
        "model_name": "deterministic_ar",
        "selected_order": int(best_order),
        "order_candidates": order_scores,
        "train_points": int(train_end),
        "test_points": int(len(test_true)),
        "one_step_rmse": float(one_step_rmse),
        "one_step_mae": float(test_mae),
        "free_run_rmse_full": float(free_run_rmse_full),
        "free_run_rmse_common": float(free_run_rmse_common),
        "common_horizon": int(common_horizon),
        "beats_persistence_one_step_by_factor": float(data["naive_baselines"]["persistence_one_step_rmse"] / one_step_rmse),
        "free_run_minus_mean_baseline": float(
            free_run_rmse_common - data["naive_baselines"]["mean_free_run_rmse_common"]
        ),
        "seed_stability": {
            "seeds": SEED_SWEEP,
            "one_step_rmse_mean": float(one_step_rmse),
            "one_step_rmse_std": 0.0,
            "free_run_rmse_common_mean": float(free_run_rmse_common),
            "free_run_rmse_common_std": 0.0,
        },
        "notes": {
            "baseline": "Ridge-regularized autoregressive model on the HAVOK forcing signal.",
            "regularization_alpha": RIDGE_ALPHA,
            "framing": "This is an extension beyond the original HAVOK paper, which does not forecast v_r.",
            "free_run_interpretation": "In recursive mode the forecast relaxes toward the unconditional mean because the forcing spends most of its time in near-zero quiescent periods. Once burst timing slips, long-horizon RMSE approaches the mean baseline.",
            "seed_usage": "The deterministic model has no stochastic component, so the seed is kept only for CLI symmetry.",
        },
    }

    plot_deterministic_figures(
        output_dir=dirs["deterministic"],
        order_scores=order_scores,
        forcing_test=test_true,
        one_step_pred=test_pred,
        free_run_pred=free_run_pred,
        dt=DT,
        order=best_order,
    )

    np.savez(
        dirs["deterministic"] / "predictions.npz",
        y_true_test=test_true,
        one_step_prediction=test_pred,
        free_run_prediction=free_run_pred,
        dt=np.asarray([DT]),
        common_horizon=np.asarray([common_horizon]),
    )
    write_json(dirs["deterministic"] / "metrics.json", metrics)
    update_summary(output_root, "deterministic", metrics)
    return metrics



def plot_deterministic_figures(
    output_dir: Path,
    order_scores: List[Dict[str, float]],
    forcing_test: np.ndarray,
    one_step_pred: np.ndarray,
    free_run_pred: np.ndarray,
    dt: float,
    order: int,
) -> None:
    orders = [entry["order"] for entry in order_scores]
    rmses = [entry["validation_rmse"] for entry in order_scores]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(orders, rmses, "o-", color="#1f77b4")
    ax.axvline(order, color="red", linestyle="--", alpha=0.7, label=f"chosen order = {order}")
    ax.set_xlabel("AR order")
    ax.set_ylabel("Validation RMSE")
    ax.set_title("Deterministic baseline: order selection")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fig06_deterministic_order_sweep.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    n_plot = min(2000, len(forcing_test))
    t_plot = np.arange(n_plot) * dt
    axes[0, 0].plot(t_plot, forcing_test[:n_plot], color="#1f77b4", label="True", lw=0.7)
    axes[0, 0].plot(t_plot, one_step_pred[:n_plot], color="#d62728", label="Predicted", lw=0.7)
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel(f"v_{RANK}")
    axes[0, 0].set_title(f"One-step prediction with AR({order})")
    axes[0, 0].legend()

    n_scatter = min(5000, len(forcing_test))
    lim_low = min(forcing_test[:n_scatter].min(), one_step_pred[:n_scatter].min())
    lim_high = max(forcing_test[:n_scatter].max(), one_step_pred[:n_scatter].max())
    axes[0, 1].scatter(forcing_test[:n_scatter], one_step_pred[:n_scatter], s=2, alpha=0.25, color="#1f77b4")
    axes[0, 1].plot([lim_low, lim_high], [lim_low, lim_high], "k--", lw=1.0)
    axes[0, 1].set_xlabel("True value")
    axes[0, 1].set_ylabel("Predicted value")
    axes[0, 1].set_title("One-step scatter")

    axes[1, 0].plot(t_plot, forcing_test[:n_plot], color="#1f77b4", label="True", lw=0.7)
    axes[1, 0].plot(t_plot, free_run_pred[:n_plot], color="#d62728", label="Recursive forecast", lw=0.7)
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel(f"v_{RANK}")
    axes[1, 0].set_title("Free-running deterministic forecast")
    axes[1, 0].legend()

    errors = (forcing_test[:n_plot] - free_run_pred[:n_plot]) ** 2
    window = min(200, len(errors))
    if window > 3:
        rolling = np.sqrt(np.convolve(errors, np.ones(window) / window, mode="valid"))
        axes[1, 1].plot(np.arange(len(rolling)) * dt, rolling, color="#d62728", lw=0.8)
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].set_ylabel("Rolling RMSE")
    axes[1, 1].set_title("Error growth in the recursive forecast")
    plt.tight_layout()
    plt.savefig(output_dir / "fig06b_deterministic_predictions.png")
    plt.close(fig)



def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic HAVOK forcing prediction on Lorenz-63")
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(Path(__file__).resolve().parent / "havok_results"),
        help="Shared output directory for figures and metrics",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    metrics = run_deterministic_model(output_root=output_root, seed=args.seed)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
