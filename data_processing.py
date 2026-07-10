"""Self-potential and hydrogeophysical processing routines (flexible inputs).

This module loads configuration from a TOML file and performs a series of
processing steps originally translated from MATLAB code. It supports:

* Gaussian despiking and exponential smoothing.
* Boxcar (moving-average) filtering.
* Least-squares polynomial fitting.
* FFT-based amplitude spectra.
* Simpson-rule integration of electric field to potential.
* Segment-wise drift correction and partitioning of SP signals.
* Temperature and conductivity drift correction (optional).

The main entry point is `process_code()`, which reads raw CSV inputs, applies
all processing steps, and optionally writes several processed CSV products.

Configuration
-------------
The module expects a TOML file with the following structure:

[paths]
input_dir = "test_data"
processed_dir = "test_data/processed"
figures_dir = "test_data/figures"

[files]
sp_data = "Self_Potential_Data_Rio_Grande.csv"
drift_data = ""                    # optional: leave blank or missing to skip
temp_cond_data = ""                # optional: leave blank or missing to skip
hfem_resistivity = ""              # reserved for future use

[processing]
dipole_length_m = 0.5588
gaussian_sigma = 30
boxcar_m = 5
boxcar_iterations = 1
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tomllib
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
ConfigDict = dict[str, Any]
ProcessResults = dict[str, FloatArray]


def load_config(config_path: str | Path = "config.toml") -> ConfigDict:
    config_path = Path(config_path)
    with config_path.open("rb") as f:
        return tomllib.load(f)


CFG = load_config()


INPUT_DIR = Path(CFG["paths"]["input_dir"])
PROCESSED_DIR = Path(CFG["paths"]["processed_dir"])
FIGURES_DIR = Path(CFG["paths"]["figures_dir"])


def optional_input_file(value: str | None) -> Path | None:
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    return INPUT_DIR / value


SP_DATA = INPUT_DIR / CFG["files"]["sp_data"]
SP_DRIFT = optional_input_file(CFG["files"].get("drift_data"))
TC_DATA = optional_input_file(CFG["files"].get("temp_cond_data"))
HFEM_FILE = optional_input_file(CFG["files"].get("hfem_resistivity"))


DIP_LO = float(CFG["processing"]["dipole_length_m"])
GAUSS_SIGMA = int(CFG["processing"]["gaussian_sigma"])
BOXCAR_M = int(CFG["processing"]["boxcar_m"])
BOXCAR_ITS = int(CFG["processing"]["boxcar_iterations"])


def gauss_filter(x: ArrayLike, sigma: int) -> FloatArray:
    x = np.asarray(x, dtype=float)
    n = len(x)
    y = np.zeros_like(x)

    k_vals = np.arange(-3 * sigma, 3 * sigma + 1)
    hk = (1.0 / np.sqrt(2.0 * np.pi * sigma**2)) * np.exp(
        -(k_vals**2) / (2.0 * sigma**2)
    )

    for i in range(n):
        if (i <= 3 * sigma) or (i >= n - 3 * sigma):
            y[i] = x[i]
        else:
            segment = x[i - 3 * sigma : i + 3 * sigma + 1]
            y[i] = np.sum(segment * hk)

    return y


def exp_smooth(a: float | ArrayLike, pad: int, *arrays: ArrayLike) -> list[FloatArray]:
    arrays = [np.asarray(v, dtype=float) for v in arrays]

    if np.isscalar(a):
        a_vec = np.ones(len(arrays)) * float(a)
    else:
        a_vec = np.asarray(a, dtype=float)
        assert len(a_vec) == len(arrays)

    outputs: list[FloatArray] = []
    for ii, x in enumerate(arrays):
        alpha = a_vec[ii]
        n = len(x)

        y1 = np.zeros_like(x)
        y2 = np.zeros_like(x)
        y = np.zeros_like(x)

        y2[-1] = x[-1]
        for i in range(n - 2, -1, -1):
            y2[i] = alpha * y2[i + 1] + (1 - alpha) * x[i]

        y1[0] = x[0]
        for i in range(1, n):
            y1[i] = alpha * y1[i - 1] + (1 - alpha) * x[i]
            y[i] = (1.0 / (1.0 + alpha)) * (
                y1[i] + y2[i] - (1 - alpha) * x[i]
            )

        y[0] = 0.0 if pad == 0 else x[0]
        outputs.append(y)

    return outputs


def boxcar(x: ArrayLike, m: int, its: int) -> FloatArray:
    x = np.asarray(x, dtype=float)
    n = len(x)
    y = np.zeros_like(x)

    for _ in range(its):
        if m >= 0.5 * n:
            raise ValueError("m must be less than 0.5*len(x)")

        scale = 1.0 / (2 * m + 1)

        i = 0
        y[i] = 0.0
        for k in range(-m, 1):
            y[i] += x[i - k] * scale

        i = 1
        while i <= m:
            if i + m < n:
                y[i] = y[i - 1] + x[i + m] * scale
            else:
                y[i] = y[i - 1]
            i += 1

        while i < n - m:
            y[i] = y[i - 1] + (x[i + m] - x[i - m - 1]) * scale
            i += 1

        while i < n:
            left_index = i - m - 1
            if 0 <= left_index < n:
                y[i] = y[i - 1] - x[left_index] * scale
            else:
                y[i] = y[i - 1]
            i += 1

        y[0] = x[0]
        y[-1] = x[-1]
        x = y.copy()

    return y


def ls_poly(order: int, x: ArrayLike, y: ArrayLike) -> tuple[FloatArray, float, FloatArray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    powers = np.arange(order + 1)
    design_matrix = np.vstack([x**p for p in powers]).T

    normal_matrix = design_matrix.T @ design_matrix
    rhs = design_matrix.T @ y
    coeffs = np.linalg.solve(normal_matrix, rhs)

    fitted = design_matrix @ coeffs

    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return coeffs, float(r2), fitted


def pspec(T: float | ArrayLike, *arrays: ArrayLike) -> tuple[list[FloatArray], list[FloatArray]]:
    arrays = [np.asarray(v, dtype=float) for v in arrays]

    if np.isscalar(T):
        t_vec = np.ones(len(arrays)) * float(T)
    else:
        t_vec = np.asarray(T, dtype=float)
        assert len(t_vec) == len(arrays)

    amps: list[FloatArray] = []
    freqs: list[FloatArray] = []

    for array, dt in zip(arrays, t_vec):
        n = len(array)
        m = int(np.ceil(np.log2(n)))
        radix = 2**m

        dw = 2.0 * np.pi / radix
        transformed = np.fft.fft(array, radix)

        i = np.arange(0, radix // 2 + 1)
        freq = (i * dw) / (2.0 * np.pi * dt)
        transformed = transformed[: len(i)]
        amp = np.abs(transformed)

        amps.append(amp)
        freqs.append(freq)

    return amps, freqs


def simpson(x: ArrayLike, dx: float) -> tuple[float, FloatArray]:
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)

    for n in range(2, len(x)):
        y[n] = (
            y[n - 2]
            + (1.0 / 3.0) * x[n]
            + (4.0 / 3.0) * x[n - 1]
            + (1.0 / 3.0) * x[n - 2]
        )

    area = y[-1] * dx
    return float(area), y


def haversine_dist_km(
    lat_ref_deg: float,
    lon_ref_deg: float,
    lats_deg: ArrayLike,
    lons_deg: ArrayLike,
    earth_radius_m: float,
) -> FloatArray:
    d2r = np.pi / 180.0
    lat_a = d2r * lat_ref_deg
    lon_a = d2r * lon_ref_deg
    lat_b = d2r * np.asarray(lats_deg, dtype=float)
    lon_b = d2r * np.asarray(lons_deg, dtype=float)

    dlon = lon_b - lon_a
    dlat = lat_b - lat_a
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_a) * np.cos(lat_b) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return (earth_radius_m * c) / 1000.0


def delete_indices(arr: ArrayLike, indices: ArrayLike) -> FloatArray:
    array = np.asarray(arr)
    index_array = np.asarray(indices, dtype=int)
    mask = np.ones(len(array), dtype=bool)
    mask[index_array] = False
    return array[mask]


def lin(m: float, x: ArrayLike, b: float) -> FloatArray:
    return m * np.asarray(x, dtype=float) + b


def cond_to_sc(S: ArrayLike, T: ArrayLike) -> FloatArray:
    conductivity = np.asarray(S, dtype=float)
    temperature = np.asarray(T, dtype=float)
    return conductivity / (1.0 + 0.002 * (temperature - 25.0))


def process_code(
    plot: bool = False,
    write: bool = False,
    config_path: str | Path = "config.toml",
) -> ProcessResults:
    del plot
    del config_path

    r_earth = 6_367_000.0
    dL = DIP_LO

    # ---- SP data ----
    sp_df = pd.read_csv(SP_DATA)

    # Expect Rio Grande-style columns: Segment_ID, ..., Longitude, Latitude, Measured_Voltage_millivolts
    spid = sp_df.iloc[:, 0].values
    spx = sp_df.iloc[:, 5].values
    spy = sp_df.iloc[:, 6].values
    spmv = sp_df.iloc[:, 7].values

    segments: dict[int, dict[str, FloatArray]] = {}
    for uid in np.unique(spid):
        mask = spid == uid
        segments[int(uid)] = {
            "SPX": np.asarray(spx[mask], dtype=float),
            "SPY": np.asarray(spy[mask], dtype=float),
            "SPmV": np.asarray(spmv[mask], dtype=float),
        }

    segment_ids = sorted(segments.keys())

    spd: dict[int, FloatArray] = {}
    spd_survey: dict[int, FloatArray] = {}
    # Distances along each segment, and survey-wide distances relative to first segment
    first_seg = segment_ids[0]
    spx_ref = segments[first_seg]["SPX"]
    spy_ref = segments[first_seg]["SPY"]

    for seg_id in segment_ids:
        spx_i = segments[seg_id]["SPX"]
        spy_i = segments[seg_id]["SPY"]
        spd_i = haversine_dist_km(spy_i[0], spx_i[0], spy_i, spx_i, r_earth)
        spd[seg_id] = spd_i

        spd_survey_i = haversine_dist_km(spy_ref[0], spx_ref[0], spy_i, spx_i, r_earth)
        spd_survey[seg_id] = spd_survey_i

    # NOTE: segment-specific deletions from MATLAB are omitted here to keep
    # workflow general over arbitrary segment IDs.

    # ---- Drift correction (optional) ----
    has_drift = SP_DRIFT is not None and SP_DRIFT.exists()

    drift_segments: dict[int, dict[str, FloatArray]] = {}
    drift_corrs = np.empty((0, 4), dtype=float)
    spmv_corr: dict[int, FloatArray] = {}

    if has_drift:
        drift_df = pd.read_csv(SP_DRIFT)
        drift_id = drift_df.iloc[:, 0].values
        drift_n = drift_df.iloc[:, 2].values
        drift_dv = drift_df.iloc[:, 4].values

        for uid in np.unique(drift_id):
            mask = drift_id == uid
            drift_segments[int(uid)] = {
                "N": np.asarray(drift_n[mask], dtype=float),
                "dV": np.asarray(drift_dv[mask], dtype=float),
            }

        drift_corrs_list: list[list[float]] = []
        for seg_id in segment_ids:
            if seg_id not in drift_segments:
                raise ValueError(f"Missing drift data for segment {seg_id}")
            dv = drift_segments[seg_id]["dV"]
            t = np.arange(1, len(dv) + 1, dtype=float)
            coeffs, r2, _ = ls_poly(order=1, x=t, y=dv)
            drift_corrs_list.append([float(seg_id), float(coeffs[1]), float(coeffs[0]), r2])
        drift_corrs = np.array(drift_corrs_list, dtype=float)

        for idx, seg_id in enumerate(segment_ids):
            sp = segments[seg_id]["SPmV"]
            x = np.arange(1, len(sp) + 1, dtype=float)
            m = drift_corrs[idx, 1]
            b = drift_corrs[idx, 2]
            trend = lin(float(m), x, float(b))
            spmv_corr[seg_id] = sp - trend
    else:
        drift_corrs = np.empty((0, 4), dtype=float)
        spmv_corr = {seg_id: segments[seg_id]["SPmV"].copy() for seg_id in segment_ids}

    # Simple inter-segment shifting analogous to original code, but only if we have multiple segments
    if len(segment_ids) >= 1:
        first = segment_ids[0]
        spmv_corr[first] = spmv_corr[first] - spmv_corr[first][0]

    for i in range(1, len(segment_ids)):
        prev = segment_ids[i - 1]
        curr = segment_ids[i]
        shift = spmv_corr[curr][0] - spmv_corr[prev][-1]
        spmv_corr[curr] = spmv_corr[curr] - shift

    # Build concatenated SP series: first pair and second pair, if available
    if len(segment_ids) >= 2:
        pair1 = segment_ids[:2]
        spmv12c = np.concatenate([spmv_corr[pair1[0]], spmv_corr[pair1[1]]])
    else:
        pair1 = segment_ids[:1]
        spmv12c = spmv_corr[pair1[0]].copy()

    if len(segment_ids) >= 4:
        pair2 = segment_ids[2:4]
        spmv34c = np.concatenate([spmv_corr[pair2[0]], spmv_corr[pair2[1]]])
    elif len(segment_ids) >= 3:
        pair2 = segment_ids[2:3]
        spmv34c = spmv_corr[pair2[0]].copy()
    else:
        pair2 = []
        spmv34c = np.array([], dtype=float)

    # Drift FFT spectra only if drift exists
    if has_drift:
        _amps, _freqs = pspec(
            [1] * len(segment_ids),
            *[drift_segments[s]["dV"] for s in segment_ids],
        )

    # ---- Filtering and partitioning for first concatenated series ----
    exp_12 = exp_smooth(0.9, 1, spmv12c)[0]
    gauss_12 = gauss_filter(spmv12c, sigma=GAUSS_SIGMA)
    edge_12 = min(120, len(gauss_12))
    gauss_12[:edge_12] = exp_12[:edge_12]
    gauss_12[-edge_12:] = exp_12[-edge_12:]
    dvl12 = gauss_12.copy()
    dvhn12 = spmv12c - dvl12

    dvh12 = boxcar(dvhn12, m=BOXCAR_M, its=BOXCAR_ITS)
    dvn12 = dvhn12 - dvh12

    e12 = -spmv12c * dL
    _, v12 = simpson(e12, dL)
    v12 = -v12

    el12 = -dvl12 * dL
    _, vl12 = simpson(el12, dL)
    vl12 = -vl12

    eh12 = -dvh12 * dL
    _, vh12 = simpson(eh12, dL)
    vh12 = -vh12

    en12 = -dvn12 * dL
    _, vn12 = simpson(en12, dL)
    vn12 = -vn12

    # ---- Filtering and partitioning for second concatenated series (optional) ----
    if len(spmv34c) > 0:
        exp_34 = exp_smooth(0.9, 1, spmv34c)[0]
        gauss_34 = gauss_filter(spmv34c, sigma=GAUSS_SIGMA)
        edge_34 = min(120, len(gauss_34))
        gauss_34[:edge_34] = exp_34[:edge_34]
        gauss_34[-edge_34:] = exp_34[-edge_34:]
        dvl34 = gauss_34.copy()
        dvhn34 = spmv34c - dvl34

        dvh34 = boxcar(dvhn34, m=BOXCAR_M, its=BOXCAR_ITS)
        dvn34 = dvhn34 - dvh34

        e34 = -spmv34c * dL
        _, v34 = simpson(e34, dL)
        v34 = -v34

        el34 = -dvl34 * dL
        _, vl34 = simpson(el34, dL)
        vl34 = -vl34

        eh34 = -dvh34 * dL
        _, vh34 = simpson(eh34, dL)
        vh34 = -vh34

        en34 = -dvn34 * dL
        _, vn34 = simpson(en34, dL)
        vn34 = -vn34
    else:
        dvl34 = np.array([], dtype=float)
        dvh34 = np.array([], dtype=float)
        dvn34 = np.array([], dtype=float)
        v34 = np.array([], dtype=float)
        vl34 = np.array([], dtype=float)
        vh34 = np.array([], dtype=float)
        vn34 = np.array([], dtype=float)

    # ---- Temperature/conductivity processing (optional) ----
    has_tc = TC_DATA is not None and TC_DATA.exists()

    tc_segments: dict[int, dict[str, FloatArray]] = {}
    temp_corrs = np.empty((0, 4), dtype=float)
    cond_corrs = np.empty((0, 4), dtype=float)
    sc_corrs = np.empty((0, 4), dtype=float)
    t_corr: dict[int, FloatArray] = {}
    s_corr: dict[int, FloatArray] = {}
    sc_corr: dict[int, FloatArray] = {}

    if has_tc:
        tc_df = pd.read_csv(TC_DATA)
        stid = tc_df.iloc[:, 0].values
        stx = tc_df.iloc[:, 5].values
        sty = tc_df.iloc[:, 6].values
        temp = tc_df.iloc[:, 7].values
        cond = tc_df.iloc[:, 8].values

        first_stid = int(np.unique(stid)[0])
        stx1 = np.asarray(stx[stid == first_stid], dtype=float)
        sty1 = np.asarray(sty[stid == first_stid], dtype=float)

        for uid in np.unique(stid):
            mask = stid == uid
            stx_i = np.asarray(stx[mask], dtype=float)
            sty_i = np.asarray(sty[mask], dtype=float)
            temp_i = np.asarray(temp[mask], dtype=float)
            cond_i = np.asarray(cond[mask], dtype=float)
            sc_i = cond_to_sc(cond_i, temp_i)

            std_i = haversine_dist_km(sty_i[0], stx_i[0], sty_i, stx_i, r_earth) * 1000.0
            survey_dist_i = haversine_dist_km(sty1[0], stx1[0], sty_i, stx_i, r_earth) * 1000.0

            tc_segments[int(uid)] = {
                "STX": stx_i,
                "STY": sty_i,
                "T": temp_i,
                "S": cond_i,
                "SC": sc_i,
                "STd": std_i,
                "STD": survey_dist_i,
            }

        seg_tc_ids = sorted(tc_segments.keys())

        temp_corrs_list: list[list[float]] = []
        for seg_id in seg_tc_ids:
            temp_i = tc_segments[seg_id]["T"]
            t_vec = np.arange(1, len(temp_i) + 1, dtype=float)
            coeffs, r2, _ = ls_poly(order=1, x=t_vec, y=temp_i)
            temp_corrs_list.append([float(seg_id), float(coeffs[1]), float(coeffs[0]), r2])
        temp_corrs = np.array(temp_corrs_list, dtype=float)

        for idx, seg_id in enumerate(seg_tc_ids):
            temp_i = tc_segments[seg_id]["T"]
            x_vec = np.arange(1, len(temp_i) + 1, dtype=float)
            m = temp_corrs[idx, 1]
            b = temp_corrs[idx, 2]
            t_corr[seg_id] = temp_i - lin(float(m), x_vec, float(b))

        cond_corrs_list: list[list[float]] = []
        for seg_id in seg_tc_ids:
            cond_i = tc_segments[seg_id]["S"]
            t_vec = np.arange(1, len(cond_i) + 1, dtype=float)
            coeffs, r2, _ = ls_poly(order=1, x=t_vec, y=cond_i)
            cond_corrs_list.append([float(seg_id), float(coeffs[1]), float(coeffs[0]), r2])
        cond_corrs = np.array(cond_corrs_list, dtype=float)

        for idx, seg_id in enumerate(seg_tc_ids):
            cond_i = tc_segments[seg_id]["S"]
            x_vec = np.arange(1, len(cond_i) + 1, dtype=float)
            m = cond_corrs[idx, 1]
            b = cond_corrs[idx, 2]
            s_corr[seg_id] = cond_i - lin(float(m), x_vec, float(b))

        sc_corrs_list: list[list[float]] = []
        for seg_id in seg_tc_ids:
            sc_i = tc_segments[seg_id]["SC"]
            t_vec = np.arange(1, len(sc_i) + 1, dtype=float)
            coeffs, r2, _ = ls_poly(order=1, x=t_vec, y=sc_i)
            sc_corrs_list.append([float(seg_id), float(coeffs[1]), float(coeffs[0]), r2])
        sc_corrs = np.array(sc_corrs_list, dtype=float)

        for idx, seg_id in enumerate(seg_tc_ids):
            sc_i = tc_segments[seg_id]["SC"]
            x_vec = np.arange(1, len(sc_i) + 1, dtype=float)
            m = sc_corrs[idx, 1]
            b = sc_corrs[idx, 2]
            sc_corr[seg_id] = sc_i - lin(float(m), x_vec, float(b))

    # ---- Writing outputs ----
    if write:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        # Gradient SP output
        grad_rows: list[FloatArray] = []
        for seg_id in segment_ids:
            n = len(segments[seg_id]["SPX"])
            seg_col = np.full(n, seg_id, dtype=float)
            rows = np.column_stack(
                [
                    seg_col,
                    segments[seg_id]["SPX"],
                    segments[seg_id]["SPY"],
                    spd[seg_id],
                    spd_survey[seg_id],
                    segments[seg_id]["SPmV"],
                    spmv_corr[seg_id],
                ]
            )
            grad_rows.append(rows)

        grad_all = np.vstack(grad_rows)
        grad_cols = [
            "segment_id",
            "x",
            "y",
            "segment_distance_km",
            "survey_distance_km",
            "raw_SP_mV",
            "drift_corrected_SP_mV",
        ]
        pd.DataFrame(grad_all, columns=grad_cols).to_csv(
            PROCESSED_DIR / "Gradient_Self_Potential_python.csv", index=False
        )

        # Electric potential output (only for concatenated series we actually have)
        interp_blocks: list[FloatArray] = []

        # First concatenated series
        spx12 = np.concatenate([segments[s]["SPX"] for s in pair1])
        spy12 = np.concatenate([segments[s]["SPY"] for s in pair1])
        seg_ids_12 = np.concatenate(
            [np.full(len(segments[s]["SPX"]), s) for s in pair1]
        )
        interp12 = np.column_stack(
            [
                seg_ids_12,
                spx12,
                spy12,
                spmv12c,
                dvl12,
                dvhn12,
                dvh12,
                dvn12,
                v12,
                vl12,
                vh12,
                vn12,
            ]
        )
        interp_blocks.append(interp12)

        # Second concatenated series, if present
        if len(spmv34c) > 0 and len(pair2) > 0:
            spx34 = np.concatenate([segments[s]["SPX"] for s in pair2])
            spy34 = np.concatenate([segments[s]["SPY"] for s in pair2])
            seg_ids_34 = np.concatenate(
                [np.full(len(segments[s]["SPX"]), s) for s in pair2]
            )
            interp34 = np.column_stack(
                [
                    seg_ids_34,
                    spx34,
                    spy34,
                    spmv34c,
                    dvl34,
                    dvhn34,
                    dvh34,
                    dvn34,
                    v34,
                    vl34,
                    vh34,
                    vn34,
                ]
            )
            interp_blocks.append(interp34)

        interp_all = np.vstack(interp_blocks)
        interp_cols = [
            "segment_id",
            "x",
            "y",
            "SPmV_drift_corrected",
            "DVL_lowfreq",
            "DVHN_high_plus_noise",
            "DVH_highfreq",
            "DVN_noise",
            "V_full",
            "VL_lowfreq",
            "VH_highfreq",
            "VN_noise",
        ]
        pd.DataFrame(interp_all, columns=interp_cols).to_csv(
            PROCESSED_DIR / "Electric_Potential_python.csv", index=False
        )

        # Temperature/conductivity outputs only if we had tc data
        if has_tc:
            temp_rows: list[FloatArray] = []
            seg_tc_ids = sorted(tc_segments.keys())
            for seg_id in seg_tc_ids:
                seg_len = len(tc_segments[seg_id]["STX"])
                seg_col = np.full(seg_len, seg_id, dtype=float)
                rows = np.column_stack(
                    [
                        seg_col,
                        tc_segments[seg_id]["STX"],
                        tc_segments[seg_id]["STY"],
                        tc_segments[seg_id]["STd"],
                        tc_segments[seg_id]["STD"],
                        tc_segments[seg_id]["S"],
                        s_corr[seg_id],
                        tc_segments[seg_id]["T"],
                        t_corr[seg_id],
                        tc_segments[seg_id]["SC"],
                        sc_corr[seg_id],
                    ]
                )
                temp_rows.append(rows)

            temp_all = np.vstack(temp_rows)
            temp_cols = [
                "segment_id",
                "x",
                "y",
                "segment_distance_m",
                "survey_distance_m",
                "cond_uS_cm",
                "cond_trend_corrected",
                "temp_degC",
                "temp_trend_corrected",
                "spec_cond_uS_cm",
                "spec_cond_trend_corrected",
            ]
            pd.DataFrame(temp_all, columns=temp_cols).to_csv(
                PROCESSED_DIR / "Temperature_Conductivity_python.csv", index=False
            )

        # Drift correction summary (only if drift was present)
        if has_drift and drift_corrs.size > 0:
            pd.DataFrame(
                drift_corrs, columns=["segment_id", "slope_m", "intercept_b", "r2"]
            ).to_csv(PROCESSED_DIR / "Drift_Correction_python.csv", index=False)

        if has_tc and temp_corrs.size > 0:
            pd.DataFrame(
                temp_corrs, columns=["segment_id", "slope_m", "intercept_b", "r2"]
            ).to_csv(PROCESSED_DIR / "Temperature_Correction_python.csv", index=False)

        if has_tc and cond_corrs.size > 0:
            pd.DataFrame(
                cond_corrs, columns=["segment_id", "slope_m", "intercept_b", "r2"]
            ).to_csv(PROCESSED_DIR / "Conductivity_Correction_python.csv", index=False)

        if has_tc and sc_corrs.size > 0:
            pd.DataFrame(
                sc_corrs, columns=["segment_id", "slope_m", "intercept_b", "r2"]
            ).to_csv(
                PROCESSED_DIR / "Specific_Conductance_Correction_python.csv", index=False
            )

    return {
        "SPmV12c": spmv12c,
        "SPmV34c": spmv34c,
        "DVL12": dvl12,
        "DVL34": dvl34,
        "DVH12": dvh12,
        "DVH34": dvh34,
        "DVN12": dvn12,
        "DVN34": dvn34,
        "V12": v12,
        "V34": v34,
        "VL12": vl12,
        "VL34": vl34,
        "VH12": vh12,
        "VH34": vh34,
        "VN12": vn12,
        "VN34": vn34,
    }


if __name__ == "__main__":
    results = process_code(plot=False, write=True)