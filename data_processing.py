from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import tomllib


# ---------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------

def load_config(config_path: str | Path = "config.toml") -> dict:
    config_path = Path(config_path)
    with config_path.open("rb") as f:
        return tomllib.load(f)


CFG = load_config()

# Base directories
INPUT_DIR = Path(CFG["paths"]["input_dir"])
PROCESSED_DIR = Path(CFG["paths"]["processed_dir"])
FIGURES_DIR = Path(CFG["paths"]["figures_dir"])

# Input file names (resolved relative to INPUT_DIR)
SP_DATA = INPUT_DIR / CFG["files"]["sp_data"]
SP_DRIFT = INPUT_DIR / CFG["files"]["drift_data"]
TC_DATA = INPUT_DIR / CFG["files"]["temp_cond_data"]
HFEM_FILE = INPUT_DIR / CFG["files"]["hfem_resistivity"]  # not used yet, but wired in

# Processing parameters
DIP_LO = float(CFG["processing"]["dipole_length_m"])
GAUSS_SIGMA = int(CFG["processing"]["gaussian_sigma"])
BOXCAR_M = int(CFG["processing"]["boxcar_m"])
BOXCAR_ITS = int(CFG["processing"]["boxcar_iterations"])


# ---------------------------------------------------------------------
# Utility functions translated from the MATLAB subroutines
# ---------------------------------------------------------------------

def gauss_filter(x, sigma):
    """
    Gaussian despiking filter (GaussFilter.m).
    x: 1D array
    sigma: half-width in samples
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    y = np.zeros_like(x)

    # Precompute Gaussian kernel over [-3*sigma, 3*sigma]
    k_vals = np.arange(-3 * sigma, 3 * sigma + 1)
    hk = (1.0 / np.sqrt(2.0 * np.pi * sigma**2)) * np.exp(-(k_vals**2) / (2.0 * sigma**2))

    for i in range(n):
        if (i <= 3 * sigma) or (i >= n - 3 * sigma):
            # keep original at ends, as in MATLAB code
            y[i] = x[i]
        else:
            segment = x[i - 3 * sigma : i + 3 * sigma + 1]
            y[i] = np.sum(segment * hk)

    return y


def exp_smooth(a, pad, *arrays):
    """
    EXPsmooth.m — parallel exponential smoothing (causal + anti-causal).
    a: scalar or array-like of smoothing factors, 0 <= a < 1
    pad: 0 or 1 (how to handle first value)
    arrays: one or more 1D sequences to smooth
    Returns list of arrays.
    """
    arrays = [np.asarray(v, dtype=float) for v in arrays]

    if np.isscalar(a):
        a_vec = np.ones(len(arrays)) * a
    else:
        a_vec = np.asarray(a, dtype=float)
        assert len(a_vec) == len(arrays)

    outputs = []
    for ii, x in enumerate(arrays):
        alpha = a_vec[ii]
        n = len(x)

        y1 = np.zeros_like(x)
        y2 = np.zeros_like(x)
        y = np.zeros_like(x)

        # anti-causal (reverse)
        y2[-1] = x[-1]
        for i in range(n - 2, -1, -1):
            y2[i] = alpha * y2[i + 1] + (1 - alpha) * x[i]

        # causal and combine
        y1[0] = x[0]
        for i in range(1, n):
            y1[i] = alpha * y1[i - 1] + (1 - alpha) * x[i]
            y[i] = (1.0 / (1.0 + alpha)) * (y1[i] + y2[i] - (1 - alpha) * x[i])

        # fix first value
        if pad == 0:
            y[0] = 0.0
        else:
            y[0] = x[0]

        outputs.append(y)

    return outputs


def boxcar(x, m, its):
    """
    Boxcarv1.m — fast moving-average filter (window length 2*m+1), iterated its times.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    y = np.zeros_like(x)

    for _ in range(its):
        # enforce m < 0.5 * n (as in MATLAB)
        if m >= 0.5 * n:
            raise ValueError("m must be less than 0.5*len(x)")

        scale = 1.0 / (2 * m + 1)

        # initialize first point using slow version (assuming out-of-bounds = 0)
        i = 0
        y[i] = 0.0
        for k in range(-m, 1):  # -m ... 0
            y[i] += x[i - k] * scale

        # second to (m+1)th point (still near left edge)
        i = 1
        while i <= m:
            if i + m < n:
                y[i] = y[i - 1] + x[i + m] * scale
            else:
                y[i] = y[i - 1]
            i += 1

        # interior points
        while i < n - m:
            y[i] = y[i - 1] + (x[i + m] - x[i - m - 1]) * scale
            i += 1

        # right edge
        while i < n:
            left_index = i - m - 1
            if 0 <= left_index < n:
                y[i] = y[i - 1] - x[left_index] * scale
            else:
                y[i] = y[i - 1]
            i += 1

        # fix endpoints for next iteration
        y[0] = x[0]
        y[-1] = x[-1]
        x = y.copy()

    return y


def ls_poly(order, x, y):
    """
    LSPoly.m — least-squares polynomial fit of given order.
    Returns coefficients C (length order+1) and r^2.
    Polynomial: P = C[0] + C[1]*X + ... + C[order]*X^order
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    powers = np.arange(order + 1)
    F = np.vstack([x**p for p in powers]).T  # shape (n, order+1)

    # solve (F^T F) C = F^T y
    A = F.T @ F
    B = F.T @ y
    C = np.linalg.solve(A, B)

    # fitted polynomial
    P = F @ C

    # r^2
    ss_res = np.sum((y - P) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return C, r2, P


def pspec(T, *arrays):
    """
    Pspec.m — FFT amplitude spectra.
    T: scalar or array-like of sample periods (s). One per input in arrays.
    Returns (Amplitude_list, Frequency_list)
    """
    arrays = [np.asarray(v, dtype=float) for v in arrays]

    if np.isscalar(T):
        T_vec = np.ones(len(arrays)) * T
    else:
        T_vec = np.asarray(T, dtype=float)
        assert len(T_vec) == len(arrays)

    amps = []
    freqs = []
    for a, dt in zip(arrays, T_vec):
        n = len(a)

        # next power-of-two length
        m = int(np.ceil(np.log2(n)))
        radix = 2**m

        dw = 2.0 * np.pi / radix
        transformed = np.fft.fft(a, radix)

        i = np.arange(0, radix // 2 + 1)
        freq = (i * dw) / (2.0 * np.pi * dt)
        transformed = transformed[: len(i)]
        amp = np.abs(transformed)

        amps.append(amp)
        freqs.append(freq)

    return amps, freqs


def simpson(x, dx):
    """
    Simpson.m — difference-equation Simpson integration.
    x: array of "E" values
    dx: scalar dipole length
    Returns (area, y) where y is cumulative integral.
    """
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    for n in range(2, len(x)):
        y[n] = y[n - 2] + (1.0 / 3.0) * x[n] + (4.0 / 3.0) * x[n - 1] + (1.0 / 3.0) * x[n - 2]
    area = y[-1] * dx
    return area, y


# ---------------------------------------------------------------------
# Main processing function translated from Process_code.m
# ---------------------------------------------------------------------

def process_code(plot=False, write=False, config_path: str | Path = "config.toml"):
    # If you ever want runtime override, reload cfg here.
    # For now we just reuse the globals already loaded.

    d2r = np.pi / 180.0
    r_earth = 6_367_000.0  # Earth radius (m)
    dL = DIP_LO            # dipole length (m) from config

    # ------------------------------------------------------------------
    # SELF-POTENTIAL DATA INTO SEGMENTS (1–4)
    # ------------------------------------------------------------------
    sp_df = pd.read_csv(SP_DATA)

    SPID = sp_df.iloc[:, 0].values
    SPX = sp_df.iloc[:, 5].values  # MATLAB col 6 (1-based)
    SPY = sp_df.iloc[:, 6].values  # MATLAB col 7
    SPmV = sp_df.iloc[:, 7].values  # MATLAB col 8

    segments = {}
    for uid in np.unique(SPID):
        mask = SPID == uid
        segments[int(uid)] = {
            "SPX": SPX[mask],
            "SPY": SPY[mask],
            "SPmV": SPmV[mask],
        }

    def haversine_dist_km(lat_ref_deg, lon_ref_deg, lats_deg, lons_deg):
        latA = d2r * lat_ref_deg
        lonA = d2r * lon_ref_deg
        latB = d2r * lats_deg
        lonB = d2r * lons_deg
        dlon = lonB - lonA
        dlat = latB - latA
        a = np.sin(dlat / 2.0) ** 2 + np.cos(latA) * np.cos(latB) * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        return (r_earth * c) / 1000.0

    # compute both segment-wise distance (SPd) and survey-wise distance (SPD)
    SPd = {}
    SPD = {}
    for seg_id in [1, 2, 3, 4]:
        SPX_i = segments[seg_id]["SPX"]
        SPY_i = segments[seg_id]["SPY"]
        SPd_i = haversine_dist_km(SPY_i[0], SPX_i[0], SPY_i, SPX_i)
        SPd[seg_id] = SPd_i

        # Survey distance relative to segment 1 start
        SPX1 = segments[1]["SPX"]
        SPY1 = segments[1]["SPY"]
        SPD_i = haversine_dist_km(SPY1[0], SPX1[0], SPY_i, SPX_i)
        SPD[seg_id] = SPD_i

    # Manual noise removal (exact indices as in MATLAB)
    def delete_indices(arr, indices):
        mask = np.ones(len(arr), dtype=bool)
        mask[indices] = False
        return arr[mask]

    # Segment 2: several index ranges
    for start, end in [(10075, 10095), (9550, 9575), (7290, 7301), (4140, 4240)]:
        idx = np.arange(start - 1, end)  # MATLAB is 1-based; Python 0-based
        for key in ["SPX", "SPY", "SPmV"]:
            segments[2][key] = delete_indices(segments[2][key], idx)
        SPd[2] = delete_indices(SPd[2], idx)
        SPD[2] = delete_indices(SPD[2], idx)

    # Segment 3: truncate from 8138:end
    cut3 = np.arange(8138 - 1, len(segments[3]["SPmV"]))
    for key in ["SPX", "SPY", "SPmV"]:
        segments[3][key] = delete_indices(segments[3][key], cut3)
    SPd[3] = delete_indices(SPd[3], cut3)
    SPD[3] = delete_indices(SPD[3], cut3)

    # Segment 4: 10750:11000
    cut4 = np.arange(10750 - 1, 11000)
    for key in ["SPX", "SPY", "SPmV"]:
        segments[4][key] = delete_indices(segments[4][key], cut4)
    SPd[4] = delete_indices(SPd[4], cut4)
    SPD[4] = delete_indices(SPD[4], cut4)

    # ------------------------------------------------------------------
    # DRIFT DATA AND DRIFT CORRECTION PARAMETERS
    # ------------------------------------------------------------------
    drift_df = pd.read_csv(SP_DRIFT)
    DRIFTID = drift_df.iloc[:, 0].values
    DRIFTN = drift_df.iloc[:, 2].values
    DRIFTdV = drift_df.iloc[:, 4].values

    drift_segments = {}
    for uid in np.unique(DRIFTID):
        mask = DRIFTID == uid
        drift_segments[int(uid)] = {
            "N": DRIFTN[mask],
            "dV": DRIFTdV[mask],
        }

    drift_corrs = []
    for seg_id in [1, 2, 3, 4]:
        dV = drift_segments[seg_id]["dV"]
        t = np.arange(1, len(dV) + 1)
        C, r2, _ = ls_poly(order=1, x=t, y=dV)
        m = C[1]
        b = C[0]
        drift_corrs.append([seg_id, m, b, r2])
    drift_corrs = np.array(drift_corrs)  # columns: ID, m, b, R2

    # ------------------------------------------------------------------
    # APPLY DRIFT CORRECTION TO RAW VOLTAGE (SPmV)
    # ------------------------------------------------------------------
    def lin(m, x, b):
        return m * x + b

    SPmV_corr = {}
    for seg_id in [1, 2, 3, 4]:
        sp = segments[seg_id]["SPmV"]
        x = np.arange(1, len(sp) + 1)
        m = drift_corrs[seg_id - 1, 1]
        b = drift_corrs[seg_id - 1, 2]
        trend = lin(m, x, b)
        SPmV_corr[seg_id] = sp - trend

    # Remove DC offsets between segments 1 and 2; 3 and 4 unchanged
    SPmV_corr[1] = SPmV_corr[1] - SPmV_corr[1][0]
    shift12 = SPmV_corr[2][0] - SPmV_corr[1][-1]
    SPmV_corr[2] = SPmV_corr[2] - shift12
    SPmV_corr[3] = SPmV_corr[3] - SPmV_corr[3][0]

    # Combine segments into 1–2 and 3–4
    SPmV12c = np.concatenate([SPmV_corr[1], SPmV_corr[2]])
    SPmV34c = np.concatenate([SPmV_corr[3], SPmV_corr[4]])

    # ------------------------------------------------------------------
    # FOURIER TRANSFORMS OF DRIFT DATA (optional)
    # ------------------------------------------------------------------
    amps, freqs = pspec(
        [1, 1, 1, 1],
        drift_segments[1]["dV"],
        drift_segments[2]["dV"],
        drift_segments[3]["dV"],
        drift_segments[4]["dV"],
    )

    # ------------------------------------------------------------------
    # SIGNAL PROCESSING OF GRADIENT SP DATA
    # ------------------------------------------------------------------
    # Combined 1–2
    EXP = exp_smooth(0.9, 1, SPmV12c)[0]
    GAUSS = gauss_filter(SPmV12c, sigma=GAUSS_SIGMA)
    GAUSS[:120] = EXP[:120]
    GAUSS[-120:] = EXP[-120:]
    DVL12 = GAUSS.copy()
    DVHN12 = SPmV12c - DVL12

    # Combined 3–4
    EXP = exp_smooth(0.9, 1, SPmV34c)[0]
    GAUSS = gauss_filter(SPmV34c, sigma=GAUSS_SIGMA)
    GAUSS[:120] = EXP[:120]
    GAUSS[-120:] = EXP[-120:]
    DVL34 = GAUSS.copy()
    DVHN34 = SPmV34c - DVL34

    # Partition H and N components using moving-average filter
    DVH12 = boxcar(DVHN12, m=BOXCAR_M, its=BOXCAR_ITS)
    DVH34 = boxcar(DVHN34, m=BOXCAR_M, its=BOXCAR_ITS)
    DVN12 = DVHN12 - DVH12
    DVN34 = DVHN34 - DVH34

    # ------------------------------------------------------------------
    # INTEGRATION TO ELECTRIC POTENTIAL (full, L, H, N components)
    # ------------------------------------------------------------------
    E12 = -SPmV12c * dL
    _, V12 = simpson(E12, dL)
    V12 = -V12

    E34 = -SPmV34c * dL
    _, V34 = simpson(E34, dL)
    V34 = -V34

    EL12 = -DVL12 * dL
    _, VL12 = simpson(EL12, dL)
    VL12 = -VL12

    EL34 = -DVL34 * dL
    _, VL34 = simpson(EL34, dL)
    VL34 = -VL34

    EH12 = -DVH12 * dL
    _, VH12 = simpson(EH12, dL)
    VH12 = -VH12

    EH34 = -DVH34 * dL
    _, VH34 = simpson(EH34, dL)
    VH34 = -VH34

    EN12 = -DVN12 * dL
    _, VN12 = simpson(EN12, dL)
    VN12 = -VN12

    EN34 = -DVN34 * dL
    _, VN34 = simpson(EN34, dL)
    VN34 = -VN34

    # ------------------------------------------------------------------
    # TEMPERATURE AND CONDUCTIVITY DATA PROCESSING
    # ------------------------------------------------------------------
    def cond_to_sc(S, T):
        return S / (1.0 + 0.002 * (T - 25.0))

    tc_df = pd.read_csv(TC_DATA)
    STID = tc_df.iloc[:, 0].values
    STX = tc_df.iloc[:, 5].values
    STY = tc_df.iloc[:, 6].values
    T = tc_df.iloc[:, 7].values
    S = tc_df.iloc[:, 8].values

    tc_segments = {}
    for uid in np.unique(STID):
        mask = STID == uid
        STX_i = STX[mask]
        STY_i = STY[mask]
        T_i = T[mask]
        S_i = S[mask]
        SC_i = cond_to_sc(S_i, T_i)

        STd_i = haversine_dist_km(STY_i[0], STX_i[0], STY_i, STX_i) * 1000.0

        STX1 = STX[STID == np.unique(STID)[0]]
        STY1 = STY[STID == np.unique(STID)[0]]
        STD_i = haversine_dist_km(STY1[0], STX1[0], STY_i, STX_i) * 1000.0

        tc_segments[int(uid)] = dict(
            STX=STX_i,
            STY=STY_i,
            T=T_i,
            S=S_i,
            SC=SC_i,
            STd=STd_i,
            STD=STD_i,
        )

    # Temperature drift corrections per segment
    temp_corrs = []
    for seg_id in [1, 2, 3, 4]:
        T_i = tc_segments[seg_id]["T"]
        t = np.arange(1, len(T_i) + 1)
        C, r2, _ = ls_poly(order=1, x=t, y=T_i)
        temp_corrs.append([seg_id, C[1], C[0], r2])
    temp_corrs = np.array(temp_corrs)

    T_corr = {}
    for seg_id in [1, 2, 3, 4]:
        T_i = tc_segments[seg_id]["T"]
        x = np.arange(1, len(T_i) + 1)
        m = temp_corrs[seg_id - 1, 1]
        b = temp_corrs[seg_id - 1, 2]
        T_corr[seg_id] = T_i - lin(m, x, b)

    # Conductivity drift corrections
    cond_corrs = []
    for seg_id in [1, 2, 3, 4]:
        S_i = tc_segments[seg_id]["S"]
        t = np.arange(1, len(S_i) + 1)
        C, r2, _ = ls_poly(order=1, x=t, y=S_i)
        cond_corrs.append([seg_id, C[1], C[0], r2])
    cond_corrs = np.array(cond_corrs)

    S_corr = {}
    for seg_id in [1, 2, 3, 4]:
        S_i = tc_segments[seg_id]["S"]
        x = np.arange(1, len(S_i) + 1)
        m = cond_corrs[seg_id - 1, 1]
        b = cond_corrs[seg_id - 1, 2]
        S_corr[seg_id] = S_i - lin(m, x, b)

    # Specific conductance trend corrections
    sc_corrs = []
    for seg_id in [1, 2, 3, 4]:
        SC_i = tc_segments[seg_id]["SC"]
        t = np.arange(1, len(SC_i) + 1)
        C, r2, _ = ls_poly(order=1, x=t, y=SC_i)
        sc_corrs.append([seg_id, C[1], C[0], r2])
    sc_corrs = np.array(sc_corrs)

    SC_corr = {}
    for seg_id in [1, 2, 3, 4]:
        SC_i = tc_segments[seg_id]["SC"]
        x = np.arange(1, len(SC_i) + 1)
        m = sc_corrs[seg_id - 1, 1]
        b = sc_corrs[seg_id - 1, 2]
        SC_corr[seg_id] = SC_i - lin(m, x, b)

    # ------------------------------------------------------------------
    # OUTPUTS
    # ------------------------------------------------------------------
    if write:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        # Gradient SP for each segment
        grad_rows = []
        for seg_id in [1, 2, 3, 4]:
            n = len(segments[seg_id]["SPX"])
            seg_col = np.full(n, seg_id)
            rows = np.column_stack(
                [
                    seg_col,
                    segments[seg_id]["SPX"],
                    segments[seg_id]["SPY"],
                    SPd[seg_id],
                    SPD[seg_id],
                    segments[seg_id]["SPmV"],
                    SPmV_corr[seg_id],
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

        # Combined interpretation segments (1–2 and 3–4)
        SPX12 = np.concatenate([segments[1]["SPX"], segments[2]["SPX"]])
        SPY12 = np.concatenate([segments[1]["SPY"], segments[2]["SPY"]])
        seg_ids_12 = np.concatenate(
            [np.full(len(segments[1]["SPX"]), 1), np.full(len(segments[2]["SPX"]), 2)]
        )
        interp12 = np.column_stack(
            [
                seg_ids_12,
                SPX12,
                SPY12,
                SPmV12c,
                DVL12,
                DVHN12,
                DVH12,
                DVN12,
                V12,
                VL12,
                VH12,
                VN12,
            ]
        )

        SPX34 = np.concatenate([segments[3]["SPX"], segments[4]["SPX"]])
        SPY34 = np.concatenate([segments[3]["SPY"], segments[4]["SPY"]])
        seg_ids_34 = np.concatenate(
            [np.full(len(segments[3]["SPX"]), 3), np.full(len(segments[4]["SPX"]), 4)]
        )
        interp34 = np.column_stack(
            [
                seg_ids_34,
                SPX34,
                SPY34,
                SPmV34c,
                DVL34,
                DVHN34,
                DVH34,
                DVN34,
                V34,
                VL34,
                VH34,
                VN34,
            ]
        )

        interp_all = np.vstack([interp12, interp34])
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

        # Temperature & conductivity
        temp_rows = []
        for seg_id in [1, 2, 3, 4]:
            seg_len = len(tc_segments[seg_id]["STX"])
            seg_col = np.full(seg_len, seg_id)
            rows = np.column_stack(
                [
                    seg_col,
                    tc_segments[seg_id]["STX"],
                    tc_segments[seg_id]["STY"],
                    tc_segments[seg_id]["STd"],
                    tc_segments[seg_id]["STD"],
                    tc_segments[seg_id]["S"],
                    S_corr[seg_id],
                    tc_segments[seg_id]["T"],
                    T_corr[seg_id],
                    tc_segments[seg_id]["SC"],
                    SC_corr[seg_id],
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

        # Correction tables
        pd.DataFrame(
            drift_corrs, columns=["segment_id", "slope_m", "intercept_b", "r2"]
        ).to_csv(PROCESSED_DIR / "Drift_Correction_python.csv", index=False)

        pd.DataFrame(
            temp_corrs, columns=["segment_id", "slope_m", "intercept_b", "r2"]
        ).to_csv(PROCESSED_DIR / "Temperature_Correction_python.csv", index=False)

        pd.DataFrame(
            cond_corrs, columns=["segment_id", "slope_m", "intercept_b", "r2"]
        ).to_csv(PROCESSED_DIR / "Conductivity_Correction_python.csv", index=False)

        pd.DataFrame(
            sc_corrs, columns=["segment_id", "slope_m", "intercept_b", "r2"]
        ).to_csv(PROCESSED_DIR / "Specific_Conductance_Correction_python.csv", index=False)

    return {
        "SPmV12c": SPmV12c,
        "SPmV34c": SPmV34c,
        "DVL12": DVL12,
        "DVL34": DVL34,
        "DVH12": DVH12,
        "DVH34": DVH34,
        "DVN12": DVN12,
        "DVN34": DVN34,
        "V12": V12,
        "V34": V34,
        "VL12": VL12,
        "VL34": VL34,
        "VH12": VH12,
        "VH34": VH34,
        "VN12": VN12,
        "VN34": VN34,
    }


if __name__ == "__main__":
    results = process_code(plot=False, write=True)