"""Self-potential and hydrogeophysical processing routines.

This module loads configuration from a TOML file and performs a series of
processing steps originally translated from MATLAB code. It supports:

* Gaussian despiking and exponential smoothing.
* Boxcar (moving-average) filtering.
* Least-squares polynomial fitting.
* FFT-based amplitude spectra.
* Simpson-rule integration of electric field to potential.
* Segment-wise drift correction and partitioning of SP signals.
* Temperature and conductivity drift correction.

The main entry point is `process_code()`, which reads raw CSV inputs, applies
all processing steps, and optionally writes several processed CSV products.

Configuration
-------------
The module expects a TOML file with the following structure:

[paths]
input_dir = "test_data"          # Directory containing raw input CSVs
processed_dir = "test_data/processed"  # Output directory for processed CSVs
figures_dir = "test_data/figures"      # Output directory for figures

[files]
sp_data = "Self_Potential_Data_Rio_Grande.csv"            # Gradient SP data
drift_data = "Self_Potential_Electrode_Drift_Data_Rio_Grande.csv"  # Drift
temp_cond_data = "Temperature_Conductivity_Data_Rio_Grande.csv"    # Temp/cond
hfem_resistivity = "HFEM_resistivity.csv"                # HFEM resistivity (reserved)

[processing]
dipole_length_m = 0.5588         # Gradient dipole length in meters
gaussian_sigma = 30              # Gaussian despiking sigma (samples)
boxcar_m = 5                     # Half-width of boxcar window (samples)
boxcar_iterations = 1            # Number of boxcar passes
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
    """Load TOML configuration.

    Args:
        config_path: Path to the TOML configuration file. Defaults to
            "config.toml" in the current working directory.

    Returns:
        A dictionary containing the parsed configuration, typically with
        "paths", "files", and "processing" sections.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        tomllib.TOMLDecodeError: If the configuration file cannot be parsed.
    """
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
HFEM_FILE = INPUT_DIR / CFG["files"]["hfem_resistivity"]  # reserved for future use

# Processing parameters
DIP_LO = float(CFG["processing"]["dipole_length_m"])
GAUSS_SIGMA = int(CFG["processing"]["gaussian_sigma"])
BOXCAR_M = int(CFG["processing"]["boxcar_m"])
BOXCAR_ITS = int(CFG["processing"]["boxcar_iterations"])


def gauss_filter(x: ArrayLike, sigma: int) -> FloatArray:
    """Apply a Gaussian despiking filter.

    This function implements the behavior of the original MATLAB GaussFilter.m
    routine. It convolves a 1D signal with a Gaussian kernel over a window
    spanning ±3 * sigma samples, while leaving values at the ends unchanged.

    Args:
        x: One-dimensional input sequence.
        sigma: Half-width of the Gaussian kernel in samples.

    Returns:
        A NumPy array of the same shape as `x` containing the filtered values.
    """
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
    """Apply parallel exponential smoothing (causal + anti-causal).

    Implements the EXPsmooth.m behavior: for each input sequence, performs an
    anti-causal pass, a causal pass, and combines them to produce a smoothed
    output. Multiple arrays can be smoothed in parallel using different or
    shared smoothing factors.

    Args:
        a: Scalar or array-like of smoothing factors in [0, 1). If scalar,
            the same factor is applied to all inputs. If array-like, the
            length must match the number of arrays.
        pad: Controls the first output value. If 0, sets the first output
            value to 0.0. Otherwise, uses the first input value.
        *arrays: One or more one-dimensional input sequences to smooth.

    Returns:
        A list of smoothed NumPy arrays, one per input sequence.

    Raises:
        AssertionError: If `a` is array-like and its length does not match
            the number of input arrays.
    """
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
    """Apply an iterated moving-average (boxcar) filter.

    This function implements Boxcarv1.m, using a window of length 2*m+1 and
    applying the filter iteratively `its` times. The interior points are
    updated using a fast recurrence; endpoints are fixed at the original
    values after each iteration.

    Args:
        x: One-dimensional input sequence.
        m: Half-width of the moving-average window. The full window length is
            2*m + 1.
        its: Number of times to apply the boxcar filter.

    Returns:
        A NumPy array containing the filtered sequence.

    Raises:
        ValueError: If `m` is greater than or equal to 0.5 * len(x).
    """
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
    """Fit a polynomial of given order by least squares.

    This function implements LSPoly.m. It constructs a Vandermonde matrix
    for the input x-values, solves the normal equations for the polynomial
    coefficients, and returns both the coefficients and the coefficient of
    determination r^2.

    Args:
        order: Polynomial order.
        x: One-dimensional x-values.
        y: One-dimensional y-values.

    Returns:
        A tuple containing the polynomial coefficients, the coefficient of
        determination r^2, and the fitted values evaluated at `x`.

    Raises:
        np.linalg.LinAlgError: If the normal equation system is singular.
    """
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
    """Compute FFT-based amplitude spectra for one or more signals.

    Implements Pspec.m: for each input array, zero-pads to the next power of
    two, computes the FFT, and returns the single-sided amplitude spectrum
    and corresponding frequency vector.

    Args:
        T: Scalar or array-like of sample periods in seconds. If scalar, the
            same sample period is used for all inputs.
        *arrays: One or more one-dimensional input sequences.

    Returns:
        A tuple of two lists:
        - Amplitude spectra.
        - Corresponding frequency arrays.

    Raises:
        AssertionError: If `T` is array-like and its length does not match
            the number of input arrays.
    """
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
    """Integrate a sequence using a Simpson-like difference equation.

    Implements Simpson.m: given a sequence of electric field values and a
    constant dipole length, it computes the cumulative integral using a
    three-point stencil and returns both the total area and the cumulative
    integral array.

    Args:
        x: One-dimensional sequence of values.
        dx: Scalar dipole length used to scale the final integral.

    Returns:
        A tuple containing the final integral value and the cumulative
        integral array.
    """
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
    """Compute great-circle distances using the haversine formula.

    Args:
        lat_ref_deg: Reference latitude in degrees.
        lon_ref_deg: Reference longitude in degrees.
        lats_deg: Target latitudes in degrees.
        lons_deg: Target longitudes in degrees.
        earth_radius_m: Earth radius in meters.

    Returns:
        Distances in kilometers from the reference point to each target point.
    """
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
    """Delete specified indices from an array.

    Args:
        arr: Input array to filter.
        indices: Integer indices to remove.

    Returns:
        A NumPy array with the requested indices removed.
    """
    array = np.asarray(arr)
    index_array = np.asarray(indices, dtype=int)
    mask = np.ones(len(array), dtype=bool)
    mask[index_array] = False
    return array[mask]


def lin(m: float, x: ArrayLike, b: float) -> FloatArray:
    """Evaluate a linear function.

    Args:
        m: Slope.
        x: Input values.
        b: Intercept.

    Returns:
        The values of m * x + b.
    """
    return m * np.asarray(x, dtype=float) + b


def cond_to_sc(S: ArrayLike, T: ArrayLike) -> FloatArray:
    """Convert conductivity to specific conductance at 25 °C.

    Args:
        S: Conductivity values.
        T: Temperature values in degrees Celsius.

    Returns:
        Specific conductance values adjusted to 25 °C.
    """
    conductivity = np.asarray(S, dtype=float)
    temperature = np.asarray(T, dtype=float)
    return conductivity / (1.0 + 0.002 * (temperature - 25.0))


def process_code(
    plot: bool = False,
    write: bool = False,
    config_path: str | Path = "config.toml",
) -> ProcessResults:
    """Run the main self-potential and hydrogeophysical processing workflow.

    This function is the Python translation of Process_code.m. It loads the
    configured SP, drift, and temperature/conductivity datasets; applies
    drift correction, filtering, and integration; and optionally writes
    processed outputs to disk.

    Args:
        plot: Whether to produce plots or figures. Currently unused.
        write: Whether to write processed CSV outputs.
        config_path: Path to a TOML configuration file. Currently reserved
            for future runtime override behavior.

    Returns:
        A dictionary containing processed SP and potential arrays.

    Raises:
        FileNotFoundError: If an expected input file is missing.
        pd.errors.ParserError: If an input CSV cannot be parsed.
        ValueError: If filter parameters are invalid.
    """
    del plot
    del config_path

    r_earth = 6_367_000.0
    dL = DIP_LO

    sp_df = pd.read_csv(SP_DATA)

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

    spd: dict[int, FloatArray] = {}
    spd_survey: dict[int, FloatArray] = {}
    for seg_id in [1, 2, 3, 4]:
        spx_i = segments[seg_id]["SPX"]
        spy_i = segments[seg_id]["SPY"]
        spd_i = haversine_dist_km(spy_i[0], spx_i[0], spy_i, spx_i, r_earth)
        spd[seg_id] = spd_i

        spx1 = segments[1]["SPX"]
        spy1 = segments[1]["SPY"]
        spd_survey_i = haversine_dist_km(spy1[0], spx1[0], spy_i, spx_i, r_earth)
        spd_survey[seg_id] = spd_survey_i

    for start, end in [(10075, 10095), (9550, 9575), (7290, 7301), (4140, 4240)]:
        idx = np.arange(start - 1, end)
        for key in ["SPX", "SPY", "SPmV"]:
            segments[2][key] = delete_indices(segments[2][key], idx)
        spd[2] = delete_indices(spd[2], idx)
        spd_survey[2] = delete_indices(spd_survey[2], idx)

    cut3 = np.arange(8138 - 1, len(segments[3]["SPmV"]))
    for key in ["SPX", "SPY", "SPmV"]:
        segments[3][key] = delete_indices(segments[3][key], cut3)
    spd[3] = delete_indices(spd[3], cut3)
    spd_survey[3] = delete_indices(spd_survey[3], cut3)

    cut4 = np.arange(10750 - 1, 11000)
    for key in ["SPX", "SPY", "SPmV"]:
        segments[4][key] = delete_indices(segments[4][key], cut4)
    spd[4] = delete_indices(spd[4], cut4)
    spd_survey[4] = delete_indices(spd_survey[4], cut4)

    drift_df = pd.read_csv(SP_DRIFT)
    drift_id = drift_df.iloc[:, 0].values
    drift_n = drift_df.iloc[:, 2].values
    drift_dv = drift_df.iloc[:, 4].values

    drift_segments: dict[int, dict[str, FloatArray]] = {}
    for uid in np.unique(drift_id):
        mask = drift_id == uid
        drift_segments[int(uid)] = {
            "N": np.asarray(drift_n[mask], dtype=float),
            "dV": np.asarray(drift_dv[mask], dtype=float),
        }

    drift_corrs_list: list[list[float]] = []
    for seg_id in [1, 2, 3, 4]:
        dv = drift_segments[seg_id]["dV"]
        t = np.arange(1, len(dv) + 1, dtype=float)
        coeffs, r2, _ = ls_poly(order=1, x=t, y=dv)
        drift_corrs_list.append([float(seg_id), float(coeffs[1]), float(coeffs[0]), r2])
    drift_corrs = np.array(drift_corrs_list, dtype=float)

    spmv_corr: dict[int, FloatArray] = {}
    for seg_id in [1, 2, 3, 4]:
        sp = segments[seg_id]["SPmV"]
        x = np.arange(1, len(sp) + 1, dtype=float)
        m = drift_corrs[seg_id - 1, 1]
        b = drift_corrs[seg_id - 1, 2]
        trend = lin(float(m), x, float(b))
        spmv_corr[seg_id] = sp - trend

    spmv_corr[1] = spmv_corr[1] - spmv_corr[1][0]
    shift12 = spmv_corr[2][0] - spmv_corr[1][-1]
    spmv_corr[2] = spmv_corr[2] - shift12
    spmv_corr[3] = spmv_corr[3] - spmv_corr[3][0]

    spmv12c = np.concatenate([spmv_corr[1], spmv_corr[2]])
    spmv34c = np.concatenate([spmv_corr[3], spmv_corr[4]])

    _amps, _freqs = pspec(
        [1, 1, 1, 1],
        drift_segments[1]["dV"],
        drift_segments[2]["dV"],
        drift_segments[3]["dV"],
        drift_segments[4]["dV"],
    )

    exp_12 = exp_smooth(0.9, 1, spmv12c)[0]
    gauss_12 = gauss_filter(spmv12c, sigma=GAUSS_SIGMA)
    gauss_12[:120] = exp_12[:120]
    gauss_12[-120:] = exp_12[-120:]
    dvl12 = gauss_12.copy()
    dvhn12 = spmv12c - dvl12

    exp_34 = exp_smooth(0.9, 1, spmv34c)[0]
    gauss_34 = gauss_filter(spmv34c, sigma=GAUSS_SIGMA)
    gauss_34[:120] = exp_34[:120]
    gauss_34[-120:] = exp_34[-120:]
    dvl34 = gauss_34.copy()
    dvhn34 = spmv34c - dvl34

    dvh12 = boxcar(dvhn12, m=BOXCAR_M, its=BOXCAR_ITS)
    dvh34 = boxcar(dvhn34, m=BOXCAR_M, its=BOXCAR_ITS)
    dvn12 = dvhn12 - dvh12
    dvn34 = dvhn34 - dvh34

    e12 = -spmv12c * dL
    _, v12 = simpson(e12, dL)
    v12 = -v12

    e34 = -spmv34c * dL
    _, v34 = simpson(e34, dL)
    v34 = -v34

    el12 = -dvl12 * dL
    _, vl12 = simpson(el12, dL)
    vl12 = -vl12

    el34 = -dvl34 * dL
    _, vl34 = simpson(el34, dL)
    vl34 = -vl34

    eh12 = -dvh12 * dL
    _, vh12 = simpson(eh12, dL)
    vh12 = -vh12

    eh34 = -dvh34 * dL
    _, vh34 = simpson(eh34, dL)
    vh34 = -vh34

    en12 = -dvn12 * dL
    _, vn12 = simpson(en12, dL)
    vn12 = -vn12

    en34 = -dvn34 * dL
    _, vn34 = simpson(en34, dL)
    vn34 = -vn34

    tc_df = pd.read_csv(TC_DATA)
    stid = tc_df.iloc[:, 0].values
    stx = tc_df.iloc[:, 5].values
    sty = tc_df.iloc[:, 6].values
    temp = tc_df.iloc[:, 7].values
    cond = tc_df.iloc[:, 8].values

    tc_segments: dict[int, dict[str, FloatArray]] = {}
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

    temp_corrs_list: list[list[float]] = []
    for seg_id in [1, 2, 3, 4]:
        temp_i = tc_segments[seg_id]["T"]
        t = np.arange(1, len(temp_i) + 1, dtype=float)
        coeffs, r2, _ = ls_poly(order=1, x=t, y=temp_i)
        temp_corrs_list.append([float(seg_id), float(coeffs[1]), float(coeffs[0]), r2])
    temp_corrs = np.array(temp_corrs_list, dtype=float)

    t_corr: dict[int, FloatArray] = {}
    for seg_id in [1, 2, 3, 4]:
        temp_i = tc_segments[seg_id]["T"]
        x = np.arange(1, len(temp_i) + 1, dtype=float)
        m = temp_corrs[seg_id - 1, 1]
        b = temp_corrs[seg_id - 1, 2]
        t_corr[seg_id] = temp_i - lin(float(m), x, float(b))

    cond_corrs_list: list[list[float]] = []
    for seg_id in [1, 2, 3, 4]:
        cond_i = tc_segments[seg_id]["S"]
        t = np.arange(1, len(cond_i) + 1, dtype=float)
        coeffs, r2, _ = ls_poly(order=1, x=t, y=cond_i)
        cond_corrs_list.append([float(seg_id), float(coeffs[1]), float(coeffs[0]), r2])
    cond_corrs = np.array(cond_corrs_list, dtype=float)

    s_corr: dict[int, FloatArray] = {}
    for seg_id in [1, 2, 3, 4]:
        cond_i = tc_segments[seg_id]["S"]
        x = np.arange(1, len(cond_i) + 1, dtype=float)
        m = cond_corrs[seg_id - 1, 1]
        b = cond_corrs[seg_id - 1, 2]
        s_corr[seg_id] = cond_i - lin(float(m), x, float(b))

    sc_corrs_list: list[list[float]] = []
    for seg_id in [1, 2, 3, 4]:
        sc_i = tc_segments[seg_id]["SC"]
        t = np.arange(1, len(sc_i) + 1, dtype=float)
        coeffs, r2, _ = ls_poly(order=1, x=t, y=sc_i)
        sc_corrs_list.append([float(seg_id), float(coeffs[1]), float(coeffs[0]), r2])
    sc_corrs = np.array(sc_corrs_list, dtype=float)

    sc_corr: dict[int, FloatArray] = {}
    for seg_id in [1, 2, 3, 4]:
        sc_i = tc_segments[seg_id]["SC"]
        x = np.arange(1, len(sc_i) + 1, dtype=float)
        m = sc_corrs[seg_id - 1, 1]
        b = sc_corrs[seg_id - 1, 2]
        sc_corr[seg_id] = sc_i - lin(float(m), x, float(b))

    if write:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        grad_rows: list[FloatArray] = []
        for seg_id in [1, 2, 3, 4]:
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

        spx12 = np.concatenate([segments[1]["SPX"], segments[2]["SPX"]])
        spy12 = np.concatenate([segments[1]["SPY"], segments[2]["SPY"]])
        seg_ids_12 = np.concatenate(
            [np.full(len(segments[1]["SPX"]), 1), np.full(len(segments[2]["SPX"]), 2)]
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

        spx34 = np.concatenate([segments[3]["SPX"], segments[4]["SPX"]])
        spy34 = np.concatenate([segments[3]["SPY"], segments[4]["SPY"]])
        seg_ids_34 = np.concatenate(
            [np.full(len(segments[3]["SPX"]), 3), np.full(len(segments[4]["SPX"]), 4)]
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

        temp_rows: list[FloatArray] = []
        for seg_id in [1, 2, 3, 4]:
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