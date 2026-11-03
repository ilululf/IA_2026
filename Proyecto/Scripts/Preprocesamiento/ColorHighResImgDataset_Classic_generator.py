#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import io
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import matplotlib.cm as cm
import numpy as np
import pandas as pd
from astropy.io import fits
from PIL import Image
from scipy.ndimage import grey_opening
from tqdm import tqdm

try:
    import pywt
except ImportError:
    pywt = None


# =========================================================
# CONFIG
# =========================================================

RAW_DATASET_ROOT = Path(r"C:\Users\Macri\Desktop\Facultad\Tesis\dataset")

VALID_STATIONS_INPUT_CSV = Path(
    r"C:\Users\Macri\Desktop\Facultad\Tesis\dataset_images_768_validstations\valid_stations_used.csv"
)

OUTPUT_ROOT = Path(
    r"C:\Users\Macri\Desktop\Facultad\Tesis\dataset_images_768_nonai_visual_p1p99"
)

TARGET_FREQ_MIN_MHZ = 45.0
TARGET_FREQ_MAX_MHZ = 870.0
IMG_H = 768
IMG_W = 768

FIT_EXTENSIONS = {".fit", ".fits", ".fit.gz", ".fits.gz"}


# =========================================================
# ARPLS / ROBUST BASELINE
# =========================================================

ARPLS_LAMBDA = 1.0e5
ARPLS_RATIO = 1.0e-3
ARPLS_MAX_ITER = 25
ARPLS_EXP_CLIP = 60.0

ROBUST_EPS = 1.0e-12
ROBUST_Z_CLIP = 50.0


# =========================================================
# HORIZONTAL STRIPE ESTIMATION
# =========================================================

HORIZONTAL_STRUCT_FRACTIONS = (0.03, 0.07, 0.15)

HORIZONTAL_STRUCT_MIN = 31
HORIZONTAL_STRUCT_MAX = 601

# Más suave que la versión anterior
HORIZONTAL_SUBTRACTION_ALPHA = 0.50

# Menor peso mínimo para no levantar/forzar tanto la resta en filas no claras
MIN_STRIPE_SUBTRACTION_WEIGHT = 0.20

ROW_WEIGHT_Z_LOW = 1.0
ROW_WEIGHT_Z_HIGH = 4.0

POST_DESTRIPE_ROW_MEDIAN_SUBTRACTION = True


# =========================================================
# FOURIER DESTRIPING
# =========================================================

USE_FOURIER_DESTRIPING = True

FFT_TIME_HALF_WIDTH = 2
FFT_FREQ_MIN_FRAC = 0.015

# Más suave: 0.70 atenúa poco; 0.35 era más agresivo.
FFT_NOTCH_ATTENUATION = 0.70


# =========================================================
# WAVELET DESTRIPING
# =========================================================

# Para esta prueba lo apagamos, porque puede levantar textura visual.
USE_WAVELET_DESTRIPING = False

WAVELET_NAME = "db2"
WAVELET_LEVEL = 2
WAVELET_HORIZONTAL_DETAIL_ATTENUATION = 0.55


# =========================================================
# QUALITY FILTERS
# =========================================================

ENABLE_NOISE_FILTER = True

MAX_STRIPE_COMPONENT_FRACTION = 0.60
MAX_HORIZONTAL_NOISE_SCORE = 4.00
MAX_BAD_ROW_FRACTION = 0.35
MAX_HOT_BACKGROUND_FRACTION = 0.35
MIN_USABLE_COVERAGE_FRACTION = 0.05

HOT_PIXEL_Z_THRESHOLD = 4.0
BAD_ROW_Z_THRESHOLD = 3.5


# =========================================================
# VISUAL NORMALIZATION
# =========================================================

# Feature tomada del código que se veía más limpio:
# no usar asinh y volver a p1-p99.
USE_ASINH_COMPRESSION = False
ASINH_SCALE_MODE = "p95"
ASINH_FIXED_SCALE = 1.0

NORM_P_LOW = 1.0
NORM_P_HIGH = 99.0


# =========================================================
# OUTPUT FILES
# =========================================================

LABELS_CSV = OUTPUT_ROOT / "labels.csv"
FAILED_CSV = OUTPUT_ROOT / "failed_files.csv"
USED_VALID_STATIONS_CSV = OUTPUT_ROOT / "used_valid_stations_copy.csv"
QUALITY_CSV = OUTPUT_ROOT / "quality_summary.csv"
SKIPPED_NOISY_CSV = OUTPUT_ROOT / "skipped_too_noisy.csv"


# =========================================================
# DATA CLASSES
# =========================================================

@dataclass
class LabelRecord:
    image_path: str
    mask_path: str
    final_label: str
    original_type_folder: str
    station: str
    year: str
    source_fit: str


@dataclass
class FailedRecord:
    source_fit: str
    reason: str


@dataclass
class QualityRecord:
    source_fit: str
    station: str
    final_label: str
    freq_min_mhz: float
    freq_max_mhz: float
    usable_coverage_fraction: float
    stripe_component_fraction: float
    horizontal_noise_score: float
    bad_row_fraction: float
    hot_background_fraction: float
    was_skipped_as_noisy: bool


@dataclass
class SkippedNoisyRecord:
    source_fit: str
    station: str
    final_label: str
    usable_coverage_fraction: float
    stripe_component_fraction: float
    horizontal_noise_score: float
    bad_row_fraction: float
    hot_background_fraction: float
    reason: str


# =========================================================
# TYPE MAPPING
# =========================================================

def normalize_type_folder_name(type_folder_name: str) -> str:
    return str(type_folder_name).strip().upper()


def map_to_final_label(type_folder_name: str) -> Optional[str]:
    folder = normalize_type_folder_name(type_folder_name)

    if folder == "II":
        return "II"
    if folder == "III":
        return "III"
    if folder == "VI":
        return "III"
    if folder == "IV":
        return "IV"

    return None


# =========================================================
# VALID STATIONS
# =========================================================

def load_valid_stations(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"No encuentro el archivo de estaciones válidas: {csv_path}")

    df = pd.read_csv(csv_path)

    if "station" not in df.columns:
        raise ValueError(f"El archivo {csv_path.name} no contiene la columna 'station'")

    return set(df["station"].astype(str).str.strip())


# =========================================================
# FILE DISCOVERY
# =========================================================

def is_fit_file(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(ext) for ext in FIT_EXTENSIONS)


def iter_raw_fit_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and is_fit_file(path):
            yield path


def infer_dataset_metadata(root: Path, fit_path: Path) -> tuple[str, str, str]:
    rel = fit_path.relative_to(root)
    parts = rel.parts

    if len(parts) < 4:
        raise ValueError(f"Estructura inesperada para {fit_path}")

    original_type_folder = parts[0]
    station = parts[1]
    year = parts[2]

    return original_type_folder, station, year


def build_output_name(final_label: str, station: str, year: str, source_fit: Path) -> str:
    stem = source_fit.name

    for ext in [".fit.gz", ".fits.gz", ".fit", ".fits"]:
        if stem.lower().endswith(ext):
            stem = stem[:-len(ext)]
            break

    station_clean = station.replace("/", "_")
    return f"{final_label}_{station_clean}_{year}_{stem}.png"


# =========================================================
# FITS HELPERS
# =========================================================

def open_fits_any(path: Path) -> fits.HDUList:
    lower = path.name.lower()

    if lower.endswith(".gz"):
        raw = gzip.decompress(path.read_bytes())
        return fits.open(io.BytesIO(raw), memmap=False, lazy_load_hdus=False)

    return fits.open(path, memmap=False, lazy_load_hdus=False)


def check_fits_callisto_hdul(hdul) -> bool:
    return len(hdul) == 2


def check_bin_table_hdul(hdul) -> bool:
    try:
        _ = hdul[1].data
        return True
    except Exception:
        return False


def sanitize_freq_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    arr = arr[(arr > 0.0) & (arr < 5000.0)]
    return arr


def first_2d_array_hdu(hdul: fits.HDUList):
    for hdu in hdul:
        data = getattr(hdu, "data", None)

        if data is None:
            continue

        arr = np.asarray(data)

        if arr.ndim == 2 and arr.size > 0:
            return hdu, arr

    raise ValueError("No se encontró una matriz 2D en el FITS")


def try_extract_freq_axis_from_bintable(hdul: fits.HDUList) -> Optional[np.ndarray]:
    for hdu in hdul:
        if not isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
            continue

        if hdu.data is None or hdu.columns is None:
            continue

        names_lower = [str(n).lower() for n in hdu.columns.names]

        if "frequency" in names_lower:
            try:
                vals = sanitize_freq_values(np.asarray(hdu.data["frequency"][0]))
                if vals.size >= 2:
                    return vals
            except Exception:
                pass

        if "freq" in names_lower:
            try:
                vals = sanitize_freq_values(np.asarray(hdu.data["freq"][0]))
                if vals.size >= 2:
                    return vals
            except Exception:
                pass

    return None


def try_extract_freq_axis_from_header(header, target_len: int) -> Optional[np.ndarray]:
    naxis = int(header.get("NAXIS", 0))

    for axis in range(1, naxis + 1):
        ctype = str(header.get(f"CTYPE{axis}", "")).upper()

        if "FREQ" not in ctype:
            continue

        n = header.get(f"NAXIS{axis}")
        crval = header.get(f"CRVAL{axis}")
        cdelt = header.get(f"CDELT{axis}")
        crpix = header.get(f"CRPIX{axis}", 1.0)

        if n is None or crval is None or cdelt is None:
            continue

        try:
            n = int(n)
            if n != target_len:
                continue

            crval = float(crval)
            cdelt = float(cdelt)
            crpix = float(crpix)

            idx = np.arange(1, n + 1, dtype=np.float64)
            vals = crval + (idx - crpix) * cdelt

            if np.nanmax(np.abs(vals)) > 1e5:
                vals = vals / 1e6

            vals = sanitize_freq_values(vals)

            if vals.size == target_len:
                return vals
        except Exception:
            pass

    min_keys = ["FREQMIN", "FRQMIN", "MINFREQ"]
    max_keys = ["FREQMAX", "FRQMAX", "MAXFREQ"]

    for kmin in min_keys:
        for kmax in max_keys:
            if kmin in header and kmax in header:
                try:
                    fmin = float(header[kmin])
                    fmax = float(header[kmax])

                    if abs(fmin) > 1e5 or abs(fmax) > 1e5:
                        fmin /= 1e6
                        fmax /= 1e6

                    vals = np.linspace(fmin, fmax, target_len, dtype=np.float64)
                    vals = sanitize_freq_values(vals)

                    if vals.size == target_len:
                        return vals
                except Exception:
                    pass

    return None


def orient_data_with_freq(hdul: fits.HDUList, data2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(data2d, dtype=np.float64)

    freq_from_bintable = try_extract_freq_axis_from_bintable(hdul)

    if freq_from_bintable is not None:
        if freq_from_bintable.size == data.shape[0]:
            return data, freq_from_bintable

        if freq_from_bintable.size == data.shape[1]:
            return data.T, freq_from_bintable

    for hdu in hdul:
        arr = getattr(hdu, "data", None)

        if arr is None:
            continue

        arr = np.asarray(arr)

        if arr.ndim != 2:
            continue

        if arr.shape != data.shape:
            continue

        freq0 = try_extract_freq_axis_from_header(hdu.header, data.shape[0])

        if freq0 is not None:
            return data, freq0

        freq1 = try_extract_freq_axis_from_header(hdu.header, data.shape[1])

        if freq1 is not None:
            return data.T, freq1

    freq0 = try_extract_freq_axis_from_header(hdul[0].header, data.shape[0])

    if freq0 is not None:
        return data, freq0

    freq1 = try_extract_freq_axis_from_header(hdul[0].header, data.shape[1])

    if freq1 is not None:
        return data.T, freq1

    if data.shape[0] <= data.shape[1]:
        freq_guess = np.linspace(TARGET_FREQ_MIN_MHZ, TARGET_FREQ_MAX_MHZ, data.shape[0], dtype=np.float64)
        return data, freq_guess

    freq_guess = np.linspace(TARGET_FREQ_MIN_MHZ, TARGET_FREQ_MAX_MHZ, data.shape[1], dtype=np.float64)
    return data.T, freq_guess


# =========================================================
# ROBUST MATH
# =========================================================

def robust_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return 0.0

    med = np.median(x)
    mad = np.median(np.abs(x - med))

    return float(mad)


def safe_robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.where(np.isfinite(x), x, 0.0)

    med = np.median(x)
    residual = x - med

    mad = robust_mad(residual)
    den = max(1.4826 * mad, ROBUST_EPS)

    z = residual / den
    z = np.clip(z, -ROBUST_Z_CLIP, ROBUST_Z_CLIP)

    return z


def ensure_odd(n: int) -> int:
    if n % 2 == 0:
        n += 1
    return n


def odd_window_from_fraction(n_cols: int, frac: float) -> int:
    raw = int(round(n_cols * frac))
    raw = max(HORIZONTAL_STRUCT_MIN, raw)
    raw = min(HORIZONTAL_STRUCT_MAX, raw)
    raw = min(raw, n_cols)

    if raw < 3:
        raw = 3

    raw = ensure_odd(raw)

    if raw > n_cols:
        raw = n_cols if n_cols % 2 == 1 else max(3, n_cols - 1)

    return raw


# =========================================================
# BACKGROUND / ARPLS
# =========================================================

def subtract_background_median(data_ft: np.ndarray) -> np.ndarray:
    med_col = np.median(data_ft, axis=1, keepdims=True)
    return data_ft - med_col


def arpls_baseline_1d(
    y: np.ndarray,
    lam: float = ARPLS_LAMBDA,
    ratio: float = ARPLS_RATIO,
    max_iter: int = ARPLS_MAX_ITER,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.size

    if n < 3:
        return y.copy()

    d = np.zeros((n - 2, n), dtype=np.float64)

    for i in range(n - 2):
        d[i, i] = 1.0
        d[i, i + 1] = -2.0
        d[i, i + 2] = 1.0

    h = lam * (d.T @ d)
    w = np.ones(n, dtype=np.float64)

    for _ in range(max_iter):
        wmat = np.diag(w)
        z = np.linalg.solve(wmat + h, w * y)

        dres = y - z
        neg = dres[dres < 0]

        if neg.size == 0:
            break

        mu = float(np.mean(neg))
        sigma = float(np.std(neg))
        sigma = max(sigma, ROBUST_EPS)

        arg = 2.0 * (dres - (2.0 * sigma - mu)) / sigma
        arg = np.clip(arg, -ARPLS_EXP_CLIP, ARPLS_EXP_CLIP)

        wt = 1.0 / (1.0 + np.exp(arg))

        denom = max(float(np.linalg.norm(w)), ROBUST_EPS)
        change = float(np.linalg.norm(w - wt) / denom)

        w = wt

        if change < ratio:
            break

    return z


def arpls_row_contamination_score(data_ft: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_energy = np.median(np.abs(data_ft), axis=1)
    baseline = arpls_baseline_1d(row_energy)
    residual = row_energy - baseline
    row_z = safe_robust_z(residual)

    return row_energy, baseline, row_z


# =========================================================
# HORIZONTAL MORPHOLOGICAL STRIPE COMPONENT
# =========================================================

def estimate_horizontal_stripe_component(
    data_ft: np.ndarray,
) -> tuple[np.ndarray, dict]:
    data = np.asarray(data_ft, dtype=np.float64)
    positive = np.maximum(data, 0.0)

    _, _, row_z = arpls_row_contamination_score(data)

    row_weight = (row_z - ROW_WEIGHT_Z_LOW) / max(ROW_WEIGHT_Z_HIGH - ROW_WEIGHT_Z_LOW, ROBUST_EPS)
    row_weight = np.clip(row_weight, 0.0, 1.0)
    row_weight = MIN_STRIPE_SUBTRACTION_WEIGHT + (1.0 - MIN_STRIPE_SUBTRACTION_WEIGHT) * row_weight

    n_cols = data.shape[1]
    components = []

    for frac in HORIZONTAL_STRUCT_FRACTIONS:
        win = odd_window_from_fraction(n_cols, frac)
        comp = grey_opening(positive, size=(1, win), mode="nearest")
        components.append(comp)

    if not components:
        stripe_component = np.zeros_like(data)
    else:
        stripe_component = np.maximum.reduce(components)

    stripe_component = stripe_component * row_weight[:, None]

    total_energy = float(np.sum(np.abs(data)) + ROBUST_EPS)
    stripe_energy = float(np.sum(np.abs(stripe_component)))
    stripe_fraction = stripe_energy / total_energy

    meta = {
        "stripe_component_fraction": stripe_fraction,
    }

    return stripe_component, meta


def subtract_horizontal_component(data_ft: np.ndarray) -> tuple[np.ndarray, dict]:
    stripe_component, meta = estimate_horizontal_stripe_component(data_ft)
    cleaned = data_ft - HORIZONTAL_SUBTRACTION_ALPHA * stripe_component

    meta["horizontal_subtraction_alpha"] = HORIZONTAL_SUBTRACTION_ALPHA

    return cleaned, meta


# =========================================================
# FOURIER DESTRIPING
# =========================================================

def soft_fourier_horizontal_notch(data_ft: np.ndarray) -> np.ndarray:
    if not USE_FOURIER_DESTRIPING:
        return data_ft

    data = np.asarray(data_ft, dtype=np.float64)

    n_rows, n_cols = data.shape

    if n_rows < 8 or n_cols < 8:
        return data.copy()

    f = np.fft.fftshift(np.fft.fft2(data))

    cy = n_rows // 2
    cx = n_cols // 2

    yy, xx = np.ogrid[:n_rows, :n_cols]

    time_band = np.abs(xx - cx) <= FFT_TIME_HALF_WIDTH
    freq_dist = np.abs(yy - cy)

    freq_min = max(1, int(round(n_rows * FFT_FREQ_MIN_FRAC)))
    freq_band = freq_dist >= freq_min

    notch = time_band & freq_band

    f[notch] *= FFT_NOTCH_ATTENUATION

    out = np.real(np.fft.ifft2(np.fft.ifftshift(f)))

    return out


# =========================================================
# WAVELET DESTRIPING
# =========================================================

def wavelet_horizontal_destripe(data_ft: np.ndarray) -> np.ndarray:
    if not USE_WAVELET_DESTRIPING:
        return data_ft

    if pywt is None:
        raise ImportError("Falta PyWavelets. Instálalo con: pip install PyWavelets")

    data = np.asarray(data_ft, dtype=np.float64)
    orig_shape = data.shape

    max_level = pywt.dwtn_max_level(orig_shape, WAVELET_NAME)
    level = min(WAVELET_LEVEL, max_level)

    if level < 1:
        return data.copy()

    coeffs = pywt.wavedec2(data, wavelet=WAVELET_NAME, level=level, mode="periodization")

    new_coeffs = [coeffs[0]]

    for detail in coeffs[1:]:
        c_h, c_v, c_d = detail
        c_h = c_h * WAVELET_HORIZONTAL_DETAIL_ATTENUATION
        new_coeffs.append((c_h, c_v, c_d))

    rec = pywt.waverec2(new_coeffs, wavelet=WAVELET_NAME, mode="periodization")
    rec = rec[:orig_shape[0], :orig_shape[1]]

    return rec


# =========================================================
# QUALITY METRICS
# =========================================================

def compute_quality_metrics(
    original_bgsub_ft: np.ndarray,
    cleaned_ft: np.ndarray,
    stripe_component_fraction: float,
    usable_coverage_fraction: float,
) -> dict:
    _, _, row_z = arpls_row_contamination_score(cleaned_ft)
    abs_row_z = np.abs(row_z[np.isfinite(row_z)])

    if abs_row_z.size == 0:
        horizontal_noise_score = 0.0
        bad_row_fraction = 0.0
    else:
        horizontal_noise_score = float(np.percentile(abs_row_z, 95))
        bad_row_fraction = float(np.mean(abs_row_z > BAD_ROW_Z_THRESHOLD))

    vals = cleaned_ft.ravel()
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        hot_background_fraction = 1.0
    else:
        pix_z = np.abs(safe_robust_z(vals))
        hot_background_fraction = float(np.mean(pix_z > HOT_PIXEL_Z_THRESHOLD))

    metrics = {
        "usable_coverage_fraction": float(usable_coverage_fraction),
        "stripe_component_fraction": float(stripe_component_fraction),
        "horizontal_noise_score": float(horizontal_noise_score),
        "bad_row_fraction": float(bad_row_fraction),
        "hot_background_fraction": float(hot_background_fraction),
    }

    return metrics


def should_skip_by_quality(metrics: dict) -> tuple[bool, str]:
    if not ENABLE_NOISE_FILTER:
        return False, ""

    if metrics["usable_coverage_fraction"] < MIN_USABLE_COVERAGE_FRACTION:
        return True, "low_usable_coverage"

    if metrics["stripe_component_fraction"] > MAX_STRIPE_COMPONENT_FRACTION:
        return True, "high_stripe_component_fraction"

    if metrics["horizontal_noise_score"] > MAX_HORIZONTAL_NOISE_SCORE:
        return True, "high_horizontal_noise_score"

    if metrics["bad_row_fraction"] > MAX_BAD_ROW_FRACTION:
        return True, "high_bad_row_fraction"

    if metrics["hot_background_fraction"] > MAX_HOT_BACKGROUND_FRACTION:
        return True, "high_hot_background_fraction"

    return False, ""


# =========================================================
# FREQUENCY / TIME RESAMPLING
# =========================================================

def sort_and_deduplicate_freq(data_ft: np.ndarray, freqs_mhz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(freqs_mhz)
    freqs = freqs_mhz[order]
    data = data_ft[order, :]

    uniq_freqs, inverse = np.unique(freqs, return_inverse=True)

    if uniq_freqs.size == freqs.size:
        return data, freqs

    collapsed = np.zeros((uniq_freqs.size, data.shape[1]), dtype=np.float64)
    counts = np.zeros(uniq_freqs.size, dtype=np.int32)

    for i, grp in enumerate(inverse):
        collapsed[grp] += data[i]
        counts[grp] += 1

    collapsed /= counts[:, None]

    return collapsed, uniq_freqs


def interpolate_frequency_axis(
    data_ft: np.ndarray,
    src_freqs: np.ndarray,
    dst_freqs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    src_data, src_freqs = sort_and_deduplicate_freq(data_ft, src_freqs)

    out = np.zeros((dst_freqs.size, src_data.shape[1]), dtype=np.float64)

    fmin = np.min(src_freqs)
    fmax = np.max(src_freqs)

    valid_mask_rows = (dst_freqs >= fmin) & (dst_freqs <= fmax)
    valid_idx = np.where(valid_mask_rows)[0]

    if valid_idx.size == 0:
        return out, valid_mask_rows.astype(np.uint8)

    for t in range(src_data.shape[1]):
        col = src_data[:, t]
        finite = np.isfinite(col) & np.isfinite(src_freqs)

        if np.count_nonzero(finite) < 2:
            continue

        xf = src_freqs[finite]
        yf = col[finite]

        out[valid_idx, t] = np.interp(dst_freqs[valid_idx], xf, yf)

    return out, valid_mask_rows.astype(np.uint8)


def resize_time_axis(data_ft: np.ndarray, target_w: int) -> np.ndarray:
    src_w = data_ft.shape[1]

    if src_w == target_w:
        return data_ft.copy()

    x_src = np.linspace(0.0, 1.0, src_w, dtype=np.float64)
    x_dst = np.linspace(0.0, 1.0, target_w, dtype=np.float64)

    out = np.zeros((data_ft.shape[0], target_w), dtype=np.float64)

    for i in range(data_ft.shape[0]):
        row = data_ft[i, :]
        finite = np.isfinite(row)

        if np.count_nonzero(finite) < 2:
            continue

        out[i, :] = np.interp(x_dst, x_src[finite], row[finite])

    return out


def build_coverage_mask(freq_row_mask: np.ndarray, target_w: int) -> np.ndarray:
    return np.repeat(freq_row_mask[:, None], target_w, axis=1).astype(np.uint8)


# =========================================================
# VISUAL NORMALIZATION
# =========================================================

def apply_asinh_compression(image_ft: np.ndarray, coverage_mask_ft: np.ndarray) -> np.ndarray:
    img = np.asarray(image_ft, dtype=np.float64).copy()
    valid = coverage_mask_ft > 0

    vals = img[valid]
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return np.zeros_like(img, dtype=np.float64)

    if ASINH_SCALE_MODE == "p95":
        scale = float(np.percentile(np.abs(vals), 95))
    else:
        scale = float(ASINH_FIXED_SCALE)

    scale = max(scale, ROBUST_EPS)

    out = np.arcsinh(img / scale)
    out[~valid] = 0.0

    return out


def robust_percentile_normalize(image_ft: np.ndarray, coverage_mask_ft: np.ndarray) -> np.ndarray:
    valid = coverage_mask_ft > 0
    vals = image_ft[valid]
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return np.zeros_like(image_ft, dtype=np.float64)

    p_low = np.percentile(vals, NORM_P_LOW)
    p_high = np.percentile(vals, NORM_P_HIGH)

    if not np.isfinite(p_low) or not np.isfinite(p_high) or p_high <= p_low:
        mn = np.min(vals)
        mx = np.max(vals)

        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
            return np.zeros_like(image_ft, dtype=np.float64)

        p_low, p_high = mn, mx

    out = (image_ft - p_low) / (p_high - p_low)
    out = np.clip(out, 0.0, 1.0)
    out[~valid] = 0.0

    return out


def apply_jet_colormap(img01: np.ndarray, coverage_mask_ft: np.ndarray) -> np.ndarray:
    rgba = cm.jet(img01)
    rgb = (rgba[:, :, :3] * 255.0).astype(np.uint8)

    invalid = coverage_mask_ft == 0
    rgb[invalid] = 0

    return rgb


# =========================================================
# FIT PROCESSING
# =========================================================

def process_fit_to_image_and_mask(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with open_fits_any(path) as hdul:
        if not check_fits_callisto_hdul(hdul):
            raise ValueError("El FITS no tiene exactamente 2 HDUs")

        if not check_bin_table_hdul(hdul):
            raise ValueError("La BinTable no es accesible o está corrupta")

        _, raw2d = first_2d_array_hdu(hdul)
        data_ft, freqs_mhz = orient_data_with_freq(hdul, raw2d)

    data_ft = np.asarray(data_ft, dtype=np.float64)
    data_ft[~np.isfinite(data_ft)] = 0.0

    freq_min = float(np.min(freqs_mhz))
    freq_max = float(np.max(freqs_mhz))

    # 1) fondo por mediana por frecuencia
    bgsub_ft = subtract_background_median(data_ft)

    # 2) morfología direccional horizontal suave
    cleaned_ft, stripe_meta = subtract_horizontal_component(bgsub_ft)

    # 3) Fourier suave
    cleaned_ft = soft_fourier_horizontal_notch(cleaned_ft)

    # 4) Wavelet apagado por defecto en esta versión
    cleaned_ft = wavelet_horizontal_destripe(cleaned_ft)

    # 5) recentrado final por fila
    if POST_DESTRIPE_ROW_MEDIAN_SUBTRACTION:
        cleaned_ft = subtract_background_median(cleaned_ft)

    # 6) grilla fija
    dst_freqs = np.linspace(TARGET_FREQ_MIN_MHZ, TARGET_FREQ_MAX_MHZ, IMG_H, dtype=np.float64)

    interp_ft, freq_row_mask = interpolate_frequency_axis(cleaned_ft, freqs_mhz, dst_freqs)

    usable_coverage_fraction = float(np.mean(freq_row_mask > 0))

    # 7) remuestreo temporal
    interp_ft = resize_time_axis(interp_ft, IMG_W)

    # 8) máscara de cobertura
    coverage_mask_ft = build_coverage_mask(freq_row_mask, IMG_W)

    # 9) métricas de calidad
    quality = compute_quality_metrics(
        original_bgsub_ft=bgsub_ft,
        cleaned_ft=cleaned_ft,
        stripe_component_fraction=stripe_meta["stripe_component_fraction"],
        usable_coverage_fraction=usable_coverage_fraction,
    )

    # 10) compresión visual apagada por defecto
    if USE_ASINH_COMPRESSION:
        interp_ft = apply_asinh_compression(interp_ft, coverage_mask_ft)

    # 11) normalización p1-p99
    norm_ft = robust_percentile_normalize(interp_ft, coverage_mask_ft)

    # 12) colormap final
    rgb_u8 = apply_jet_colormap(norm_ft, coverage_mask_ft)
    mask_u8 = (coverage_mask_ft * 255).astype(np.uint8)

    # altas arriba, bajas abajo
    rgb_u8 = np.flipud(rgb_u8)
    mask_u8 = np.flipud(mask_u8)

    meta = {
        "freq_min_mhz": freq_min,
        "freq_max_mhz": freq_max,
        "usable_coverage_fraction": quality["usable_coverage_fraction"],
        "stripe_component_fraction": quality["stripe_component_fraction"],
        "horizontal_noise_score": quality["horizontal_noise_score"],
        "bad_row_fraction": quality["bad_row_fraction"],
        "hot_background_fraction": quality["hot_background_fraction"],
    }

    return rgb_u8, mask_u8, meta


# =========================================================
# SAVE OUTPUTS
# =========================================================

def save_rgb_png(array_u8: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array_u8, mode="RGB").save(path)


def save_gray_png(array_u8: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array_u8, mode="L").save(path)


# =========================================================
# MAIN
# =========================================================

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "images").mkdir(exist_ok=True)
    (OUTPUT_ROOT / "masks").mkdir(exist_ok=True)

    label_rows: list[dict] = []
    failed_rows: list[dict] = []
    quality_rows: list[dict] = []
    skipped_rows: list[dict] = []

    valid_stations = load_valid_stations(VALID_STATIONS_INPUT_CSV)
    pd.read_csv(VALID_STATIONS_INPUT_CSV).to_csv(USED_VALID_STATIONS_CSV, index=False)

    print(f"[INFO] Estaciones válidas cargadas desde referencia local: {len(valid_stations)}")

    fit_files = list(iter_raw_fit_files(RAW_DATASET_ROOT))

    print(f"[INFO] FITS encontrados en total: {len(fit_files)}")

    valid_fit_files: list[Path] = []

    # -----------------------------------------------------
    # 1) Filtrar por tipo válido y estación válida
    # -----------------------------------------------------
    for fit_path in tqdm(fit_files, desc="Filtrando FITS válidos", unit="fit"):
        try:
            original_type_folder, station, year = infer_dataset_metadata(RAW_DATASET_ROOT, fit_path)
            final_label = map_to_final_label(original_type_folder)

            if final_label is None:
                continue

            if station not in valid_stations:
                continue

            valid_fit_files.append(fit_path)

        except Exception as exc:
            failed_rows.append(asdict(FailedRecord(
                source_fit=str(fit_path),
                reason=f"[filter-pass] {exc}",
            )))

    print(f"[INFO] FITS que serán procesados: {len(valid_fit_files)}")

    # -----------------------------------------------------
    # 2) Generar imágenes
    # -----------------------------------------------------
    for fit_path in tqdm(valid_fit_files, desc="Procesando FITS", unit="fit"):
        try:
            original_type_folder, station, year = infer_dataset_metadata(RAW_DATASET_ROOT, fit_path)
            final_label = map_to_final_label(original_type_folder)

            if final_label is None:
                continue

            image_rgb_u8, mask_u8, meta = process_fit_to_image_and_mask(fit_path)

            was_skipped, skip_reason = should_skip_by_quality(meta)

            quality_rows.append(asdict(QualityRecord(
                source_fit=str(fit_path),
                station=station,
                final_label=final_label,
                freq_min_mhz=meta["freq_min_mhz"],
                freq_max_mhz=meta["freq_max_mhz"],
                usable_coverage_fraction=meta["usable_coverage_fraction"],
                stripe_component_fraction=meta["stripe_component_fraction"],
                horizontal_noise_score=meta["horizontal_noise_score"],
                bad_row_fraction=meta["bad_row_fraction"],
                hot_background_fraction=meta["hot_background_fraction"],
                was_skipped_as_noisy=was_skipped,
            )))

            if was_skipped:
                skipped_rows.append(asdict(SkippedNoisyRecord(
                    source_fit=str(fit_path),
                    station=station,
                    final_label=final_label,
                    usable_coverage_fraction=meta["usable_coverage_fraction"],
                    stripe_component_fraction=meta["stripe_component_fraction"],
                    horizontal_noise_score=meta["horizontal_noise_score"],
                    bad_row_fraction=meta["bad_row_fraction"],
                    hot_background_fraction=meta["hot_background_fraction"],
                    reason=skip_reason,
                )))
                continue

            out_name = build_output_name(final_label, station, year, fit_path)

            img_path = OUTPUT_ROOT / "images" / final_label / out_name
            mask_path = OUTPUT_ROOT / "masks" / final_label / out_name

            save_rgb_png(image_rgb_u8, img_path)
            save_gray_png(mask_u8, mask_path)

            label_rows.append(asdict(LabelRecord(
                image_path=str(img_path),
                mask_path=str(mask_path),
                final_label=final_label,
                original_type_folder=original_type_folder,
                station=station,
                year=year,
                source_fit=str(fit_path),
            )))

        except Exception as exc:
            failed_rows.append(asdict(FailedRecord(
                source_fit=str(fit_path),
                reason=f"[image-pass] {exc}",
            )))

    # -----------------------------------------------------
    # 3) Guardar CSVs
    # -----------------------------------------------------
    if label_rows:
        with LABELS_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(label_rows[0].keys()))
            writer.writeheader()
            writer.writerows(label_rows)

    if failed_rows:
        with FAILED_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(failed_rows[0].keys()))
            writer.writeheader()
            writer.writerows(failed_rows)

    if quality_rows:
        with QUALITY_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(quality_rows[0].keys()))
            writer.writeheader()
            writer.writerows(quality_rows)

    if skipped_rows:
        with SKIPPED_NOISY_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(skipped_rows[0].keys()))
            writer.writeheader()
            writer.writerows(skipped_rows)

    print("[INFO] Proceso terminado.")
    print(f"[INFO] Imágenes generadas: {len(label_rows)}")
    print(f"[INFO] Imágenes descartadas por ruido: {len(skipped_rows)}")
    print(f"[INFO] Archivos con error: {len(failed_rows)}")
    print(f"[INFO] labels.csv: {LABELS_CSV}")
    print(f"[INFO] failed_files.csv: {FAILED_CSV}")
    print(f"[INFO] used_valid_stations_copy.csv: {USED_VALID_STATIONS_CSV}")
    print(f"[INFO] quality_summary.csv: {QUALITY_CSV}")
    print(f"[INFO] skipped_too_noisy.csv: {SKIPPED_NOISY_CSV}")


if __name__ == "__main__":
    main()