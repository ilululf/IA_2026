#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import matplotlib.cm as cm
import numpy as np
import pandas as pd
from astropy.io import fits
from PIL import Image
from tqdm import tqdm


# ============================================================
# CONFIGURACIÓN DEFAULT
# ============================================================

DEFAULT_RAW_DATASET_ROOT = Path(r"C:\Users\Macri\Desktop\Facultad\Tesis\dataset")

DEFAULT_OUTPUT_ROOT = Path(
    r"C:\Users\Macri\Desktop\Facultad\Tesis\dataset_images_656_top6_meanfreq_50_870"
)

TOP6_STATIONS = {
    "HUMAIN",
    "SSRT",
    "SOUTHAFRICA-SANSA",
    "KASI",
    "INDIA-OOTY",
    "MONGOLIA-UB",
}

VALID_TOP_LEVEL_TYPES = {"II", "III", "IV", "VI"}

TYPE_TO_CLASS = {
    "II": "II",
    "III": "III",
    "VI": "III",
    "IV": "IV",
}

FIT_EXTENSIONS = (".fit", ".fits", ".fit.gz", ".fits.gz")

DEFAULT_TARGET_FREQ_MIN_MHZ = 50.0
DEFAULT_TARGET_FREQ_MAX_MHZ = 870.0

DEFAULT_IMAGE_SIZE = 656

DEFAULT_NORM_P_LOW = 1.0
DEFAULT_NORM_P_HIGH = 99.0

EPS = 1.0e-12


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class LabelRecord:
    image_path: str
    mask_path: str
    final_label: str
    original_type_folder: str
    station: str
    station_folder: str
    year: str
    source_fit: str
    image_size: int
    target_freq_min_mhz: float
    target_freq_max_mhz: float
    prefilter_method: str
    normalization_method: str


@dataclass
class QualityRecord:
    source_fit: str
    fits_file_name: str
    status: str
    error: str
    original_type_folder: str
    final_label: str
    station: str
    station_folder: str
    year: str
    n_freq_original: int
    n_time_original: int
    n_freq_after_crop: int
    n_time_after_crop: int
    finite_fraction_original: float
    finite_fraction_after_crop: float
    freq_min_original_mhz: float
    freq_max_original_mhz: float
    freq_min_after_crop_mhz: float
    freq_max_after_crop_mhz: float
    target_freq_min_mhz: float
    target_freq_max_mhz: float
    coverage_fraction: float
    raw_dynamic_range_p05_p95: float
    residual_noise_sigma: float
    residual_noise_relative: float
    common_mode_sigma_after_prefilter: float
    common_mode_strength_after_prefilter: float
    common_mode_energy_ratio_after_prefilter: float
    robust_peak_snr_after_prefilter: float
    norm_p_low_value: float
    norm_p_high_value: float
    norm_min_value: float
    norm_max_value: float
    freq_source_used: str
    output_image_path: str
    output_mask_path: str


@dataclass
class FailedRecord:
    source_fit: str
    stage: str
    reason: str


# ============================================================
# UTILIDADES GENERALES
# ============================================================

def canonical_name(name: str) -> str:
    return str(name).strip().upper()


def normalize_type_folder_name(type_folder_name: str) -> str:
    return str(type_folder_name).strip().upper()


def map_to_final_label(type_folder_name: str) -> Optional[str]:
    return TYPE_TO_CLASS.get(normalize_type_folder_name(type_folder_name))


def station_to_key(station_folder: str, allowed_stations: set[str]) -> Optional[str]:
    """
    Permite:
    - HUMAIN -> HUMAIN
    - HUMAIN_59 -> HUMAIN
    - SOUTHAFRICA-SANSA -> SOUTHAFRICA-SANSA
    """
    station = canonical_name(station_folder)

    if station in allowed_stations:
        return station

    if "_" in station:
        base = station.split("_")[0]
        if base in allowed_stations:
            return base

    return None


def safe_float(value: Any) -> float:
    try:
        value = float(value)
        if math.isfinite(value):
            return value
        return float("nan")
    except Exception:
        return float("nan")


def robust_mad_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return float("nan")

    med = np.median(x)
    mad = np.median(np.abs(x - med))

    return float(1.4826 * mad)


def safe_percentile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return float("nan")

    return float(np.percentile(x, q))


def compute_overlap_fraction(
    freq_min: float,
    freq_max: float,
    target_freq_min: float,
    target_freq_max: float,
) -> float:
    if not np.isfinite(freq_min) or not np.isfinite(freq_max):
        return float("nan")

    lo = min(freq_min, freq_max)
    hi = max(freq_min, freq_max)

    overlap_min = max(lo, target_freq_min)
    overlap_max = min(hi, target_freq_max)

    overlap = max(0.0, overlap_max - overlap_min)
    target_bw = max(target_freq_max - target_freq_min, EPS)

    return float(overlap / target_bw)


def short_hash_path(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()
    return digest[:10]


# ============================================================
# BÚSQUEDA DE FITS
# ============================================================

def is_fits_file(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(ext) for ext in FIT_EXTENSIONS)


def iter_raw_fit_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and is_fits_file(path):
            yield path


def infer_dataset_metadata(root: Path, fit_path: Path) -> tuple[str, str, str]:
    """
    Estructura esperada:
    dataset/TIPO/ESTACION/AÑO/archivo.fit.gz
    """
    rel = fit_path.relative_to(root)
    parts = rel.parts

    if len(parts) < 4:
        raise ValueError(f"Estructura inesperada: {fit_path}")

    original_type_folder = str(parts[0]).strip()
    station_folder = str(parts[1]).strip()
    year = str(parts[2]).strip()

    return original_type_folder, station_folder, year


def build_output_name(
    final_label: str,
    station_key: str,
    station_folder: str,
    year: str,
    source_fit: Path,
) -> str:
    stem = source_fit.name

    for ext in [".fit.gz", ".fits.gz", ".fit", ".fits"]:
        if stem.lower().endswith(ext):
            stem = stem[:-len(ext)]
            break

    station_clean = station_key.replace("/", "_").replace("\\", "_").replace(" ", "_")
    folder_clean = station_folder.replace("/", "_").replace("\\", "_").replace(" ", "_")
    h = short_hash_path(source_fit)

    return f"{final_label}_{station_clean}_{folder_clean}_{year}_{stem}_{h}.png"


# ============================================================
# LECTURA FITS
# ============================================================

def open_fits_any(path: Path) -> fits.HDUList:
    lower = path.name.lower()

    if lower.endswith(".gz"):
        raw = gzip.decompress(path.read_bytes())
        return fits.open(
            io.BytesIO(raw),
            mode="readonly",
            memmap=False,
            lazy_load_hdus=False,
        )

    return fits.open(
        path,
        mode="readonly",
        memmap=False,
        lazy_load_hdus=False,
    )


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

        if arr.ndim == 2 and arr.size > 0 and np.issubdtype(arr.dtype, np.number):
            return hdu, arr

    raise ValueError("No se encontró una matriz 2D numérica en el FITS.")


def try_extract_freq_axis_from_bintable(hdul: fits.HDUList) -> Optional[np.ndarray]:
    for hdu in hdul:
        if not isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
            continue

        if hdu.data is None or hdu.columns is None:
            continue

        colnames = [str(n).lower() for n in hdu.columns.names]

        if "frequency" in colnames:
            try:
                vals = sanitize_freq_values(np.asarray(hdu.data["frequency"][0]))
                if vals.size >= 2:
                    return vals
            except Exception:
                pass

        if "freq" in colnames:
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


def orient_data_with_freq(
    hdul: fits.HDUList,
    data2d: np.ndarray,
    target_freq_min: float,
    target_freq_max: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Devuelve:
    - matriz [frecuencia, tiempo]
    - eje de frecuencia en MHz
    - fuente/orientación usada
    """
    data = np.asarray(data2d, dtype=np.float64)

    freq_from_bintable = try_extract_freq_axis_from_bintable(hdul)

    if freq_from_bintable is not None:
        if freq_from_bintable.size == data.shape[0]:
            return data, freq_from_bintable, "bintable_frequency_rows"

        if freq_from_bintable.size == data.shape[1]:
            return data.T, freq_from_bintable, "bintable_frequency_cols_transposed"

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
            return data, freq0, "header_rows"

        freq1 = try_extract_freq_axis_from_header(hdu.header, data.shape[1])

        if freq1 is not None:
            return data.T, freq1, "header_cols_transposed"

    freq0 = try_extract_freq_axis_from_header(hdul[0].header, data.shape[0])

    if freq0 is not None:
        return data, freq0, "primary_header_rows"

    freq1 = try_extract_freq_axis_from_header(hdul[0].header, data.shape[1])

    if freq1 is not None:
        return data.T, freq1, "primary_header_cols_transposed"

    # Último recurso: heurística.
    # Se conserva para no perder FITS raros, pero queda registrado en freq_source_used.
    if data.shape[0] <= data.shape[1]:
        freq_guess = np.linspace(
            target_freq_min,
            target_freq_max,
            data.shape[0],
            dtype=np.float64,
        )
        return data, freq_guess, "heuristic_rows"

    freq_guess = np.linspace(
        target_freq_min,
        target_freq_max,
        data.shape[1],
        dtype=np.float64,
    )
    return data.T, freq_guess, "heuristic_cols_transposed"


# ============================================================
# PREPROCESAMIENTO
# ============================================================

def crop_frequency_range(
    data_ft: np.ndarray,
    freqs_mhz: np.ndarray,
    freq_min_mhz: float,
    freq_max_mhz: float,
) -> tuple[np.ndarray, np.ndarray]:
    freqs = np.asarray(freqs_mhz, dtype=np.float64).ravel()
    data = np.asarray(data_ft, dtype=np.float64)

    if data.shape[0] != freqs.size:
        raise ValueError(
            f"No coincide eje de frecuencia con filas de datos: "
            f"data.shape={data.shape}, freqs={freqs.size}"
        )

    mask = (
        np.isfinite(freqs)
        & (freqs >= freq_min_mhz)
        & (freqs <= freq_max_mhz)
    )

    if np.count_nonzero(mask) < 2:
        raise ValueError(
            f"Menos de 2 canales dentro del rango {freq_min_mhz}-{freq_max_mhz} MHz."
        )

    return data[mask, :], freqs[mask]


def replace_nonfinite_with_median(data: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.asarray(data, dtype=np.float64)
    finite = np.isfinite(x)
    finite_fraction = float(np.mean(finite))

    if np.count_nonzero(finite) == 0:
        raise ValueError("La matriz no contiene valores finitos.")

    median_value = float(np.median(x[finite]))
    x = np.where(finite, x, median_value)

    return x, finite_fraction


def subtract_temporal_mean_per_frequency(data_ft: np.ndarray) -> np.ndarray:
    """
    Resta de media temporal por frecuencia.

    X'(f,t) = X(f,t) - mean_t(X(f,t))

    Se usa para eliminar el fondo estable de cada canal y bandas horizontales
    constantes.
    """
    x = np.asarray(data_ft, dtype=np.float64)
    mean_per_freq = np.mean(x, axis=1, keepdims=True)
    return x - mean_per_freq


# ============================================================
# MÉTRICAS DE CALIDAD
# ============================================================

def compute_quality_metrics_after_prefilter(
    raw_cropped_ft: np.ndarray,
    filtered_ft: np.ndarray,
    freqs_cropped_mhz: np.ndarray,
    target_freq_min: float,
    target_freq_max: float,
) -> dict[str, float]:
    raw = np.asarray(raw_cropped_ft, dtype=np.float64)
    residual = np.asarray(filtered_ft, dtype=np.float64)
    freqs = np.asarray(freqs_cropped_mhz, dtype=np.float64)

    freq_min_after_crop = float(np.nanmin(freqs))
    freq_max_after_crop = float(np.nanmax(freqs))

    coverage_fraction = compute_overlap_fraction(
        freq_min=freq_min_after_crop,
        freq_max=freq_max_after_crop,
        target_freq_min=target_freq_min,
        target_freq_max=target_freq_max,
    )

    raw_p05 = safe_percentile(raw, 5.0)
    raw_p95 = safe_percentile(raw, 95.0)
    raw_dynamic_range = raw_p95 - raw_p05

    row_sigmas = np.array(
        [robust_mad_sigma(residual[i, :]) for i in range(residual.shape[0])],
        dtype=np.float64,
    )

    row_sigmas = row_sigmas[np.isfinite(row_sigmas)]

    if row_sigmas.size == 0:
        residual_noise_sigma = float("nan")
    else:
        residual_noise_sigma = float(np.median(row_sigmas))

    residual_noise_relative = (
        float(residual_noise_sigma / (raw_dynamic_range + EPS))
        if np.isfinite(residual_noise_sigma) and np.isfinite(raw_dynamic_range)
        else float("nan")
    )

    common_temporal = np.median(residual, axis=0)
    common_mode_sigma = robust_mad_sigma(common_temporal)

    common_mode_strength = (
        float(common_mode_sigma / (residual_noise_sigma + EPS))
        if np.isfinite(common_mode_sigma) and np.isfinite(residual_noise_sigma)
        else float("nan")
    )

    residual_energy = float(np.mean(residual ** 2))
    common_energy = float(np.mean(common_temporal ** 2))

    common_mode_energy_ratio = (
        float(common_energy / (residual_energy + EPS))
        if np.isfinite(common_energy) and np.isfinite(residual_energy)
        else float("nan")
    )

    r_p50 = safe_percentile(residual, 50.0)
    r_p995 = safe_percentile(residual, 99.5)

    robust_peak_snr = (
        float((r_p995 - r_p50) / (residual_noise_sigma + EPS))
        if np.isfinite(r_p995) and np.isfinite(r_p50) and np.isfinite(residual_noise_sigma)
        else float("nan")
    )

    return {
        "freq_min_after_crop_mhz": freq_min_after_crop,
        "freq_max_after_crop_mhz": freq_max_after_crop,
        "coverage_fraction": float(coverage_fraction),
        "raw_dynamic_range_p05_p95": float(raw_dynamic_range),
        "residual_noise_sigma": float(residual_noise_sigma),
        "residual_noise_relative": float(residual_noise_relative),
        "common_mode_sigma_after_prefilter": float(common_mode_sigma),
        "common_mode_strength_after_prefilter": float(common_mode_strength),
        "common_mode_energy_ratio_after_prefilter": float(common_mode_energy_ratio),
        "robust_peak_snr_after_prefilter": float(robust_peak_snr),
    }


# ============================================================
# REMUESTREO EN FRECUENCIA Y TIEMPO
# ============================================================

def sort_and_deduplicate_freq(
    data_ft: np.ndarray,
    freqs_mhz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    freqs = np.asarray(freqs_mhz, dtype=np.float64).ravel()
    data = np.asarray(data_ft, dtype=np.float64)

    order = np.argsort(freqs)
    freqs = freqs[order]
    data = data[order, :]

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

    fmin = float(np.min(src_freqs))
    fmax = float(np.max(src_freqs))

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

    if src_w < 2:
        raise ValueError("No se puede remuestrear tiempo con menos de 2 columnas.")

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


# ============================================================
# NORMALIZACIÓN VISUAL Y COLORMAP
# ============================================================

def robust_percentile_normalize(
    image_ft: np.ndarray,
    coverage_mask_ft: np.ndarray,
    p_low: float,
    p_high: float,
) -> tuple[np.ndarray, dict[str, float]]:
    valid = coverage_mask_ft > 0
    vals = image_ft[valid]
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        meta = {
            "norm_p_low_value": float("nan"),
            "norm_p_high_value": float("nan"),
            "norm_min_value": float("nan"),
            "norm_max_value": float("nan"),
        }
        return np.zeros_like(image_ft, dtype=np.float64), meta

    p_low_value = float(np.percentile(vals, p_low))
    p_high_value = float(np.percentile(vals, p_high))
    min_value = float(np.min(vals))
    max_value = float(np.max(vals))

    lo = p_low_value
    hi = p_high_value

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = min_value
        hi = max_value

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            meta = {
                "norm_p_low_value": p_low_value,
                "norm_p_high_value": p_high_value,
                "norm_min_value": min_value,
                "norm_max_value": max_value,
            }
            return np.zeros_like(image_ft, dtype=np.float64), meta

    out = (image_ft - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    out[~valid] = 0.0

    meta = {
        "norm_p_low_value": p_low_value,
        "norm_p_high_value": p_high_value,
        "norm_min_value": min_value,
        "norm_max_value": max_value,
    }

    return out, meta


def apply_jet_colormap(img01: np.ndarray, coverage_mask_ft: np.ndarray) -> np.ndarray:
    rgba = cm.jet(img01)
    rgb = (rgba[:, :, :3] * 255.0).astype(np.uint8)

    invalid = coverage_mask_ft == 0
    rgb[invalid] = 0

    return rgb


# ============================================================
# PROCESAMIENTO DE UN FITS
# ============================================================

def process_fit_to_image_mask_and_quality(
    fit_path: Path,
    target_freq_min: float,
    target_freq_max: float,
    image_size: int,
    norm_p_low: float,
    norm_p_high: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with open_fits_any(fit_path) as hdul:
        _, raw2d = first_2d_array_hdu(hdul)

        data_ft, freqs_mhz, freq_source = orient_data_with_freq(
            hdul=hdul,
            data2d=raw2d,
            target_freq_min=target_freq_min,
            target_freq_max=target_freq_max,
        )

    data_ft = np.asarray(data_ft, dtype=np.float64)
    freqs_mhz = np.asarray(freqs_mhz, dtype=np.float64).ravel()

    n_freq_original = int(data_ft.shape[0])
    n_time_original = int(data_ft.shape[1])

    finite_fraction_original = float(np.mean(np.isfinite(data_ft)))

    freq_min_original = float(np.nanmin(freqs_mhz))
    freq_max_original = float(np.nanmax(freqs_mhz))

    # 1) Recorte inferior y superior.
    cropped_ft, cropped_freqs = crop_frequency_range(
        data_ft=data_ft,
        freqs_mhz=freqs_mhz,
        freq_min_mhz=target_freq_min,
        freq_max_mhz=target_freq_max,
    )

    n_freq_after_crop = int(cropped_ft.shape[0])
    n_time_after_crop = int(cropped_ft.shape[1])

    # 2) Sanitización mínima.
    cropped_ft, finite_fraction_after_crop = replace_nonfinite_with_median(cropped_ft)

    # 3) Prefiltrado base: resta de media temporal por frecuencia.
    filtered_ft = subtract_temporal_mean_per_frequency(cropped_ft)

    # 4) Métricas en matriz ya prefiltrada.
    quality = compute_quality_metrics_after_prefilter(
        raw_cropped_ft=cropped_ft,
        filtered_ft=filtered_ft,
        freqs_cropped_mhz=cropped_freqs,
        target_freq_min=target_freq_min,
        target_freq_max=target_freq_max,
    )

    # 5) Grilla común en frecuencia.
    dst_freqs = np.linspace(
        target_freq_min,
        target_freq_max,
        image_size,
        dtype=np.float64,
    )

    interp_ft, freq_row_mask = interpolate_frequency_axis(
        data_ft=filtered_ft,
        src_freqs=cropped_freqs,
        dst_freqs=dst_freqs,
    )

    # 6) Remuestreo temporal.
    interp_ft = resize_time_axis(interp_ft, image_size)

    # 7) Máscara de cobertura.
    coverage_mask_ft = build_coverage_mask(freq_row_mask, image_size)

    # 8) Normalización visual p1-p99.
    norm_ft, norm_meta = robust_percentile_normalize(
        image_ft=interp_ft,
        coverage_mask_ft=coverage_mask_ft,
        p_low=norm_p_low,
        p_high=norm_p_high,
    )

    # 9) Colormap.
    rgb_u8 = apply_jet_colormap(norm_ft, coverage_mask_ft)
    mask_u8 = (coverage_mask_ft * 255).astype(np.uint8)

    # Frecuencias altas arriba, bajas abajo.
    rgb_u8 = np.flipud(rgb_u8)
    mask_u8 = np.flipud(mask_u8)

    meta: dict[str, Any] = {
        "n_freq_original": n_freq_original,
        "n_time_original": n_time_original,
        "n_freq_after_crop": n_freq_after_crop,
        "n_time_after_crop": n_time_after_crop,
        "finite_fraction_original": finite_fraction_original,
        "finite_fraction_after_crop": finite_fraction_after_crop,
        "freq_min_original_mhz": freq_min_original,
        "freq_max_original_mhz": freq_max_original,
        "freq_source_used": freq_source,
    }

    meta.update(quality)
    meta.update(norm_meta)

    return rgb_u8, mask_u8, meta


# ============================================================
# GUARDADO
# ============================================================

def save_rgb_png(array_u8: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array_u8, mode="RGB").save(path)


def save_gray_png(array_u8: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array_u8, mode="L").save(path)


def write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_used_stations_csv(path: Path, stations: set[str]) -> None:
    rows = [
        {
            "station": station,
            "included": True,
            "reason": "top6_selected_station",
        }
        for station in sorted(stations)
    ]

    write_dict_csv(path, rows)


def build_station_class_summary(labels_df: pd.DataFrame) -> pd.DataFrame:
    if labels_df.empty:
        return pd.DataFrame()

    summary = (
        labels_df
        .pivot_table(
            index="station",
            columns="final_label",
            values="image_path",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )

    for cls in ["II", "III", "IV"]:
        if cls not in summary.columns:
            summary[cls] = 0

    summary = summary.rename(
        columns={
            "II": "n_II",
            "III": "n_III",
            "IV": "n_IV",
        }
    )

    summary["n_total"] = summary[["n_II", "n_III", "n_IV"]].sum(axis=1)

    return summary.sort_values("n_total", ascending=False).reset_index(drop=True)


def build_station_quality_summary(
    quality_df: pd.DataFrame,
    failed_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Genera resumen de calidad por estación.

    Nota:
    El conteo de fallos se toma desde quality_df, no desde failed_df,
    porque failed_df puede no tener columna 'station' en errores ocurridos
    durante la etapa inicial de filtrado.
    """
    if quality_df.empty or "status" not in quality_df.columns:
        return pd.DataFrame()

    ok_df = quality_df[quality_df["status"].eq("ok")].copy()

    if ok_df.empty:
        return pd.DataFrame()

    grouped = (
        ok_df
        .groupby("station")
        .agg(
            n_images_generated=("output_image_path", "count"),
            coverage_median=("coverage_fraction", "median"),
            coverage_p25=("coverage_fraction", lambda x: np.nanpercentile(x, 25)),
            coverage_p75=("coverage_fraction", lambda x: np.nanpercentile(x, 75)),
            residual_noise_relative_median=("residual_noise_relative", "median"),
            residual_noise_relative_p75=("residual_noise_relative", lambda x: np.nanpercentile(x, 75)),
            common_mode_strength_median=("common_mode_strength_after_prefilter", "median"),
            common_mode_strength_p75=("common_mode_strength_after_prefilter", lambda x: np.nanpercentile(x, 75)),
            robust_peak_snr_median=("robust_peak_snr_after_prefilter", "median"),
            robust_peak_snr_p25=("robust_peak_snr_after_prefilter", lambda x: np.nanpercentile(x, 25)),
            finite_fraction_after_crop_median=("finite_fraction_after_crop", "median"),
            freq_min_after_crop_median=("freq_min_after_crop_mhz", "median"),
            freq_max_after_crop_median=("freq_max_after_crop_mhz", "median"),
        )
        .reset_index()
    )

    failed_quality_df = quality_df[~quality_df["status"].eq("ok")].copy()

    if failed_quality_df.empty or "station" not in failed_quality_df.columns:
        grouped["failed_count"] = 0
    else:
        failed_quality_df = failed_quality_df[
            failed_quality_df["station"].astype(str).str.strip().ne("")
        ].copy()

        if failed_quality_df.empty:
            grouped["failed_count"] = 0
        else:
            failed_counts = (
                failed_quality_df
                .groupby("station", dropna=False)
                .size()
                .reset_index(name="failed_count")
            )

            grouped = grouped.merge(failed_counts, on="station", how="left")
            grouped["failed_count"] = grouped["failed_count"].fillna(0).astype(int)

    return grouped.sort_values(
        "n_images_generated",
        ascending=False,
    ).reset_index(drop=True)


def build_dataset_summary(
    labels_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    failed_df: pd.DataFrame,
    output_root: Path,
    raw_root: Path,
    target_freq_min: float,
    target_freq_max: float,
    image_size: int,
    norm_p_low: float,
    norm_p_high: float,
) -> pd.DataFrame:
    total_images = int(len(labels_df))
    total_failed = int(len(failed_df))

    counts = {
        "II": 0,
        "III": 0,
        "IV": 0,
    }

    if not labels_df.empty:
        vc = labels_df["final_label"].value_counts().to_dict()
        for cls in counts:
            counts[cls] = int(vc.get(cls, 0))

    row = {
        "dataset_name": output_root.name,
        "raw_dataset_root": str(raw_root),
        "output_root": str(output_root),
        "stations": ";".join(sorted(TOP6_STATIONS)),
        "image_size": int(image_size),
        "target_freq_min_mhz": float(target_freq_min),
        "target_freq_max_mhz": float(target_freq_max),
        "prefilter_method": "subtract_temporal_mean_per_frequency",
        "normalization_method": f"percentile_p{norm_p_low:g}_p{norm_p_high:g}_per_image",
        "total_images": total_images,
        "total_II": counts["II"],
        "total_III": counts["III"],
        "total_IV": counts["IV"],
        "total_failed": total_failed,
        "quality_rows": int(len(quality_df)),
    }

    return pd.DataFrame([row])


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Genera dataset preliminar de imágenes SRB II/III/IV usando top 6 estaciones, "
            "recorte 50-870 MHz y resta de media temporal por frecuencia."
        )
    )

    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_DATASET_ROOT,
        help="Raíz del dataset FITS: dataset/TIPO/ESTACION/AÑO/*.fit.gz",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Carpeta de salida del dataset generado.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Tamaño final cuadrado de imagen y máscara. Default: 656.",
    )

    parser.add_argument(
        "--freq-min",
        type=float,
        default=DEFAULT_TARGET_FREQ_MIN_MHZ,
        help="Frecuencia mínima efectiva. Default: 50 MHz.",
    )

    parser.add_argument(
        "--freq-max",
        type=float,
        default=DEFAULT_TARGET_FREQ_MAX_MHZ,
        help="Frecuencia máxima efectiva. Default: 870 MHz.",
    )

    parser.add_argument(
        "--p-low",
        type=float,
        default=DEFAULT_NORM_P_LOW,
        help="Percentil bajo para normalización visual. Default: 1.",
    )

    parser.add_argument(
        "--p-high",
        type=float,
        default=DEFAULT_NORM_P_HIGH,
        help="Percentil alto para normalización visual. Default: 99.",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Procesa solo N FITS para prueba rápida.",
    )

    args = parser.parse_args()

    raw_root: Path = args.raw_root
    output_root: Path = args.out
    image_size: int = int(args.image_size)
    target_freq_min: float = float(args.freq_min)
    target_freq_max: float = float(args.freq_max)
    norm_p_low: float = float(args.p_low)
    norm_p_high: float = float(args.p_high)

    if image_size < 32:
        raise ValueError("image-size demasiado pequeño.")

    if target_freq_max <= target_freq_min:
        raise ValueError("freq-max debe ser mayor que freq-min.")

    if not raw_root.exists():
        raise FileNotFoundError(f"No existe raw-root: {raw_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "images").mkdir(exist_ok=True)
    (output_root / "masks").mkdir(exist_ok=True)

    labels_csv = output_root / "labels.csv"
    quality_csv = output_root / "quality_by_image.csv"
    station_class_summary_csv = output_root / "station_class_summary.csv"
    station_quality_summary_csv = output_root / "station_quality_summary.csv"
    dataset_summary_csv = output_root / "dataset_summary.csv"
    used_stations_csv = output_root / "used_stations.csv"
    failed_csv = output_root / "failed_files.csv"

    allowed_stations = set(TOP6_STATIONS)

    print("[INFO] Generador de dataset preliminar top 6")
    print(f"[INFO] Dataset FITS: {raw_root}")
    print(f"[INFO] Salida: {output_root}")
    print(f"[INFO] Estaciones: {', '.join(sorted(allowed_stations))}")
    print(f"[INFO] Clases: II, III, IV con VI -> III")
    print(f"[INFO] Frecuencia efectiva: {target_freq_min:g}-{target_freq_max:g} MHz")
    print(f"[INFO] Tamaño de imagen: {image_size}x{image_size}")
    print(f"[INFO] Prefiltro: resta de media temporal por frecuencia")
    print(f"[INFO] Normalización visual: p{norm_p_low:g}-p{norm_p_high:g}")
    print()

    write_used_stations_csv(used_stations_csv, allowed_stations)

    all_fits = list(iter_raw_fit_files(raw_root))
    print(f"[INFO] FITS encontrados en total: {len(all_fits)}")

    candidate_fits: list[Path] = []
    failed_rows: list[dict[str, Any]] = []

    for fit_path in tqdm(all_fits, desc="Filtrando candidatos", unit="fit"):
        try:
            original_type_folder, station_folder, year = infer_dataset_metadata(raw_root, fit_path)

            original_type_norm = normalize_type_folder_name(original_type_folder)

            if original_type_norm not in VALID_TOP_LEVEL_TYPES:
                continue

            final_label = map_to_final_label(original_type_norm)

            if final_label is None:
                continue

            station_key = station_to_key(station_folder, allowed_stations)

            if station_key is None:
                continue

            candidate_fits.append(fit_path)

        except Exception as exc:
            failed_rows.append(asdict(FailedRecord(
                source_fit=str(fit_path),
                stage="filter-pass",
                reason=str(exc),
            )))

    candidate_fits = sorted(candidate_fits)

    if args.max_files is not None:
        candidate_fits = candidate_fits[: args.max_files]

    print(f"[INFO] FITS candidatos top 6: {len(candidate_fits)}")
    print()

    label_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []

    for fit_path in tqdm(candidate_fits, desc="Generando imágenes", unit="fit"):
        original_type_folder = ""
        station_folder = ""
        station_key = ""
        year = ""
        final_label = ""

        try:
            original_type_folder, station_folder, year = infer_dataset_metadata(raw_root, fit_path)
            original_type_norm = normalize_type_folder_name(original_type_folder)
            final_label = map_to_final_label(original_type_norm)

            if final_label is None:
                raise ValueError(f"Tipo no reconocido: {original_type_folder}")

            station_key = station_to_key(station_folder, allowed_stations)

            if station_key is None:
                raise ValueError(f"Estación no incluida en top 6: {station_folder}")

            image_rgb_u8, mask_u8, meta = process_fit_to_image_mask_and_quality(
                fit_path=fit_path,
                target_freq_min=target_freq_min,
                target_freq_max=target_freq_max,
                image_size=image_size,
                norm_p_low=norm_p_low,
                norm_p_high=norm_p_high,
            )

            out_name = build_output_name(
                final_label=final_label,
                station_key=station_key,
                station_folder=station_folder,
                year=year,
                source_fit=fit_path,
            )

            img_path = output_root / "images" / final_label / out_name
            mask_path = output_root / "masks" / final_label / out_name

            save_rgb_png(image_rgb_u8, img_path)
            save_gray_png(mask_u8, mask_path)

            label_rows.append(asdict(LabelRecord(
                image_path=str(img_path),
                mask_path=str(mask_path),
                final_label=final_label,
                original_type_folder=original_type_norm,
                station=station_key,
                station_folder=station_folder,
                year=year,
                source_fit=str(fit_path),
                image_size=image_size,
                target_freq_min_mhz=target_freq_min,
                target_freq_max_mhz=target_freq_max,
                prefilter_method="subtract_temporal_mean_per_frequency",
                normalization_method=f"percentile_p{norm_p_low:g}_p{norm_p_high:g}_per_image",
            )))

            quality_rows.append(asdict(QualityRecord(
                source_fit=str(fit_path),
                fits_file_name=fit_path.name,
                status="ok",
                error="",
                original_type_folder=original_type_norm,
                final_label=final_label,
                station=station_key,
                station_folder=station_folder,
                year=year,
                n_freq_original=int(meta["n_freq_original"]),
                n_time_original=int(meta["n_time_original"]),
                n_freq_after_crop=int(meta["n_freq_after_crop"]),
                n_time_after_crop=int(meta["n_time_after_crop"]),
                finite_fraction_original=safe_float(meta["finite_fraction_original"]),
                finite_fraction_after_crop=safe_float(meta["finite_fraction_after_crop"]),
                freq_min_original_mhz=safe_float(meta["freq_min_original_mhz"]),
                freq_max_original_mhz=safe_float(meta["freq_max_original_mhz"]),
                freq_min_after_crop_mhz=safe_float(meta["freq_min_after_crop_mhz"]),
                freq_max_after_crop_mhz=safe_float(meta["freq_max_after_crop_mhz"]),
                target_freq_min_mhz=target_freq_min,
                target_freq_max_mhz=target_freq_max,
                coverage_fraction=safe_float(meta["coverage_fraction"]),
                raw_dynamic_range_p05_p95=safe_float(meta["raw_dynamic_range_p05_p95"]),
                residual_noise_sigma=safe_float(meta["residual_noise_sigma"]),
                residual_noise_relative=safe_float(meta["residual_noise_relative"]),
                common_mode_sigma_after_prefilter=safe_float(meta["common_mode_sigma_after_prefilter"]),
                common_mode_strength_after_prefilter=safe_float(meta["common_mode_strength_after_prefilter"]),
                common_mode_energy_ratio_after_prefilter=safe_float(meta["common_mode_energy_ratio_after_prefilter"]),
                robust_peak_snr_after_prefilter=safe_float(meta["robust_peak_snr_after_prefilter"]),
                norm_p_low_value=safe_float(meta["norm_p_low_value"]),
                norm_p_high_value=safe_float(meta["norm_p_high_value"]),
                norm_min_value=safe_float(meta["norm_min_value"]),
                norm_max_value=safe_float(meta["norm_max_value"]),
                freq_source_used=str(meta["freq_source_used"]),
                output_image_path=str(img_path),
                output_mask_path=str(mask_path),
            )))

        except Exception as exc:
            failed_rows.append(asdict(FailedRecord(
                source_fit=str(fit_path),
                stage="image-pass",
                reason=str(exc),
            )))

            quality_rows.append(asdict(QualityRecord(
                source_fit=str(fit_path),
                fits_file_name=fit_path.name,
                status="failed",
                error=str(exc),
                original_type_folder=normalize_type_folder_name(original_type_folder) if original_type_folder else "",
                final_label=final_label if final_label else "",
                station=station_key if station_key else "",
                station_folder=station_folder if station_folder else "",
                year=year if year else "",
                n_freq_original=0,
                n_time_original=0,
                n_freq_after_crop=0,
                n_time_after_crop=0,
                finite_fraction_original=float("nan"),
                finite_fraction_after_crop=float("nan"),
                freq_min_original_mhz=float("nan"),
                freq_max_original_mhz=float("nan"),
                freq_min_after_crop_mhz=float("nan"),
                freq_max_after_crop_mhz=float("nan"),
                target_freq_min_mhz=target_freq_min,
                target_freq_max_mhz=target_freq_max,
                coverage_fraction=float("nan"),
                raw_dynamic_range_p05_p95=float("nan"),
                residual_noise_sigma=float("nan"),
                residual_noise_relative=float("nan"),
                common_mode_sigma_after_prefilter=float("nan"),
                common_mode_strength_after_prefilter=float("nan"),
                common_mode_energy_ratio_after_prefilter=float("nan"),
                robust_peak_snr_after_prefilter=float("nan"),
                norm_p_low_value=float("nan"),
                norm_p_high_value=float("nan"),
                norm_min_value=float("nan"),
                norm_max_value=float("nan"),
                freq_source_used="",
                output_image_path="",
                output_mask_path="",
            )))

    write_dict_csv(labels_csv, label_rows)
    write_dict_csv(quality_csv, quality_rows)
    write_dict_csv(failed_csv, failed_rows)

    labels_df = pd.DataFrame(label_rows)
    quality_df = pd.DataFrame(quality_rows)
    failed_df = pd.DataFrame(failed_rows)

    station_class_summary = build_station_class_summary(labels_df)
    station_class_summary.to_csv(station_class_summary_csv, index=False, encoding="utf-8")

    station_quality_summary = build_station_quality_summary(quality_df, failed_df)
    station_quality_summary.to_csv(station_quality_summary_csv, index=False, encoding="utf-8")

    dataset_summary = build_dataset_summary(
        labels_df=labels_df,
        quality_df=quality_df,
        failed_df=failed_df,
        output_root=output_root,
        raw_root=raw_root,
        target_freq_min=target_freq_min,
        target_freq_max=target_freq_max,
        image_size=image_size,
        norm_p_low=norm_p_low,
        norm_p_high=norm_p_high,
    )
    dataset_summary.to_csv(dataset_summary_csv, index=False, encoding="utf-8")

    print()
    print("[INFO] Proceso terminado.")
    print(f"[INFO] Imágenes generadas: {len(label_rows)}")
    print(f"[INFO] Archivos con error: {len(failed_rows)}")

    if not labels_df.empty:
        print()
        print("[INFO] Conteo por clase:")
        print(labels_df["final_label"].value_counts().sort_index().to_string())

        print()
        print("[INFO] Conteo por estación:")
        print(labels_df["station"].value_counts().sort_index().to_string())

    print()
    print(f"[INFO] labels.csv: {labels_csv}")
    print(f"[INFO] quality_by_image.csv: {quality_csv}")
    print(f"[INFO] station_class_summary.csv: {station_class_summary_csv}")
    print(f"[INFO] station_quality_summary.csv: {station_quality_summary_csv}")
    print(f"[INFO] dataset_summary.csv: {dataset_summary_csv}")
    print(f"[INFO] used_stations.csv: {used_stations_csv}")
    print(f"[INFO] failed_files.csv: {failed_csv}")


if __name__ == "__main__":
    main()