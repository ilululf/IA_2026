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
from scipy.interpolate import PchipInterpolator
from tqdm import tqdm


# ============================================================
# CONFIGURACIÓN DEFAULT
# ============================================================

DEFAULT_RAW_DATASET_ROOT = Path(
    "/home/igonzalez/ecallisto/no_srb_fits"
)

DEFAULT_OUTPUT_ROOT = Path(
    "/home/igonzalez/ecallisto/dataset_images_656_top6_pchip_gapfill_meanfreq_positive_50_870"
)

TOP6_STATIONS = {
    "HUMAIN",
    "SSRT",
    "SOUTHAFRICA-SANSA",
    "KASI",
    "INDIA-OOTY",
    "MONGOLIA-UB",
}

VALID_TOP_LEVEL_TYPES = {"II", "III", "IV", "VI", "NO_SRB"}

TYPE_TO_CLASS = {
    "II": "II",
    "III": "III",
    "VI": "III",
    "IV": "IV",
    "NO_SRB": "NO_SRB",
}

FIT_EXTENSIONS = (".fit", ".fits", ".fit.gz", ".fits.gz")

DEFAULT_TARGET_FREQ_MIN_MHZ = 50.0
DEFAULT_TARGET_FREQ_MAX_MHZ = 870.0

DEFAULT_IMAGE_SIZE = 656

# Regrillado conservativo:
# limita el ancho efectivo de un canal para no rellenar huecos enormes
# como si fueran cobertura real.
DEFAULT_MAX_CHANNEL_WIDTH_FACTOR = 3.0

# Normalización visual positiva tipo PyCallisto:
# valores <= 0 quedan como fondo.
DEFAULT_POSITIVE_P_HIGH = 99.7
DEFAULT_VISUAL_GAMMA = 1.0

# Clipping local del PCHIP:
# se limita el valor interpolado al rango local de canales reales vecinos.
DEFAULT_PCHIP_CLIP_CONTEXT_ROWS = 4

EPS = 1.0e-12


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class LabelRecord:
    image_path: str
    mask_path: str
    scientific_mask_path: str
    gapfill_mask_path: str
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
    freq_regrid_method: str
    gap_fill_method: str
    time_resample_method: str
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
    coverage_fraction_scientific: float
    coverage_fraction_visual: float

    raw_dynamic_range_p05_p95: float
    residual_noise_sigma: float
    residual_noise_relative: float
    common_mode_sigma_after_prefilter: float
    common_mode_strength_after_prefilter: float
    common_mode_energy_ratio_after_prefilter: float
    robust_peak_snr_after_prefilter: float

    n_gap_rows_filled: int
    gapfill_fraction_total_grid: float
    gapfill_fraction_observed_band: float
    first_valid_freq_after_regrid_mhz: float
    last_valid_freq_after_regrid_mhz: float

    norm_p_high: float
    norm_p_high_value: float
    norm_min_value: float
    norm_max_value: float
    visual_gamma: float

    freq_source_used: str
    output_image_path: str
    output_mask_path: str
    output_scientific_mask_path: str
    output_gapfill_mask_path: str


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
    Estructura esperada para NO_SRB:

    no_srb_fits/ESTACION/AÑO/archivo.fit.gz

    Todos los FITS reciben la etiqueta fija NO_SRB.
    """
    rel = fit_path.relative_to(root)
    parts = rel.parts

    if len(parts) < 3:
        raise ValueError(f"Estructura inesperada: {fit_path}")

    original_type_folder = "NO_SRB"
    station_folder = str(parts[0]).strip()
    year = str(parts[1]).strip()

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

        for colname in hdu.columns.names:
            name_lower = str(colname).lower()

            if name_lower not in {"frequency", "freq"}:
                continue

            try:
                vals = sanitize_freq_values(np.asarray(hdu.data[colname][0]))
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

def sort_and_deduplicate_freq(
    data_ft: np.ndarray,
    freqs_mhz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    freqs = np.asarray(freqs_mhz, dtype=np.float64).ravel()
    data = np.asarray(data_ft, dtype=np.float64)

    if data.shape[0] != freqs.size:
        raise ValueError(
            f"No coincide eje de frecuencia con filas de datos: "
            f"data.shape={data.shape}, freqs={freqs.size}"
        )

    valid = np.isfinite(freqs)

    if np.count_nonzero(valid) < 2:
        raise ValueError("Menos de 2 frecuencias válidas.")

    data = data[valid, :]
    freqs = freqs[valid]

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


def crop_frequency_range(
    data_ft: np.ndarray,
    freqs_mhz: np.ndarray,
    freq_min_mhz: float,
    freq_max_mhz: float,
) -> tuple[np.ndarray, np.ndarray]:
    data, freqs = sort_and_deduplicate_freq(data_ft, freqs_mhz)

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
) -> dict[str, float]:
    raw = np.asarray(raw_cropped_ft, dtype=np.float64)
    residual = np.asarray(filtered_ft, dtype=np.float64)
    freqs = np.asarray(freqs_cropped_mhz, dtype=np.float64)

    freq_min_after_crop = float(np.nanmin(freqs))
    freq_max_after_crop = float(np.nanmax(freqs))

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
        "raw_dynamic_range_p05_p95": float(raw_dynamic_range),
        "residual_noise_sigma": float(residual_noise_sigma),
        "residual_noise_relative": float(residual_noise_relative),
        "common_mode_sigma_after_prefilter": float(common_mode_sigma),
        "common_mode_strength_after_prefilter": float(common_mode_strength),
        "common_mode_energy_ratio_after_prefilter": float(common_mode_energy_ratio),
        "robust_peak_snr_after_prefilter": float(robust_peak_snr),
    }


# ============================================================
# REGRILLADO CONSERVATIVO EN FRECUENCIA
# ============================================================

def estimate_channel_intervals_from_centers(
    centers: np.ndarray,
    max_channel_width_factor: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Estima intervalos [lo, hi] para cada canal desde sus centros.

    Si entre dos canales hay un salto muy grande, se limita el ancho efectivo
    del canal para no rellenar el hueco completo como si hubiera cobertura real.
    """
    c = np.asarray(centers, dtype=np.float64).ravel()

    if c.size < 2:
        raise ValueError("Se necesitan al menos 2 centros de frecuencia.")

    if not np.all(np.diff(c) > 0):
        raise ValueError("Los centros de frecuencia deben estar estrictamente crecientes.")

    diffs = np.diff(c)
    positive_diffs = diffs[diffs > 0]

    if positive_diffs.size == 0:
        raise ValueError("No se pudo estimar el paso de frecuencia.")

    median_step = float(np.median(positive_diffs))
    max_half_width = 0.5 * median_step * float(max_channel_width_factor)

    lo = np.zeros_like(c)
    hi = np.zeros_like(c)

    for i in range(c.size):
        if i == 0:
            left_half = 0.5 * median_step
        else:
            left_half = 0.5 * (c[i] - c[i - 1])

        if i == c.size - 1:
            right_half = 0.5 * median_step
        else:
            right_half = 0.5 * (c[i + 1] - c[i])

        left_half = min(left_half, max_half_width)
        right_half = min(right_half, max_half_width)

        lo[i] = c[i] - left_half
        hi[i] = c[i] + right_half

        if hi[i] <= lo[i]:
            lo[i] = c[i] - 0.5 * median_step
            hi[i] = c[i] + 0.5 * median_step

    return lo, hi, median_step


def conservative_regrid_frequency(
    data_ft: np.ndarray,
    src_freqs: np.ndarray,
    target_freq_min: float,
    target_freq_max: float,
    target_h: int,
    max_channel_width_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """
    Regrillado conservativo por solapamiento de bins.

    Devuelve:
    - data_regridded: matriz [freq_destino, tiempo]
    - scientific_row_mask: 1 si esa fila tiene cobertura real directa
    - dst_freq_centers: centro de frecuencia de cada fila destino
    - meta
    """
    src_data, src_freqs = sort_and_deduplicate_freq(data_ft, src_freqs)

    src_lo, src_hi, median_step = estimate_channel_intervals_from_centers(
        centers=src_freqs,
        max_channel_width_factor=max_channel_width_factor,
    )

    dst_edges = np.linspace(
        target_freq_min,
        target_freq_max,
        target_h + 1,
        dtype=np.float64,
    )
    dst_lo = dst_edges[:-1]
    dst_hi = dst_edges[1:]
    dst_centers = 0.5 * (dst_lo + dst_hi)

    overlap = np.maximum(
        0.0,
        np.minimum(dst_hi[:, None], src_hi[None, :])
        - np.maximum(dst_lo[:, None], src_lo[None, :]),
    )

    weights = overlap.sum(axis=1)
    scientific_valid_rows = weights > EPS

    out = np.zeros((target_h, src_data.shape[1]), dtype=np.float64)

    if np.any(scientific_valid_rows):
        out[scientific_valid_rows, :] = overlap[scientific_valid_rows, :] @ src_data
        out[scientific_valid_rows, :] /= weights[scientific_valid_rows, None]

    meta = {
        "src_freq_bin_median_width_mhz": float(median_step),
        "coverage_fraction_scientific": float(np.mean(scientific_valid_rows)),
    }

    return out, scientific_valid_rows.astype(np.uint8), dst_centers, meta


# ============================================================
# RELLENO DE GAPS INTERNOS CON PCHIP
# ============================================================

def contiguous_true_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """
    Devuelve segmentos contiguos True como pares inclusivos: (start, end).
    """
    m = np.asarray(mask, dtype=bool).ravel()

    if m.size == 0:
        return []

    idx = np.where(m)[0]

    if idx.size == 0:
        return []

    segments: list[tuple[int, int]] = []
    start = int(idx[0])
    prev = int(idx[0])

    for current in idx[1:]:
        current = int(current)

        if current == prev + 1:
            prev = current
        else:
            segments.append((start, prev))
            start = current
            prev = current

    segments.append((start, prev))
    return segments


def clip_filled_gaps_locally(
    filled_data: np.ndarray,
    original_data: np.ndarray,
    scientific_row_mask: np.ndarray,
    gapfill_row_mask: np.ndarray,
    context_rows: int,
) -> np.ndarray:
    """
    Limita los valores interpolados al rango local de canales reales vecinos.

    Esto reduce el riesgo de que PCHIP genere valores demasiado altos o bajos
    dentro de gaps anchos.
    """
    out = np.asarray(filled_data, dtype=np.float64).copy()
    original = np.asarray(original_data, dtype=np.float64)
    scientific_valid = np.asarray(scientific_row_mask, dtype=bool).ravel()
    gap_rows = np.asarray(gapfill_row_mask, dtype=bool).ravel()

    valid_idx = np.where(scientific_valid)[0]

    if valid_idx.size == 0:
        return out

    segments = contiguous_true_segments(gap_rows)

    for start, end in segments:
        left_valid = valid_idx[valid_idx < start]
        right_valid = valid_idx[valid_idx > end]

        left_context = left_valid[-context_rows:] if left_valid.size else np.array([], dtype=int)
        right_context = right_valid[:context_rows] if right_valid.size else np.array([], dtype=int)

        context_idx = np.concatenate([left_context, right_context])

        if context_idx.size < 2:
            continue

        context_data = original[context_idx, :]

        local_min = np.nanmin(context_data, axis=0)
        local_max = np.nanmax(context_data, axis=0)

        lo = np.minimum(local_min, local_max)
        hi = np.maximum(local_min, local_max)

        out[start:end + 1, :] = np.clip(
            out[start:end + 1, :],
            lo[None, :],
            hi[None, :],
        )

    return out


def fill_internal_frequency_gaps_pchip(
    data_ft: np.ndarray,
    scientific_row_mask: np.ndarray,
    dst_freq_centers: np.ndarray,
    clip_context_rows: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """
    Rellena gaps internos de frecuencia con PCHIP por columna temporal,
    pero de forma vectorizada.

    Diferencia clave respecto a la versión lenta:
    - Antes: se creaba un PCHIP por cada columna temporal.
    - Ahora: se crea un solo PCHIP por FITS usando axis=0.

    Reglas:
    - Solo rellena filas entre la primera y última fila con cobertura real.
    - No extrapola por debajo ni por encima del rango real observado.
    - Usa PCHIP si hay al menos 4 filas reales finitas.
    - Usa interpolación lineal vectorizada como fallback si hay 2 o 3 filas reales.
    - Aplica clipping local para evitar overshoot.
    """
    data = np.asarray(data_ft, dtype=np.float64)
    filled = data.copy()

    scientific_valid = np.asarray(scientific_row_mask, dtype=bool).ravel()
    dst_freqs = np.asarray(dst_freq_centers, dtype=np.float64).ravel()

    if data.shape[0] != scientific_valid.size:
        raise ValueError("scientific_row_mask no coincide con filas de data_ft.")

    if data.shape[0] != dst_freqs.size:
        raise ValueError("dst_freq_centers no coincide con filas de data_ft.")

    gapfill_row_mask = np.zeros(data.shape[0], dtype=np.uint8)

    valid_idx = np.where(scientific_valid)[0]

    if valid_idx.size < 2:
        meta = {
            "n_gap_rows_filled": 0,
            "gapfill_fraction_total_grid": 0.0,
            "gapfill_fraction_observed_band": 0.0,
            "first_valid_freq_after_regrid_mhz": float("nan"),
            "last_valid_freq_after_regrid_mhz": float("nan"),
        }
        return filled, gapfill_row_mask, meta

    first_valid = int(valid_idx[0])
    last_valid = int(valid_idx[-1])

    internal_rows = np.arange(first_valid, last_valid + 1)
    internal_gap_rows = internal_rows[~scientific_valid[internal_rows]]

    if internal_gap_rows.size == 0:
        meta = {
            "n_gap_rows_filled": 0,
            "gapfill_fraction_total_grid": 0.0,
            "gapfill_fraction_observed_band": 0.0,
            "first_valid_freq_after_regrid_mhz": float(dst_freqs[first_valid]),
            "last_valid_freq_after_regrid_mhz": float(dst_freqs[last_valid]),
        }
        return filled, gapfill_row_mask, meta

    gapfill_row_mask[internal_gap_rows] = 1

    x_valid = dst_freqs[scientific_valid]
    y_valid = data[scientific_valid, :]
    x_target = dst_freqs[internal_gap_rows]

    finite_rows = np.isfinite(x_valid) & np.all(np.isfinite(y_valid), axis=1)

    x_valid = x_valid[finite_rows]
    y_valid = y_valid[finite_rows, :]

    if x_valid.size >= 4:
        interpolator = PchipInterpolator(
            x_valid,
            y_valid,
            axis=0,
            extrapolate=False,
        )

        y_fill = interpolator(x_target)

    elif x_valid.size >= 2:
        # Fallback lineal vectorizado para casos raros con muy pocas filas válidas.
        y_fill = np.empty((x_target.size, data.shape[1]), dtype=np.float64)

        for col_idx in range(data.shape[1]):
            y_fill[:, col_idx] = np.interp(
                x_target,
                x_valid,
                y_valid[:, col_idx],
            )

    else:
        meta = {
            "n_gap_rows_filled": 0,
            "gapfill_fraction_total_grid": 0.0,
            "gapfill_fraction_observed_band": 0.0,
            "first_valid_freq_after_regrid_mhz": float(dst_freqs[first_valid]),
            "last_valid_freq_after_regrid_mhz": float(dst_freqs[last_valid]),
        }
        return filled, np.zeros_like(gapfill_row_mask), meta

    y_fill = np.asarray(y_fill, dtype=np.float64)
    y_fill[~np.isfinite(y_fill)] = 0.0

    filled[internal_gap_rows, :] = y_fill

    filled = clip_filled_gaps_locally(
        filled_data=filled,
        original_data=data,
        scientific_row_mask=scientific_row_mask,
        gapfill_row_mask=gapfill_row_mask,
        context_rows=clip_context_rows,
    )

    observed_band_rows = max(last_valid - first_valid + 1, 1)

    meta = {
        "n_gap_rows_filled": int(np.count_nonzero(gapfill_row_mask)),
        "gapfill_fraction_total_grid": float(np.mean(gapfill_row_mask)),
        "gapfill_fraction_observed_band": float(
            np.count_nonzero(gapfill_row_mask) / observed_band_rows
        ),
        "first_valid_freq_after_regrid_mhz": float(dst_freqs[first_valid]),
        "last_valid_freq_after_regrid_mhz": float(dst_freqs[last_valid]),
    }

    return filled, gapfill_row_mask, meta


# ============================================================
# REMUESTREO TEMPORAL POR ÁREA
# ============================================================

def area_resample_time_axis(data_ft: np.ndarray, target_w: int) -> np.ndarray:
    """
    Remuestreo temporal por promedio ponderado de área.

    Para CALLISTO suele ser:
    3600 muestras temporales -> 656 columnas.
    """
    data = np.asarray(data_ft, dtype=np.float64)
    n_freq, src_w = data.shape

    if src_w == target_w:
        return data.copy()

    if src_w < 2:
        raise ValueError("No se puede remuestrear tiempo con menos de 2 columnas.")

    src_edges = np.linspace(0.0, 1.0, src_w + 1, dtype=np.float64)
    dst_edges = np.linspace(0.0, 1.0, target_w + 1, dtype=np.float64)

    out = np.zeros((n_freq, target_w), dtype=np.float64)

    for i in range(target_w):
        lo = dst_edges[i]
        hi = dst_edges[i + 1]

        j0 = int(np.searchsorted(src_edges, lo, side="right") - 1)
        j1 = int(np.searchsorted(src_edges, hi, side="left"))

        j0 = max(0, min(j0, src_w - 1))
        j1 = max(j0, min(j1, src_w - 1))

        idx = np.arange(j0, j1 + 1)

        left = np.maximum(lo, src_edges[idx])
        right = np.minimum(hi, src_edges[idx + 1])
        weights = np.maximum(0.0, right - left)

        wsum = float(np.sum(weights))

        if wsum <= EPS:
            continue

        out[:, i] = data[:, idx] @ weights / wsum

    return out


def build_coverage_mask(freq_row_mask: np.ndarray, target_w: int) -> np.ndarray:
    return np.repeat(freq_row_mask[:, None], target_w, axis=1).astype(np.uint8)


# ============================================================
# NORMALIZACIÓN VISUAL POSITIVA Y COLORMAP
# ============================================================

def positive_percentile_normalize(
    image_ft: np.ndarray,
    coverage_mask_ft: np.ndarray,
    p_high: float,
    gamma: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Normalización positiva:
    - valores negativos se llevan a 0;
    - 0 queda como fondo;
    - p_high de valores positivos se mapea a 1;
    - opcionalmente se aplica gamma visual.
    """
    valid = coverage_mask_ft > 0

    img = np.asarray(image_ft, dtype=np.float64).copy()
    img[~np.isfinite(img)] = 0.0

    img = np.maximum(img, 0.0)
    img[~valid] = 0.0

    vals = img[valid]
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        meta = {
            "norm_p_high_value": float("nan"),
            "norm_min_value": float("nan"),
            "norm_max_value": float("nan"),
        }
        return np.zeros_like(img, dtype=np.float64), meta

    vals_pos = vals[vals > 0.0]

    if vals_pos.size == 0:
        meta = {
            "norm_p_high_value": 0.0,
            "norm_min_value": float(np.min(vals)),
            "norm_max_value": float(np.max(vals)),
        }
        return np.zeros_like(img, dtype=np.float64), meta

    hi = float(np.percentile(vals_pos, p_high))
    hi = max(hi, EPS)

    out = img / hi
    out = np.clip(out, 0.0, 1.0)

    if gamma != 1.0:
        gamma = max(float(gamma), EPS)
        out = out ** gamma

    out[~valid] = 0.0

    meta = {
        "norm_p_high_value": hi,
        "norm_min_value": float(np.min(vals)),
        "norm_max_value": float(np.max(vals)),
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
    max_channel_width_factor: float,
    pchip_clip_context_rows: int,
    positive_p_high: float,
    visual_gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
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

    # 1) Recorte inferior y superior por centros reales del FITS.
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

    # 4) Métricas en matriz nativa ya prefiltrada.
    quality = compute_quality_metrics_after_prefilter(
        raw_cropped_ft=cropped_ft,
        filtered_ft=filtered_ft,
        freqs_cropped_mhz=cropped_freqs,
    )

    # 5) Regrillado conservativo.
    regridded_ft, scientific_row_mask, dst_freq_centers, regrid_meta = conservative_regrid_frequency(
        data_ft=filtered_ft,
        src_freqs=cropped_freqs,
        target_freq_min=target_freq_min,
        target_freq_max=target_freq_max,
        target_h=image_size,
        max_channel_width_factor=max_channel_width_factor,
    )

    # 6) Relleno de gaps internos con PCHIP + clipping local.
    gapfilled_ft, gapfill_row_mask, gapfill_meta = fill_internal_frequency_gaps_pchip(
        data_ft=regridded_ft,
        scientific_row_mask=scientific_row_mask,
        dst_freq_centers=dst_freq_centers,
        clip_context_rows=pchip_clip_context_rows,
    )

    visual_row_mask = np.maximum(scientific_row_mask, gapfill_row_mask).astype(np.uint8)

    # 7) Remuestreo temporal por área.
    final_ft = area_resample_time_axis(
        data_ft=gapfilled_ft,
        target_w=image_size,
    )

    # 8) Máscaras.
    visual_mask_ft = build_coverage_mask(visual_row_mask, image_size)
    scientific_mask_ft = build_coverage_mask(scientific_row_mask, image_size)
    gapfill_mask_ft = build_coverage_mask(gapfill_row_mask, image_size)

    coverage_fraction_visual = float(np.mean(visual_row_mask))

    # 9) Normalización visual positiva.
    norm_ft, norm_meta = positive_percentile_normalize(
        image_ft=final_ft,
        coverage_mask_ft=visual_mask_ft,
        p_high=positive_p_high,
        gamma=visual_gamma,
    )

    # 10) Colormap.
    rgb_u8 = apply_jet_colormap(norm_ft, visual_mask_ft)

    visual_mask_u8 = (visual_mask_ft * 255).astype(np.uint8)
    scientific_mask_u8 = (scientific_mask_ft * 255).astype(np.uint8)
    gapfill_mask_u8 = (gapfill_mask_ft * 255).astype(np.uint8)

    # Frecuencias altas arriba, bajas abajo.
    rgb_u8 = np.flipud(rgb_u8)
    visual_mask_u8 = np.flipud(visual_mask_u8)
    scientific_mask_u8 = np.flipud(scientific_mask_u8)
    gapfill_mask_u8 = np.flipud(gapfill_mask_u8)

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
        "coverage_fraction_visual": coverage_fraction_visual,
    }

    meta.update(quality)
    meta.update(regrid_meta)
    meta.update(gapfill_meta)
    meta.update(norm_meta)

    return rgb_u8, visual_mask_u8, scientific_mask_u8, gapfill_mask_u8, meta


# ============================================================
# GUARDADO
# ============================================================

def save_rgb_png(array_u8: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array_u8, mode="RGB").save(path)


def save_gray_png(array_u8: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array_u8, mode="L").save(path)

def merge_existing_csv(
    path: Path,
    new_rows: list[dict[str, Any]],
    deduplicate_by: str,
) -> pd.DataFrame:
    """
    Agrega nuevas filas a un CSV existente sin perder datos anteriores.

    Si el CSV no existe, crea uno nuevo.
    Si una fila nueva tiene el mismo valor de deduplicate_by,
    la nueva reemplaza a la anterior.
    """
    new_df = pd.DataFrame(new_rows)

    if path.exists():
        existing_df = pd.read_csv(path)
    else:
        existing_df = pd.DataFrame()

    if existing_df.empty:
        combined = new_df.copy()
    elif new_df.empty:
        combined = existing_df.copy()
    else:
        combined = pd.concat(
            [existing_df, new_df],
            ignore_index=True,
            sort=False,
        )

    if not combined.empty and deduplicate_by in combined.columns:
        combined = combined.drop_duplicates(
            subset=[deduplicate_by],
            keep="last",
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False, encoding="utf-8")

    return combined


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

    for cls in ["II", "III", "IV", "NO_SRB"]:
        if cls not in summary.columns:
            summary[cls] = 0

    summary = summary.rename(
        columns={
            "II": "n_II",
            "III": "n_III",
            "IV": "n_IV",
            "NO_SRB": "n_NO_SRB",
        }
    )

    summary["n_total"] = summary[["n_II", "n_III", "n_IV", "n_NO_SRB"]].sum(axis=1)

    return summary.sort_values("n_total",ascending=False,).reset_index(drop=True)


def build_station_quality_summary(
    quality_df: pd.DataFrame,
    failed_df: pd.DataFrame,
) -> pd.DataFrame:
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
            coverage_scientific_median=("coverage_fraction_scientific", "median"),
            coverage_visual_median=("coverage_fraction_visual", "median"),
            gapfill_fraction_observed_band_median=("gapfill_fraction_observed_band", "median"),
            gapfill_fraction_observed_band_p75=("gapfill_fraction_observed_band", lambda x: np.nanpercentile(x, 75)),
            n_gap_rows_filled_median=("n_gap_rows_filled", "median"),
            n_gap_rows_filled_p75=("n_gap_rows_filled", lambda x: np.nanpercentile(x, 75)),
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
    max_channel_width_factor: float,
    pchip_clip_context_rows: int,
    positive_p_high: float,
    visual_gamma: float,
) -> pd.DataFrame:
    total_images = int(len(labels_df))
    total_failed = int(len(failed_df))

    counts = {
    "II": 0,
    "III": 0,
    "IV": 0,
    "NO_SRB": 0,
    }

    if not labels_df.empty:
        vc = labels_df["final_label"].value_counts().to_dict()
        for cls in counts:
            counts[cls] = int(vc.get(cls, 0))

    if quality_df.empty:
        median_gapfill_band = float("nan")
        median_gap_rows = float("nan")
    else:
        ok_df = quality_df[quality_df["status"].eq("ok")]
        median_gapfill_band = safe_float(ok_df["gapfill_fraction_observed_band"].median()) if not ok_df.empty else float("nan")
        median_gap_rows = safe_float(ok_df["n_gap_rows_filled"].median()) if not ok_df.empty else float("nan")

    row = {
        "dataset_name": output_root.name,
        "raw_dataset_root": str(raw_root),
        "output_root": str(output_root),
        "stations": ";".join(sorted(TOP6_STATIONS)),
        "image_size": int(image_size),
        "target_freq_min_mhz": float(target_freq_min),
        "target_freq_max_mhz": float(target_freq_max),
        "prefilter_method": "subtract_temporal_mean_per_frequency",
        "freq_regrid_method": "conservative_bin_overlap",
        "gap_fill_method": "pchip_internal_only_local_clipping",
        "time_resample_method": "area_weighted_mean",
        "normalization_method": f"positive_percentile_p{positive_p_high:g}_gamma_{visual_gamma:g}",
        "max_channel_width_factor": float(max_channel_width_factor),
        "pchip_clip_context_rows": int(pchip_clip_context_rows),
        "total_images": total_images,
        "total_II": counts["II"],
        "total_III": counts["III"],
        "total_IV": counts["IV"],
        "total_NO_SRB": counts["NO_SRB"],
        "total_failed": total_failed,
        "quality_rows": int(len(quality_df)),
        "median_gapfill_fraction_observed_band": median_gapfill_band,
        "median_gap_rows_filled": median_gap_rows,
    }

    return pd.DataFrame([row])


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Genera dataset preliminar top 6 con recorte 50-870 MHz, "
            "resta de media temporal por frecuencia, regrillado conservativo, "
            "relleno de gaps internos con PCHIP y normalización positiva."
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
        "--max-channel-width-factor",
        type=float,
        default=DEFAULT_MAX_CHANNEL_WIDTH_FACTOR,
        help=(
            "Factor máximo de ancho de canal para regrillado conservativo. "
            "Default: 3.0."
        ),
    )

    parser.add_argument(
        "--pchip-clip-context-rows",
        type=int,
        default=DEFAULT_PCHIP_CLIP_CONTEXT_ROWS,
        help=(
            "Cantidad de filas reales vecinas usadas para clipping local del PCHIP. "
            "Default: 4."
        ),
    )

    parser.add_argument(
        "--positive-p-high",
        type=float,
        default=DEFAULT_POSITIVE_P_HIGH,
        help="Percentil alto positivo para normalización visual. Default: 99.7.",
    )

    parser.add_argument(
        "--visual-gamma",
        type=float,
        default=DEFAULT_VISUAL_GAMMA,
        help="Gamma visual. 1.0 no altera; >1 oscurece valores bajos. Default: 1.0.",
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
    max_channel_width_factor: float = float(args.max_channel_width_factor)
    pchip_clip_context_rows: int = int(args.pchip_clip_context_rows)
    positive_p_high: float = float(args.positive_p_high)
    visual_gamma: float = float(args.visual_gamma)

    if image_size < 32:
        raise ValueError("image-size demasiado pequeño.")

    if target_freq_max <= target_freq_min:
        raise ValueError("freq-max debe ser mayor que freq-min.")

    if max_channel_width_factor <= 0:
        raise ValueError("max-channel-width-factor debe ser mayor que 0.")

    if pchip_clip_context_rows < 1:
        raise ValueError("pchip-clip-context-rows debe ser al menos 1.")

    if positive_p_high <= 0 or positive_p_high >= 100:
        raise ValueError("positive-p-high debe estar entre 0 y 100.")

    if visual_gamma <= 0:
        raise ValueError("visual-gamma debe ser mayor que 0.")

    if not raw_root.exists():
        raise FileNotFoundError(f"No existe raw-root: {raw_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "images").mkdir(exist_ok=True)
    (output_root / "masks").mkdir(exist_ok=True)
    (output_root / "scientific_masks").mkdir(exist_ok=True)
    (output_root / "gapfill_masks").mkdir(exist_ok=True)

    labels_csv = output_root / "labels.csv"
    quality_csv = output_root / "quality_by_image.csv"
    station_class_summary_csv = output_root / "station_class_summary.csv"
    station_quality_summary_csv = output_root / "station_quality_summary.csv"
    dataset_summary_csv = output_root / "dataset_summary.csv"
    used_stations_csv = output_root / "used_stations.csv"
    failed_csv = output_root / "failed_files.csv"

    allowed_stations = set(TOP6_STATIONS)

    print("[INFO] Generador NO_SRB top 6 - PCHIP gap fill")
    print(f"[INFO] Dataset FITS: {raw_root}")
    print(f"[INFO] Salida: {output_root}")
    print(f"[INFO] Estaciones: {', '.join(sorted(allowed_stations))}")
    print("[INFO] Clase generada: NO_SRB")
    print(f"[INFO] Frecuencia efectiva: {target_freq_min:g}-{target_freq_max:g} MHz")
    print(f"[INFO] Tamaño de imagen: {image_size}x{image_size}")
    print("[INFO] Prefiltro: resta de media temporal por frecuencia")
    print("[INFO] Regrillado frecuencia: conservative_bin_overlap")
    print("[INFO] Gap fill: PCHIP solo en gaps internos + clipping local")
    print("[INFO] Remuestreo temporal: area_weighted_mean")
    print(f"[INFO] Normalización visual: positive p{positive_p_high:g}, gamma={visual_gamma:g}")
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

    print(f"[INFO] FITS candidatos NO_SRB top 6: {len(candidate_fits)}")
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

            image_rgb_u8, mask_u8, scientific_mask_u8, gapfill_mask_u8, meta = process_fit_to_image_mask_and_quality(
                fit_path=fit_path,
                target_freq_min=target_freq_min,
                target_freq_max=target_freq_max,
                image_size=image_size,
                max_channel_width_factor=max_channel_width_factor,
                pchip_clip_context_rows=pchip_clip_context_rows,
                positive_p_high=positive_p_high,
                visual_gamma=visual_gamma,
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
            scientific_mask_path = output_root / "scientific_masks" / final_label / out_name
            gapfill_mask_path = output_root / "gapfill_masks" / final_label / out_name

            save_rgb_png(image_rgb_u8, img_path)
            save_gray_png(mask_u8, mask_path)
            save_gray_png(scientific_mask_u8, scientific_mask_path)
            save_gray_png(gapfill_mask_u8, gapfill_mask_path)

            label_rows.append(asdict(LabelRecord(
                image_path=str(img_path),
                mask_path=str(mask_path),
                scientific_mask_path=str(scientific_mask_path),
                gapfill_mask_path=str(gapfill_mask_path),
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
                freq_regrid_method="conservative_bin_overlap",
                gap_fill_method="pchip_internal_only_local_clipping",
                time_resample_method="area_weighted_mean",
                normalization_method=f"positive_percentile_p{positive_p_high:g}_gamma_{visual_gamma:g}",
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
                coverage_fraction_scientific=safe_float(meta["coverage_fraction_scientific"]),
                coverage_fraction_visual=safe_float(meta["coverage_fraction_visual"]),

                raw_dynamic_range_p05_p95=safe_float(meta["raw_dynamic_range_p05_p95"]),
                residual_noise_sigma=safe_float(meta["residual_noise_sigma"]),
                residual_noise_relative=safe_float(meta["residual_noise_relative"]),
                common_mode_sigma_after_prefilter=safe_float(meta["common_mode_sigma_after_prefilter"]),
                common_mode_strength_after_prefilter=safe_float(meta["common_mode_strength_after_prefilter"]),
                common_mode_energy_ratio_after_prefilter=safe_float(meta["common_mode_energy_ratio_after_prefilter"]),
                robust_peak_snr_after_prefilter=safe_float(meta["robust_peak_snr_after_prefilter"]),

                n_gap_rows_filled=int(meta["n_gap_rows_filled"]),
                gapfill_fraction_total_grid=safe_float(meta["gapfill_fraction_total_grid"]),
                gapfill_fraction_observed_band=safe_float(meta["gapfill_fraction_observed_band"]),
                first_valid_freq_after_regrid_mhz=safe_float(meta["first_valid_freq_after_regrid_mhz"]),
                last_valid_freq_after_regrid_mhz=safe_float(meta["last_valid_freq_after_regrid_mhz"]),

                norm_p_high=positive_p_high,
                norm_p_high_value=safe_float(meta["norm_p_high_value"]),
                norm_min_value=safe_float(meta["norm_min_value"]),
                norm_max_value=safe_float(meta["norm_max_value"]),
                visual_gamma=visual_gamma,

                freq_source_used=str(meta["freq_source_used"]),
                output_image_path=str(img_path),
                output_mask_path=str(mask_path),
                output_scientific_mask_path=str(scientific_mask_path),
                output_gapfill_mask_path=str(gapfill_mask_path),
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
                coverage_fraction_scientific=float("nan"),
                coverage_fraction_visual=float("nan"),

                raw_dynamic_range_p05_p95=float("nan"),
                residual_noise_sigma=float("nan"),
                residual_noise_relative=float("nan"),
                common_mode_sigma_after_prefilter=float("nan"),
                common_mode_strength_after_prefilter=float("nan"),
                common_mode_energy_ratio_after_prefilter=float("nan"),
                robust_peak_snr_after_prefilter=float("nan"),

                n_gap_rows_filled=0,
                gapfill_fraction_total_grid=float("nan"),
                gapfill_fraction_observed_band=float("nan"),
                first_valid_freq_after_regrid_mhz=float("nan"),
                last_valid_freq_after_regrid_mhz=float("nan"),

                norm_p_high=positive_p_high,
                norm_p_high_value=float("nan"),
                norm_min_value=float("nan"),
                norm_max_value=float("nan"),
                visual_gamma=visual_gamma,

                freq_source_used="",
                output_image_path="",
                output_mask_path="",
                output_scientific_mask_path="",
                output_gapfill_mask_path="",
            )))

    labels_df = merge_existing_csv(
        path=labels_csv,
        new_rows=label_rows,
        deduplicate_by="source_fit",
    )

    quality_df = merge_existing_csv(
        path=quality_csv,
        new_rows=quality_rows,
        deduplicate_by="source_fit",
    )

    failed_df = merge_existing_csv(
        path=failed_csv,
        new_rows=failed_rows,
        deduplicate_by="source_fit",
    )

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
        max_channel_width_factor=max_channel_width_factor,
        pchip_clip_context_rows=pchip_clip_context_rows,
        positive_p_high=positive_p_high,
        visual_gamma=visual_gamma,
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