#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import io
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from astropy.io import fits
from PIL import Image
from scipy.ndimage import median_filter
from tqdm import tqdm

# =========================================================
# CONFIG
# =========================================================

RAW_DATASET_ROOT = Path(r"C:\Users\Macri\Desktop\Facultad\Tesis\dataset")

# CSV local que ya generaste antes con rangos por estación
STATION_RANGES_INPUT_CSV = Path(
    r"C:\Users\Macri\Desktop\Facultad\Tesis\dataset_images_768_medianmed_debug\station_frequency_ranges.csv"
)

OUTPUT_ROOT = Path(r"C:\Users\Macri\Desktop\Facultad\Tesis\dataset_images_768_validstations")

# Rango estándar e-CALLISTO
TARGET_FREQ_MIN_MHZ = 45.0
TARGET_FREQ_MAX_MHZ = 870.0

# Tolerancia pequeña para evitar excluir casos tipo 44.988 o 869.938 si hiciera falta
FREQ_TOL_MHZ = 0.5

# Tamaño final
IMG_H = 768
IMG_W = 768

# Filtro final
MEDIAN_FILTER_SIZE = 3

FIT_EXTENSIONS = {".fit", ".fits", ".fit.gz", ".fits.gz"}

LABELS_CSV = OUTPUT_ROOT / "labels.csv"
FAILED_CSV = OUTPUT_ROOT / "failed_files.csv"
VALID_STATIONS_CSV = OUTPUT_ROOT / "valid_stations_used.csv"
INVALID_STATIONS_CSV = OUTPUT_ROOT / "invalid_stations_excluded.csv"


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
# STATION FILTERING FROM LOCAL CSV
# =========================================================

def load_valid_invalid_stations_from_csv(csv_path: Path) -> tuple[set[str], pd.DataFrame, pd.DataFrame]:
    if not csv_path.exists():
        raise FileNotFoundError(f"No encuentro el CSV local de rangos por estación: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"station", "global_min_mhz", "global_max_mhz"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {csv_path.name}: {missing}")

    df = df.copy()
    df["station"] = df["station"].astype(str).str.strip()

    valid_mask = (
        (df["global_min_mhz"] >= (TARGET_FREQ_MIN_MHZ - FREQ_TOL_MHZ)) &
        (df["global_max_mhz"] <= (TARGET_FREQ_MAX_MHZ + FREQ_TOL_MHZ))
    )

    valid_df = df.loc[valid_mask].sort_values("station").reset_index(drop=True)
    invalid_df = df.loc[~valid_mask].sort_values("station").reset_index(drop=True)

    valid_stations = set(valid_df["station"].tolist())
    return valid_stations, valid_df, invalid_df


# =========================================================
# FITS HELPERS
# =========================================================

def open_fits_any(path: Path) -> fits.HDUList:
    lower = path.name.lower()
    if lower.endswith(".gz"):
        raw = gzip.decompress(path.read_bytes())
        return fits.open(io.BytesIO(raw), memmap=False, lazy_load_hdus=False)
    return fits.open(path, memmap=False, lazy_load_hdus=False)


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


def try_extract_freq_axis_from_table(hdul: fits.HDUList) -> Optional[np.ndarray]:
    for hdu in hdul:
        if not isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
            continue
        if hdu.data is None or hdu.columns is None:
            continue
        for col in hdu.columns:
            name = col.name.lower()
            if "freq" in name:
                try:
                    vals = sanitize_freq_values(np.asarray(hdu.data[col.name]))
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


def orient_data_with_freq(hdul: fits.HDUList, data2d: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    data = np.asarray(data2d, dtype=np.float64)

    freq_from_table = try_extract_freq_axis_from_table(hdul)
    if freq_from_table is not None:
        if freq_from_table.size == data.shape[0]:
            return data, freq_from_table, False
        if freq_from_table.size == data.shape[1]:
            return data.T, freq_from_table, True

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
            return data, freq0, False

        freq1 = try_extract_freq_axis_from_header(hdu.header, data.shape[1])
        if freq1 is not None:
            return data.T, freq1, True

    freq0 = try_extract_freq_axis_from_header(hdul[0].header, data.shape[0])
    if freq0 is not None:
        return data, freq0, False

    freq1 = try_extract_freq_axis_from_header(hdul[0].header, data.shape[1])
    if freq1 is not None:
        return data.T, freq1, True

    # heurística final
    if data.shape[0] <= data.shape[1]:
        freq_guess = np.linspace(TARGET_FREQ_MIN_MHZ, TARGET_FREQ_MAX_MHZ, data.shape[0], dtype=np.float64)
        return data, freq_guess, False
    else:
        freq_guess = np.linspace(TARGET_FREQ_MIN_MHZ, TARGET_FREQ_MAX_MHZ, data.shape[1], dtype=np.float64)
        return data.T, freq_guess, True


# =========================================================
# PROCESSING
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


def subtract_frequency_median_background(data_ft: np.ndarray) -> np.ndarray:
    data = np.asarray(data_ft, dtype=np.float64).copy()
    med = np.median(data, axis=1, keepdims=True)
    return data - med


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


def robust_normalize(image_ft: np.ndarray, coverage_mask_ft: np.ndarray) -> np.ndarray:
    valid = coverage_mask_ft > 0
    vals = image_ft[valid]

    if vals.size == 0:
        return np.zeros_like(image_ft, dtype=np.float64)

    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros_like(image_ft, dtype=np.float64)

    p2 = np.percentile(vals, 2)
    p98 = np.percentile(vals, 98)

    if not np.isfinite(p2) or not np.isfinite(p98) or p98 <= p2:
        mn = np.min(vals)
        mx = np.max(vals)
        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
            return np.zeros_like(image_ft, dtype=np.float64)
        p2, p98 = mn, mx

    out = (image_ft - p2) / (p98 - p2)
    out = np.clip(out, 0.0, 1.0)
    out[~valid] = 0.0
    return out


def apply_median_filter(image_ft: np.ndarray, coverage_mask_ft: np.ndarray, size: int = 3) -> np.ndarray:
    filtered = median_filter(image_ft, size=size, mode="nearest")
    filtered[coverage_mask_ft == 0] = 0.0
    return filtered


def to_uint8_image(img01: np.ndarray) -> np.ndarray:
    return np.clip(img01 * 255.0, 0, 255).astype(np.uint8)


def process_fit_to_image_and_mask(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open_fits_any(path) as hdul:
        _, raw2d = first_2d_array_hdu(hdul)
        data_ft, freqs_mhz, _ = orient_data_with_freq(hdul, raw2d)

    data_ft = np.asarray(data_ft, dtype=np.float64)
    data_ft[~np.isfinite(data_ft)] = 0.0

    # 1) quitar fondo persistente por frecuencia
    data_ft = subtract_frequency_median_background(data_ft)

    # 2) grilla objetivo fija 45–870 MHz
    dst_freqs = np.linspace(TARGET_FREQ_MIN_MHZ, TARGET_FREQ_MAX_MHZ, IMG_H, dtype=np.float64)

    # 3) interpolar sobre frecuencia
    interp_ft, freq_row_mask = interpolate_frequency_axis(data_ft, freqs_mhz, dst_freqs)

    # 4) remuestrear tiempo
    interp_ft = resize_time_axis(interp_ft, IMG_W)

    # 5) máscara
    coverage_mask_ft = build_coverage_mask(freq_row_mask, IMG_W)

    # 6) normalización robusta
    norm_ft = robust_normalize(interp_ft, coverage_mask_ft)

    # 7) filtro de mediana
    filtered_ft = apply_median_filter(norm_ft, coverage_mask_ft, size=MEDIAN_FILTER_SIZE)

    img_u8 = to_uint8_image(filtered_ft)
    mask_u8 = (coverage_mask_ft * 255).astype(np.uint8)

    # frecuencias altas arriba, bajas abajo
    img_u8 = np.flipud(img_u8)
    mask_u8 = np.flipud(mask_u8)

    return img_u8, mask_u8


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


# =========================================================
# SAVE OUTPUTS
# =========================================================

def save_png(array_u8: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array_u8, mode="L").save(path)


def build_output_name(final_label: str, station: str, year: str, source_fit: Path) -> str:
    stem = source_fit.name
    for ext in [".fit.gz", ".fits.gz", ".fit", ".fits"]:
        if stem.lower().endswith(ext):
            stem = stem[:-len(ext)]
            break
    station_clean = station.replace("/", "_")
    return f"{final_label}_{station_clean}_{year}_{stem}.png"


# =========================================================
# MAIN
# =========================================================

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "images").mkdir(exist_ok=True)
    (OUTPUT_ROOT / "masks").mkdir(exist_ok=True)

    label_rows: list[dict] = []
    failed_rows: list[dict] = []

    # -----------------------------------------------------
    # 1) Cargar estaciones válidas/no válidas desde CSV local
    # -----------------------------------------------------
    valid_stations, valid_df, invalid_df = load_valid_invalid_stations_from_csv(STATION_RANGES_INPUT_CSV)

    valid_df.to_csv(VALID_STATIONS_CSV, index=False)
    invalid_df.to_csv(INVALID_STATIONS_CSV, index=False)

    print(f"[INFO] Estaciones válidas encontradas: {len(valid_df)}")
    print(f"[INFO] Estaciones no válidas excluidas: {len(invalid_df)}")

    if len(invalid_df) > 0:
        print("[INFO] Estaciones NO válidas excluidas:")
        for _, row in invalid_df.iterrows():
            print(
                f"  - {row['station']}: "
                f"{row['global_min_mhz']:.3f} MHz a {row['global_max_mhz']:.3f} MHz"
            )

    fit_files = list(iter_raw_fit_files(RAW_DATASET_ROOT))
    print(f"[INFO] FITS encontrados en total: {len(fit_files)}")

    # -----------------------------------------------------
    # 2) Filtrar FITS por tipo válido y estación válida
    # -----------------------------------------------------
    valid_fit_files: list[Path] = []

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
    # 3) Generar imágenes y máscaras
    # -----------------------------------------------------
    for fit_path in tqdm(valid_fit_files, desc="Procesando FITS", unit="fit"):
        try:
            original_type_folder, station, year = infer_dataset_metadata(RAW_DATASET_ROOT, fit_path)
            final_label = map_to_final_label(original_type_folder)

            if final_label is None:
                continue

            image_u8, mask_u8 = process_fit_to_image_and_mask(fit_path)

            out_name = build_output_name(final_label, station, year, fit_path)

            img_path = OUTPUT_ROOT / "images" / final_label / out_name
            mask_path = OUTPUT_ROOT / "masks" / final_label / out_name

            save_png(image_u8, img_path)
            save_png(mask_u8, mask_path)

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
    # 4) Guardar manifests
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

    print("[INFO] Proceso terminado.")
    print(f"[INFO] Imágenes generadas: {len(label_rows)}")
    print(f"[INFO] Archivos con error: {len(failed_rows)}")
    print(f"[INFO] labels.csv: {LABELS_CSV}")
    print(f"[INFO] failed_files.csv: {FAILED_CSV}")
    print(f"[INFO] valid_stations_used.csv: {VALID_STATIONS_CSV}")
    print(f"[INFO] invalid_stations_excluded.csv: {INVALID_STATIONS_CSV}")


if __name__ == "__main__":
    main()