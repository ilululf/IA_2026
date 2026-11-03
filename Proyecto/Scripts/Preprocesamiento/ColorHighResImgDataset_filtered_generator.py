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
from scipy.ndimage import median_filter
from tqdm import tqdm


# =========================================================
# CONFIG
# =========================================================

RAW_DATASET_ROOT = Path(r"C:\Users\Macri\Desktop\Facultad\Tesis\dataset")

# Archivo local ya generado con estaciones válidas
VALID_STATIONS_INPUT_CSV = Path(
    r"C:\Users\Macri\Desktop\Facultad\Tesis\dataset_images_768_validstations\valid_stations_used.csv"
)

OUTPUT_ROOT = Path(r"C:\Users\Macri\Desktop\Facultad\Tesis\dataset_images_768_rfi_cleaned")

# Grilla fija
TARGET_FREQ_MIN_MHZ = 45.0
TARGET_FREQ_MAX_MHZ = 870.0
IMG_H = 768
IMG_W = 768

FIT_EXTENSIONS = {".fit", ".fits", ".fit.gz", ".fits.gz"}

# Parámetros de limpieza RFI
RFI_MEDIAN_WINDOW = 31       # mediana móvil sobre frecuencia
RFI_Z_THRESHOLD = 6.0        # umbral robusto
RFI_DILATE_ROWS = 1          # expande máscara ±1 fila

LABELS_CSV = OUTPUT_ROOT / "labels.csv"
FAILED_CSV = OUTPUT_ROOT / "failed_files.csv"
USED_VALID_STATIONS_CSV = OUTPUT_ROOT / "used_valid_stations_copy.csv"
RFI_SUMMARY_CSV = OUTPUT_ROOT / "rfi_cleaning_summary.csv"


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
class RFISummaryRecord:
    source_fit: str
    station: str
    final_label: str
    raw_shape: str
    freq_min_mhz: float
    freq_max_mhz: float
    n_flagged_channels: int
    frac_flagged_channels: float


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
    """
    Prioridad principal: columna real de frecuencias en la BinTable.
    """
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
    """
    Devuelve:
    - data orientada como [freq, time]
    - eje de frecuencias en MHz
    """
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

    # heurística final
    if data.shape[0] <= data.shape[1]:
        freq_guess = np.linspace(TARGET_FREQ_MIN_MHZ, TARGET_FREQ_MAX_MHZ, data.shape[0], dtype=np.float64)
        return data, freq_guess
    else:
        freq_guess = np.linspace(TARGET_FREQ_MIN_MHZ, TARGET_FREQ_MAX_MHZ, data.shape[1], dtype=np.float64)
        return data.T, freq_guess


# =========================================================
# PREPROCESSING
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


def subtract_background_median(data_ft: np.ndarray) -> np.ndarray:
    """
    Método base de Iván:
    resta la mediana temporal por frecuencia.
    """
    med_col = np.median(data_ft, axis=1, keepdims=True)
    return data_ft - med_col


def robust_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(mad)


def detect_persistent_rfi_rows(data_ft: np.ndarray) -> np.ndarray:
    """
    Detecta filas (frecuencias) persistentemente contaminadas.
    """
    row_energy = np.median(np.abs(data_ft), axis=1)

    local_baseline = median_filter(row_energy, size=RFI_MEDIAN_WINDOW, mode="nearest")
    residual = row_energy - local_baseline

    mad = robust_mad(residual)
    if not np.isfinite(mad) or mad <= 0:
        return np.zeros(data_ft.shape[0], dtype=bool)

    robust_z = residual / (1.4826 * mad)
    flagged = robust_z > RFI_Z_THRESHOLD

    if RFI_DILATE_ROWS > 0 and np.any(flagged):
        dilated = flagged.copy()
        idx = np.where(flagged)[0]
        for i in idx:
            lo = max(0, i - RFI_DILATE_ROWS)
            hi = min(flagged.size, i + RFI_DILATE_ROWS + 1)
            dilated[lo:hi] = True
        flagged = dilated

    return flagged


def interpolate_bad_rows(data_ft: np.ndarray, bad_rows: np.ndarray) -> np.ndarray:
    """
    Reemplaza filas contaminadas por interpolación lineal usando frecuencias vecinas sanas.
    """
    out = np.asarray(data_ft, dtype=np.float64).copy()
    n_rows, n_cols = out.shape

    good_idx = np.where(~bad_rows)[0]
    bad_idx = np.where(bad_rows)[0]

    if good_idx.size < 2:
        return out

    for t in range(n_cols):
        col = out[:, t]
        out[bad_idx, t] = np.interp(bad_idx.astype(np.float64), good_idx.astype(np.float64), col[good_idx])

    return out


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


def robust_percentile_normalize(image_ft: np.ndarray, coverage_mask_ft: np.ndarray) -> np.ndarray:
    valid = coverage_mask_ft > 0
    vals = image_ft[valid]

    if vals.size == 0:
        return np.zeros_like(image_ft, dtype=np.float64)

    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros_like(image_ft, dtype=np.float64)

    p1 = np.percentile(vals, 1)
    p99 = np.percentile(vals, 99)

    if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
        mn = np.min(vals)
        mx = np.max(vals)
        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
            return np.zeros_like(image_ft, dtype=np.float64)
        p1, p99 = mn, mx

    out = (image_ft - p1) / (p99 - p1)
    out = np.clip(out, 0.0, 1.0)
    out[~valid] = 0.0
    return out


def apply_jet_colormap(img01: np.ndarray, coverage_mask_ft: np.ndarray) -> np.ndarray:
    rgba = cm.jet(img01)
    rgb = (rgba[:, :, :3] * 255.0).astype(np.uint8)

    invalid = coverage_mask_ft == 0
    rgb[invalid] = 0
    return rgb


def process_fit_to_image_and_mask(path: Path) -> tuple[np.ndarray, np.ndarray, int, float, float]:
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

    # 1) fondo base estilo Iván
    data_ft = subtract_background_median(data_ft)

    # 2) detectar y limpiar canales RFI persistentes
    bad_rows = detect_persistent_rfi_rows(data_ft)
    n_flagged = int(np.count_nonzero(bad_rows))
    data_ft = interpolate_bad_rows(data_ft, bad_rows)

    # 3) grilla fija total
    dst_freqs = np.linspace(TARGET_FREQ_MIN_MHZ, TARGET_FREQ_MAX_MHZ, IMG_H, dtype=np.float64)

    # 4) interpolación en frecuencia
    interp_ft, freq_row_mask = interpolate_frequency_axis(data_ft, freqs_mhz, dst_freqs)

    # 5) remuestreo temporal
    interp_ft = resize_time_axis(interp_ft, IMG_W)

    # 6) máscara
    coverage_mask_ft = build_coverage_mask(freq_row_mask, IMG_W)

    # 7) normalización robusta
    norm_ft = robust_percentile_normalize(interp_ft, coverage_mask_ft)

    # 8) jet al final
    rgb_u8 = apply_jet_colormap(norm_ft, coverage_mask_ft)
    mask_u8 = (coverage_mask_ft * 255).astype(np.uint8)

    # altas arriba, bajas abajo
    rgb_u8 = np.flipud(rgb_u8)
    mask_u8 = np.flipud(mask_u8)

    return rgb_u8, mask_u8, n_flagged, freq_min, freq_max


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
    rfi_rows: list[dict] = []

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
    # 2) Generar imágenes y máscaras
    # -----------------------------------------------------
    for fit_path in tqdm(valid_fit_files, desc="Procesando FITS", unit="fit"):
        try:
            original_type_folder, station, year = infer_dataset_metadata(RAW_DATASET_ROOT, fit_path)
            final_label = map_to_final_label(original_type_folder)

            if final_label is None:
                continue

            image_rgb_u8, mask_u8, n_flagged, freq_min, freq_max = process_fit_to_image_and_mask(fit_path)

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

            rfi_rows.append(asdict(RFISummaryRecord(
                source_fit=str(fit_path),
                station=station,
                final_label=final_label,
                raw_shape="",
                freq_min_mhz=freq_min,
                freq_max_mhz=freq_max,
                n_flagged_channels=n_flagged,
                frac_flagged_channels=(n_flagged / max(1, IMG_H)),
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

    if rfi_rows:
        with RFI_SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rfi_rows[0].keys()))
            writer.writeheader()
            writer.writerows(rfi_rows)

    print("[INFO] Proceso terminado.")
    print(f"[INFO] Imágenes generadas: {len(label_rows)}")
    print(f"[INFO] Archivos con error: {len(failed_rows)}")
    print(f"[INFO] labels.csv: {LABELS_CSV}")
    print(f"[INFO] failed_files.csv: {FAILED_CSV}")
    print(f"[INFO] used_valid_stations_copy.csv: {USED_VALID_STATIONS_CSV}")
    print(f"[INFO] rfi_cleaning_summary.csv: {RFI_SUMMARY_CSV}")


if __name__ == "__main__":
    main()