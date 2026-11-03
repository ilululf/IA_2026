#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

LABEL_ALIASES = {
    "NO_SRB":"NO_SRB","NO SRB":"NO_SRB","NO-SRB":"NO_SRB","NOSRB":"NO_SRB",
    "NON_SRB":"NO_SRB","NON SRB":"NO_SRB","NON-SRB":"NO_SRB",
    "NO_BURST":"NO_SRB","NO BURST":"NO_SRB",
    "II":"II","TYPE II":"II","TYPE_II":"II","TYPE-II":"II",
    "III":"III","TYPE III":"III","TYPE_III":"III","TYPE-III":"III",
    "IV":"IV","TYPE IV":"IV","TYPE_IV":"IV","TYPE-IV":"IV",
}
SUPPORTED_LABELS = ["NO_SRB","II","III","IV"]

def canonicalize_label(value: Any) -> str:
    label = " ".join(str(value).strip().upper().split())
    return LABEL_ALIASES.get(label, label)

def resolve_dataset_file_path(path_value: Any, dataset_root: Path) -> Path:
    raw = str(path_value).strip()
    if not raw or raw.lower() == "nan":
        return Path("")
    direct = Path(raw)
    if direct.exists():
        return direct.resolve()
    normalized = raw.replace("\\", "/")
    parts = [p for p in normalized.split("/") if p]
    dataset_name = dataset_root.name
    if dataset_name in parts:
        idx = parts.index(dataset_name)
        return dataset_root.joinpath(*parts[idx+1:]).resolve()
    for anchor in ["images","masks","scientific_masks","gapfill_masks"]:
        if anchor in parts:
            idx = parts.index(anchor)
            return dataset_root.joinpath(*parts[idx:]).resolve()
    return (dataset_root / normalized).resolve()

def fair_allocation(capacities: dict[str,int], target_total: int, station_order: list[str]) -> dict[str,int]:
    if sum(capacities.values()) < target_total:
        raise ValueError(f"Disponibles={sum(capacities.values())}, requeridas={target_total}.")
    allocation = {s:0 for s in station_order}
    remaining = target_total
    while remaining > 0:
        active = [s for s in station_order if allocation[s] < capacities[s]]
        if not active:
            raise RuntimeError("No queda capacidad para completar la asignación.")
        base, extra = divmod(remaining, len(active))
        added = 0
        if base == 0:
            for s in active[:extra]:
                allocation[s] += 1
                added += 1
        else:
            for i, s in enumerate(active):
                desired = base + (1 if i < extra else 0)
                room = capacities[s] - allocation[s]
                take = min(desired, room)
                allocation[s] += take
                added += take
        if added == 0:
            raise RuntimeError("No se pudo avanzar en la redistribución.")
        remaining -= added
    return allocation

def unique_destination(output_root: Path, kind: str, label: str, source: Path, used: dict[Path,Path]) -> Path:
    dest_dir = output_root / kind / label
    dest = dest_dir / source.name
    prev = used.get(dest)
    if prev is None or prev == source:
        used[dest] = source
        return dest
    digest = hashlib.sha1(str(source).encode()).hexdigest()[:10]
    dest = dest_dir / f"{source.stem}_{digest}{source.suffix}"
    used[dest] = source
    return dest

def main() -> None:
    p = argparse.ArgumentParser(description="Dataset para Modelo 1: SRB vs NO_SRB")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--per-srb-type", type=int, default=456)
    p.add_argument("--no-srb-count", type=str, default="all")
    p.add_argument("--mask-column", type=str, default="mask_path")
    p.add_argument("--station-column", type=str, default="station")
    p.add_argument("--expected-stations", type=int, default=6)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    labels_path = dataset_root / "labels.csv"

    if not labels_path.exists():
        raise FileNotFoundError(labels_path)
    if output_root == dataset_root:
        raise ValueError("output-root no puede ser igual al dataset original.")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"La carpeta de salida ya existe y no está vacía: {output_root}")

    df = pd.read_csv(labels_path)
    required = {"image_path", args.mask_column, "final_label", args.station_column}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")

    df = df.copy()
    df["_source_row_index"] = df.index
    df["canonical_label"] = df["final_label"].map(canonicalize_label)
    df["_station_clean"] = df[args.station_column].astype(str).str.strip()
    df = df[df["canonical_label"].isin(SUPPORTED_LABELS)].copy()
    df = df[df["_station_clean"].ne("") & df["_station_clean"].str.lower().ne("nan")].copy()

    df["_resolved_image"] = df["image_path"].map(lambda x: resolve_dataset_file_path(x, dataset_root))
    df["_resolved_mask"] = df[args.mask_column].map(lambda x: resolve_dataset_file_path(x, dataset_root))
    valid = df["_resolved_image"].map(lambda x: x.is_file()) & df["_resolved_mask"].map(lambda x: x.is_file())
    invalid_count = int((~valid).sum())
    if invalid_count:
        print(f"[WARN] Se excluyen {invalid_count} filas porque falta imagen o máscara.")
    df = df[valid].copy()

    stations = sorted(df["_station_clean"].unique().tolist())
    if len(stations) != args.expected_stations:
        raise ValueError(f"Se esperaban {args.expected_stations} estaciones; se encontraron {len(stations)}: {stations}")

    available = pd.crosstab(df["canonical_label"], df["_station_clean"]).reindex(index=SUPPORTED_LABELS, columns=stations, fill_value=0)
    print("\n[INFO] Disponibilidad por clase y estación:")
    print(available.to_string())

    totals = df["canonical_label"].value_counts().reindex(SUPPORTED_LABELS, fill_value=0)
    no_srb_available = int(totals["NO_SRB"])
    no_srb_target = no_srb_available if args.no_srb_count.lower() == "all" else int(args.no_srb_count)
    if no_srb_target > no_srb_available:
        raise ValueError(f"NO_SRB solicitadas={no_srb_target}, disponibles={no_srb_available}")

    targets = {"NO_SRB": no_srb_target, "II": args.per_srb_type, "III": args.per_srb_type, "IV": args.per_srb_type}
    allocations = {}
    for label in SUPPORTED_LABELS:
        caps = {s:int(available.loc[label,s]) for s in stations}
        target = int(targets[label])
        allocations[label] = caps.copy() if target == sum(caps.values()) else fair_allocation(caps, target, stations)

    allocation_df = pd.DataFrame.from_dict(allocations, orient="index").reindex(index=SUPPORTED_LABELS, columns=stations, fill_value=0)
    allocation_df.index.name = "final_label"
    print("\n[INFO] Asignación seleccionada:")
    print(allocation_df.to_string())

    pieces = []
    for li, label in enumerate(SUPPORTED_LABELS):
        for si, station in enumerate(stations):
            n = int(allocation_df.loc[label, station])
            if n == 0:
                continue
            sub = df[(df["canonical_label"] == label) & (df["_station_clean"] == station)]
            sampled = sub.copy() if n == len(sub) else sub.sample(n=n, replace=False, random_state=args.seed + li*1000 + si)
            pieces.append(sampled)

    selected = pd.concat(pieces, ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    selected["model1_label"] = selected["canonical_label"].map(lambda x: "NO_SRB" if x == "NO_SRB" else "SRB")

    output_root.mkdir(parents=True, exist_ok=True)
    for label in SUPPORTED_LABELS:
        (output_root/"images"/label).mkdir(parents=True, exist_ok=True)
        (output_root/"masks"/label).mkdir(parents=True, exist_ok=True)

    used_img, used_mask = {}, {}
    rows, manifest = [], []
    for i, (_, row) in enumerate(selected.iterrows(), start=1):
        label = row["canonical_label"]
        station = row["_station_clean"]
        src_img, src_mask = row["_resolved_image"], row["_resolved_mask"]
        dst_img = unique_destination(output_root, "images", label, src_img, used_img)
        dst_mask = unique_destination(output_root, "masks", label, src_mask, used_mask)
        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_mask, dst_mask)

        out = row.drop(labels=["_resolved_image","_resolved_mask","canonical_label","_station_clean"]).to_dict()
        out["image_path"] = dst_img.relative_to(output_root).as_posix()
        out[args.mask_column] = dst_mask.relative_to(output_root).as_posix()
        out["final_label"] = label
        out["model1_label"] = "NO_SRB" if label == "NO_SRB" else "SRB"
        rows.append(out)

        manifest.append({
            "source_row_index": int(row["_source_row_index"]),
            "final_label": label,
            "model1_label": out["model1_label"],
            "station": station,
            "source_image": str(src_img),
            "balanced_image": out["image_path"],
            "source_mask": str(src_mask),
            "balanced_mask": out[args.mask_column],
        })
        if i % 500 == 0 or i == len(selected):
            print(f"[INFO] Copiados {i}/{len(selected)}")

    out_df = pd.DataFrame(rows)
    if "_source_row_index" in out_df.columns:
        out_df = out_df.drop(columns=["_source_row_index"])
    out_df.to_csv(output_root/"labels.csv", index=False)
    pd.DataFrame(manifest).to_csv(output_root/"selection_manifest.csv", index=False)

    class_station = pd.crosstab(selected["canonical_label"], selected["_station_clean"]).reindex(index=SUPPORTED_LABELS, columns=stations, fill_value=0)
    class_station.to_csv(output_root/"class_station_counts.csv")

    binary_counts = selected["model1_label"].value_counts().reindex(["NO_SRB","SRB"], fill_value=0)
    pd.DataFrame({"model1_label":["NO_SRB","SRB"],"count":[int(binary_counts["NO_SRB"]),int(binary_counts["SRB"])]}).to_csv(output_root/"binary_summary.csv", index=False)

    print("\n[OK] Dataset creado:", output_root)
    print("\n[INFO] Conteos por clase original:")
    print(selected["canonical_label"].value_counts().reindex(SUPPORTED_LABELS, fill_value=0).to_string())
    print("\n[INFO] Conteos binarios:")
    print(binary_counts.to_string())
    print("\n[INFO] Clase x estación:")
    print(class_station.to_string())
    print("\n[INFO] El data augmentation de NO_SRB debe aplicarse SOLO en TRAIN, después del split.")

if __name__ == "__main__":
    main()
