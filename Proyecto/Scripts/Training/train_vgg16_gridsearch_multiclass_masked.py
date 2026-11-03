#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tqdm import tqdm


# ============================================================
# CONFIGURACIÓN DEFAULT
# ============================================================

DEFAULT_DATASET_ROOT = Path(
    "/home/igonzalez/ecallisto/dataset_images_656_top6_pchip_gapfill_meanfreq_positive_50_870"
)

DEFAULT_RUNS_ROOT = Path(
    "/home/igonzalez/ecallisto/ecallisto-srb-ai/training_runs"
)

CLASS_NAMES = ["II", "III", "IV"]

CLASS_TO_ID = {
    "II": 0,
    "III": 1,
    "IV": 2,
}

ID_TO_CLASS = {
    0: "II",
    1: "III",
    2: "IV",
}

MINORITY_CLASS_IDS = {0, 2}  # II y IV
MAJORITY_CLASS_ID = 1        # III

SEED = 7
AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# UTILIDADES GENERALES
# ============================================================

def set_reproducibility(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
            print("[INFO] Determinismo TensorFlow activado.")
        except Exception as exc:
            print(f"[WARN] No se pudo activar determinismo TensorFlow: {exc}")


def configure_gpu_memory_growth() -> None:
    physical_devices = tf.config.list_physical_devices("GPU")

    if not physical_devices:
        print("[INFO] GPU no detectada. Se usará CPU.")
        return

    print(f"[INFO] GPU detectada: {physical_devices}")

    for gpu in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[INFO] Memory growth activado para {gpu}.")
        except Exception as exc:
            print(f"[WARN] No se pudo activar memory growth para {gpu}: {exc}")


def configure_mixed_precision(enabled: bool) -> None:
    if not enabled:
        print("[INFO] Mixed precision desactivado.")
        return

    mixed_precision.set_global_policy("mixed_float16")
    print("[INFO] Mixed precision activado: mixed_float16.")


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_class_sample_weights(text: str) -> list[float]:
    values = parse_float_list(text)

    if len(values) != 3:
        raise ValueError(
            "class-sample-weights debe tener 3 valores: peso_II,peso_III,peso_IV"
        )

    total = sum(values)

    if total <= 0:
        raise ValueError("La suma de class-sample-weights debe ser mayor que 0.")

    return [float(v / total) for v in values]


def slug_float(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_confusion_matrix_csv(
    cm_array: np.ndarray,
    path: Path,
    class_names: list[str],
) -> None:
    df = pd.DataFrame(cm_array, index=class_names, columns=class_names)
    df.index.name = "actual"
    df.columns.name = "predicted"
    ensure_dir(path.parent)
    df.to_csv(path, encoding="utf-8")


def save_confusion_matrix_png(
    cm_array: np.ndarray,
    path: Path,
    class_names: list[str],
    title: str,
) -> None:
    ensure_dir(path.parent)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_array)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            ax.text(j, i, str(cm_array[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def resolve_dataset_file_path(path_value: str, dataset_root: Path) -> str:
    """
    Convierte rutas absolutas antiguas de Windows a rutas válidas en Linux,
    usando como ancla el nombre de la carpeta del dataset.

    Ejemplo:
    C:\\Users\\Macri\\...\\dataset_images_xxx\\images\\III\\file.png

    pasa a:
    /home/igonzalez/ecallisto/dataset_images_xxx/images/III/file.png
    """
    raw = str(path_value).strip()

    if not raw or raw.lower() == "nan":
        return raw

    direct_path = Path(raw)

    if direct_path.exists():
        return str(direct_path)

    normalized = raw.replace("\\", "/")
    dataset_name = dataset_root.name
    parts = [part for part in normalized.split("/") if part]

    if dataset_name in parts:
        idx = parts.index(dataset_name)
        suffix_parts = parts[idx + 1:]
        candidate = dataset_root.joinpath(*suffix_parts)
        return str(candidate)

    # Fallback útil si el CSV contiene algo como:
    # images/III/file.png, masks/III/file.png, scientific_masks/III/file.png, etc.
    for anchor in ["images", "masks", "scientific_masks", "gapfill_masks"]:
        if anchor in parts:
            idx = parts.index(anchor)
            candidate = dataset_root.joinpath(*parts[idx:])
            return str(candidate)

    # Último recurso: interpretarlo como relativo al dataset_root.
    candidate = dataset_root / normalized
    return str(candidate)


# ============================================================
# LECTURA DEL DATASET
# ============================================================

def load_labels_dataframe(
    dataset_root: Path,
    mask_column: str,
    max_samples_per_class: int | None,
    seed: int,
) -> pd.DataFrame:
    labels_csv = dataset_root / "labels.csv"

    if not labels_csv.exists():
        raise FileNotFoundError(f"No existe labels.csv: {labels_csv}")

    df = pd.read_csv(labels_csv)

    required_cols = {
        "image_path",
        mask_column,
        "final_label",
        "source_fit",
        "station",
    }

    missing = sorted(required_cols - set(df.columns))

    if missing:
        raise ValueError(
            f"labels.csv no contiene las columnas requeridas: {missing}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    df = df.copy()
    df["final_label"] = df["final_label"].astype(str).str.strip().str.upper()
    df = df[df["final_label"].isin(CLASS_TO_ID.keys())].copy()
    df["label_id"] = df["final_label"].map(CLASS_TO_ID).astype(np.int32)

    df["image_path"] = df["image_path"].astype(str).map(
        lambda p: resolve_dataset_file_path(p, dataset_root)
    )

    df["mask_path_used"] = df[mask_column].astype(str).map(
        lambda p: resolve_dataset_file_path(p, dataset_root)
    )

    # Verificación básica de existencia.
    exists_image = df["image_path"].map(lambda p: Path(p).exists())
    exists_mask = df["mask_path_used"].map(lambda p: Path(p).exists())

    missing_count = int((~exists_image | ~exists_mask).sum())

    if missing_count > 0:
        print(f"[WARN] Se excluirán {missing_count} filas con imagen o máscara inexistente.")
        df = df[exists_image & exists_mask].copy()

    if df.empty:
        raise ValueError("No quedan muestras válidas después de verificar paths.")

    if max_samples_per_class is not None:
        sampled = []

        for class_name, sub in df.groupby("final_label"):
            n = min(len(sub), max_samples_per_class)
            sampled.append(sub.sample(n=n, random_state=seed))

        df = (
            pd.concat(sampled, axis=0)
            .sample(frac=1.0, random_state=seed)
            .reset_index(drop=True)
        )

    return df.reset_index(drop=True)


def split_train_val_test(
    df: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if val_size <= 0 or test_size <= 0:
        raise ValueError("val-size y test-size deben ser mayores que 0.")

    if val_size + test_size >= 0.8:
        raise ValueError("val-size + test-size es demasiado alto.")

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=df["label_id"],
    )

    adjusted_val_size = val_size / (1.0 - test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        random_state=seed,
        shuffle=True,
        stratify=train_val_df["label_id"],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def print_split_summary(name: str, df: pd.DataFrame) -> None:
    counts = df["final_label"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    total = int(len(df))

    print(f"[INFO] {name}: {total} muestras")

    for cls in CLASS_NAMES:
        n = int(counts.loc[cls])
        pct = 100.0 * n / max(total, 1)
        print(f"       {cls}: {n} ({pct:.2f}%)")


# ============================================================
# PIPELINE TF.DATA CON IMÁGENES Y MÁSCARAS
# ============================================================

def load_image_and_mask_tensor(
    image_path: tf.Tensor,
    mask_path: tf.Tensor,
    label: tf.Tensor,
    image_size: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(image_path)
    image = tf.image.decode_png(image_bytes, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32) * 255.0

    mask_bytes = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask_bytes, channels=1)
    mask = tf.image.convert_image_dtype(mask, dtype=tf.float32)

    image = tf.image.resize(
        image,
        size=(image_size, image_size),
        method="bilinear",
        antialias=True,
    )

    mask = tf.image.resize(
        mask,
        size=(image_size, image_size),
        method="nearest",
    )

    mask = tf.where(mask > 0.5, 1.0, 0.0)

    return image, mask, label


def apply_minority_augmentation(
    image: tf.Tensor,
    mask: tf.Tensor,
    label: tf.Tensor,
    translation_layer: keras.Model,
    gaussian_noise_std: float,
    brightness_delta: float,
    contrast_low: float,
    contrast_high: float,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Aumento controlado para clases minoritarias II y IV.

    No se usan flips ni rotaciones.
    La translación es horizontal; representa desplazamiento temporal.

    Nota:
    Con mixed precision activado, algunas capas de Keras pueden devolver float16.
    Por eso se fuerza image/mask a float32 después de la translación, para evitar
    errores de tipo durante ruido, brillo y contraste.
    """
    combined = tf.concat([image, mask], axis=-1)
    combined = translation_layer(combined, training=True)

    image_aug = combined[..., :3]
    mask_aug = combined[..., 3:4]

    # Mantener el pipeline de augmentation en float32.
    image_aug = tf.cast(image_aug, tf.float32)
    mask_aug = tf.cast(mask_aug, tf.float32)

    mask_aug = tf.where(mask_aug > 0.5, 1.0, 0.0)
    mask_rgb = tf.repeat(mask_aug, repeats=3, axis=-1)
    mask_rgb = tf.cast(mask_rgb, tf.float32)

    # Ruido gaussiano leve, solo en región válida.
    if gaussian_noise_std > 0:
        noise = tf.random.normal(
            shape=tf.shape(image_aug),
            mean=0.0,
            stddev=float(gaussian_noise_std),
            dtype=tf.float32,
        )
        image_aug = image_aug + noise * mask_rgb

    # Contraste leve.
    image_aug = tf.image.random_contrast(
        image_aug,
        lower=float(contrast_low),
        upper=float(contrast_high),
    )

    # Brillo leve.
    if brightness_delta > 0:
        delta = tf.random.uniform(
            shape=[],
            minval=-float(brightness_delta),
            maxval=float(brightness_delta),
            dtype=tf.float32,
        )
        image_aug = image_aug + delta

    image_aug = tf.clip_by_value(image_aug, 0.0, 255.0)
    image_aug = image_aug * mask_rgb

    return image_aug, mask_aug


def preprocess_for_vgg16(
    image: tf.Tensor,
    mask: tf.Tensor,
    label: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Preprocesamiento final para VGG16.

    Se aplica la máscara antes y después de preprocess_input para que
    las regiones sin cobertura permanezcan anuladas.
    """
    image = tf.cast(image, tf.float32)
    mask = tf.cast(mask, tf.float32)

    mask_rgb = tf.repeat(mask, repeats=3, axis=-1)
    mask_rgb = tf.cast(mask_rgb, tf.float32)

    image = image * mask_rgb
    image = preprocess_input(image)
    image = tf.cast(image, tf.float32)
    image = image * mask_rgb

    label = tf.cast(label, tf.int32)

    return image, label


def make_class_dataset(
    sub_df: pd.DataFrame,
    image_size: int,
    augment: bool,
    seed: int,
    gaussian_noise_std: float,
    brightness_delta: float,
    contrast_low: float,
    contrast_high: float,
    translation_width_factor: float,
) -> tf.data.Dataset:
    image_paths = sub_df["image_path"].to_numpy(dtype=str)
    mask_paths = sub_df["mask_path_used"].to_numpy(dtype=str)
    labels = sub_df["label_id"].to_numpy(dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths, labels))

    ds = ds.shuffle(
        buffer_size=max(len(sub_df), 1),
        seed=seed,
        reshuffle_each_iteration=True,
    )

    ds = ds.repeat()

    translation_layer = keras.Sequential([
        layers.RandomTranslation(
            height_factor=0.0,
            width_factor=float(translation_width_factor),
            fill_mode="constant",
            fill_value=0.0,
            seed=seed,
        )
    ])

    def _map(image_path, mask_path, label):
        image, mask, label_out = load_image_and_mask_tensor(
            image_path=image_path,
            mask_path=mask_path,
            label=label,
            image_size=image_size,
        )

        if augment:
            image, mask = apply_minority_augmentation(
                image=image,
                mask=mask,
                label=label_out,
                translation_layer=translation_layer,
                gaussian_noise_std=gaussian_noise_std,
                brightness_delta=brightness_delta,
                contrast_low=contrast_low,
                contrast_high=contrast_high,
            )

        return preprocess_for_vgg16(image, mask, label_out)

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)

    return ds


def make_balanced_train_dataset(
    train_df: pd.DataFrame,
    image_size: int,
    batch_size: int,
    class_sample_weights: list[float],
    seed: int,
    gaussian_noise_std: float,
    brightness_delta: float,
    contrast_low: float,
    contrast_high: float,
    translation_width_factor: float,
) -> tf.data.Dataset:
    class_datasets = []

    for class_id in [0, 1, 2]:
        sub = train_df[train_df["label_id"].eq(class_id)].copy()

        if sub.empty:
            raise ValueError(f"No hay muestras de clase {ID_TO_CLASS[class_id]} en train.")

        augment = class_id in MINORITY_CLASS_IDS

        ds_class = make_class_dataset(
            sub_df=sub,
            image_size=image_size,
            augment=augment,
            seed=seed + class_id,
            gaussian_noise_std=gaussian_noise_std,
            brightness_delta=brightness_delta,
            contrast_low=contrast_low,
            contrast_high=contrast_high,
            translation_width_factor=translation_width_factor,
        )

        class_datasets.append(ds_class)

    ds = tf.data.Dataset.sample_from_datasets(
        class_datasets,
        weights=class_sample_weights,
        seed=seed,
    )

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)

    return ds


def make_eval_dataset(
    df: pd.DataFrame,
    image_size: int,
    batch_size: int,
) -> tf.data.Dataset:
    image_paths = df["image_path"].to_numpy(dtype=str)
    mask_paths = df["mask_path_used"].to_numpy(dtype=str)
    labels = df["label_id"].to_numpy(dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths, labels))

    def _map(image_path, mask_path, label):
        image, mask, label_out = load_image_and_mask_tensor(
            image_path=image_path,
            mask_path=mask_path,
            label=label,
            image_size=image_size,
        )
        return preprocess_for_vgg16(image, mask, label_out)

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)

    return ds


def collect_labels_from_dataset(ds: tf.data.Dataset) -> np.ndarray:
    labels = []

    for _, y_batch in ds:
        labels.append(y_batch.numpy())

    if not labels:
        return np.array([], dtype=np.int32)

    return np.concatenate(labels).astype(np.int32)


# ============================================================
# MODELO VGG16 MULTICLASE
# ============================================================

def build_vgg16_multiclass_model(
    image_size: int,
    dense_units: int,
    dropout: float,
    learning_rate: float,
    train_base: bool,
    unfreeze_last_block: bool,
    unfreeze_from_block: str,
) -> keras.Model:
    inputs = keras.Input(shape=(image_size, image_size, 3))

    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_tensor=inputs,
    )

    if train_base:
        base_model.trainable = True
        effective_unfreeze = "all"
    else:
        base_model.trainable = False

        effective_unfreeze = str(unfreeze_from_block).strip().lower()

        # Compatibilidad con el argumento viejo.
        if unfreeze_last_block and effective_unfreeze == "none":
            effective_unfreeze = "block5"

        if effective_unfreeze != "none":
            base_model.trainable = True

            if effective_unfreeze == "block5":
                trainable_prefixes = ("block5",)
            elif effective_unfreeze == "block4":
                trainable_prefixes = ("block4", "block5")
            else:
                raise ValueError(
                    "unfreeze_from_block debe ser 'none', 'block5' o 'block4'. "
                    f"Valor recibido: {unfreeze_from_block}"
                )

            for layer in base_model.layers:
                layer.trainable = layer.name.startswith(trainable_prefixes)

    print("[INFO] Estado de entrenamiento de VGG16:")
    print(f"       train_base: {train_base}")
    print(f"       unfreeze_last_block: {unfreeze_last_block}")
    print(f"       unfreeze_from_block: {unfreeze_from_block}")
    print(f"       effective_unfreeze: {effective_unfreeze}")

    n_trainable = sum(int(layer.trainable) for layer in base_model.layers)
    n_total = len(base_model.layers)
    print(f"       capas entrenables VGG16: {n_trainable}/{n_total}")

    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(int(dense_units), activation="relu", name="dense_head")(x)
    x = layers.Dropout(float(dropout), name="dropout_head")(x)

    outputs = layers.Dense(
        3,
        activation="softmax",
        dtype="float32",
        name="class_output",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(learning_rate)),
        loss="sparse_categorical_crossentropy",
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )

    return model


# ============================================================
# CALLBACK MACRO-F1 PARA GRID SEARCH
# ============================================================

class ValidationMacroF1Callback(keras.callbacks.Callback):
    def __init__(
        self,
        val_ds: tf.data.Dataset,
        class_names: list[str],
    ) -> None:
        super().__init__()
        self.val_ds = val_ds
        self.class_names = class_names
        self.y_true = collect_labels_from_dataset(val_ds)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        y_prob = self.model.predict(self.val_ds, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)

        macro_f1 = f1_score(
            self.y_true,
            y_pred,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )

        balanced_acc = balanced_accuracy_score(
            self.y_true,
            y_pred,
        )

        logs["val_macro_f1"] = float(macro_f1)
        logs["val_balanced_accuracy"] = float(balanced_acc)

        print(
            f" - val_macro_f1: {macro_f1:.4f}"
            f" - val_balanced_accuracy: {balanced_acc:.4f}"
        )


# ============================================================
# EVALUACIÓN
# ============================================================

def evaluate_model_on_dataset(
    model: keras.Model,
    ds: tf.data.Dataset,
    class_names: list[str],
) -> tuple[dict[str, Any], np.ndarray, pd.DataFrame]:
    y_true = collect_labels_from_dataset(ds)

    y_prob = model.predict(ds, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average="weighted",
        zero_division=0,
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        zero_division=0,
    )

    cm_array = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2],
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report_dict).T

    metrics = {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),

        "precision_II": float(precision[0]),
        "precision_III": float(precision[1]),
        "precision_IV": float(precision[2]),

        "recall_II": float(recall[0]),
        "recall_III": float(recall[1]),
        "recall_IV": float(recall[2]),

        "f1_II": float(f1[0]),
        "f1_III": float(f1[1]),
        "f1_IV": float(f1[2]),

        "support_II": int(support[0]),
        "support_III": int(support[1]),
        "support_IV": int(support[2]),
    }

    return metrics, cm_array, report_df


def save_evaluation_outputs(
    output_dir: Path,
    prefix: str,
    metrics: dict[str, Any],
    cm_array: np.ndarray,
    report_df: pd.DataFrame,
) -> None:
    ensure_dir(output_dir)

    write_json(output_dir / f"{prefix}_metrics.json", metrics)

    save_confusion_matrix_csv(
        cm_array=cm_array,
        path=output_dir / f"{prefix}_confusion_matrix.csv",
        class_names=CLASS_NAMES,
    )

    save_confusion_matrix_png(
        cm_array=cm_array,
        path=output_dir / f"{prefix}_confusion_matrix.png",
        class_names=CLASS_NAMES,
        title=f"{prefix} confusion matrix",
    )

    report_df.to_csv(
        output_dir / f"{prefix}_classification_report.csv",
        encoding="utf-8",
    )


# ============================================================
# GRID SEARCH
# ============================================================

def run_one_configuration(
    config_id: int,
    total_configs: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    run_dir: Path,
    image_size: int,
    batch_size: int,
    learning_rate: float,
    dense_units: int,
    dropout: float,
    epochs: int,
    patience: int,
    steps_multiplier: float,
    class_sample_weights: list[float],
    seed: int,
    train_base: bool,
    unfreeze_last_block: bool,
    unfreeze_from_block: str,
    gaussian_noise_std: float,
    brightness_delta: float,
    contrast_low: float,
    contrast_high: float,
    translation_width_factor: float,
) -> dict[str, Any]:
    config_name = (
        f"cfg_{config_id:03d}_of_{total_configs:03d}"
        f"_is{image_size}"
        f"_bs{batch_size}"
        f"_lr{slug_float(learning_rate)}"
        f"_du{dense_units}"
        f"_do{slug_float(dropout)}"
    )

    config_dir = run_dir / "configs" / config_name
    ensure_dir(config_dir)

    print()
    print("=" * 80)
    print(f"[INFO] Configuración {config_id}/{total_configs}: {config_name}")
    print("=" * 80)

    config_data = {
        "config_id": config_id,
        "total_configs": total_configs,
        "image_size": image_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "dense_units": dense_units,
        "dropout": dropout,
        "epochs": epochs,
        "patience": patience,
        "steps_multiplier": steps_multiplier,
        "class_sample_weights_II_III_IV": class_sample_weights,
        "train_base": train_base,
        "unfreeze_last_block": unfreeze_last_block,
        "unfreeze_from_block": unfreeze_from_block,
        "gaussian_noise_std": gaussian_noise_std,
        "brightness_delta": brightness_delta,
        "contrast_low": contrast_low,
        "contrast_high": contrast_high,
        "translation_width_factor": translation_width_factor,
    }

    write_json(config_dir / "config.json", config_data)

    train_ds = make_balanced_train_dataset(
        train_df=train_df,
        image_size=image_size,
        batch_size=batch_size,
        class_sample_weights=class_sample_weights,
        seed=seed,
        gaussian_noise_std=gaussian_noise_std,
        brightness_delta=brightness_delta,
        contrast_low=contrast_low,
        contrast_high=contrast_high,
        translation_width_factor=translation_width_factor,
    )

    val_ds = make_eval_dataset(
        df=val_df,
        image_size=image_size,
        batch_size=batch_size,
    )

    test_ds = make_eval_dataset(
        df=test_df,
        image_size=image_size,
        batch_size=batch_size,
    )

    steps_per_epoch = int(
        math.ceil((len(train_df) * float(steps_multiplier)) / float(batch_size))
    )
    steps_per_epoch = max(steps_per_epoch, 1)

    model = build_vgg16_multiclass_model(
      image_size=image_size,
      dense_units=dense_units,
      dropout=dropout,
      learning_rate=learning_rate,
      train_base=train_base,
      unfreeze_last_block=unfreeze_last_block,
      unfreeze_from_block=unfreeze_from_block,
    )

    best_model_path = config_dir / "best_model.keras"
    last_model_path = config_dir / "last_model.keras"

    macro_f1_callback = ValidationMacroF1Callback(
        val_ds=val_ds,
        class_names=CLASS_NAMES,
    )

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_macro_f1",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_macro_f1",
        mode="max",
        patience=patience,
        verbose=1,
        restore_best_weights=False,
    )

    callbacks = [
        macro_f1_callback,
        checkpoint,
        early_stopping,
    ]

    start_time = time.monotonic()

    history = model.fit(
        train_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )

    end_time = time.monotonic()
    train_seconds = float(end_time - start_time)

    print(f"[INFO] Tiempo de entrenamiento: {timedelta(seconds=train_seconds)}")

    model.save(str(last_model_path))

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(config_dir / "history.csv", index=False, encoding="utf-8")

    if best_model_path.exists():
        best_model = keras.models.load_model(str(best_model_path))
    else:
        print("[WARN] No se encontró best_model.keras. Se usará el último modelo.")
        best_model = model

    val_metrics, val_cm, val_report = evaluate_model_on_dataset(
        model=best_model,
        ds=val_ds,
        class_names=CLASS_NAMES,
    )

    test_metrics, test_cm, test_report = evaluate_model_on_dataset(
        model=best_model,
        ds=test_ds,
        class_names=CLASS_NAMES,
    )

    save_evaluation_outputs(
        output_dir=config_dir,
        prefix="val",
        metrics=val_metrics,
        cm_array=val_cm,
        report_df=val_report,
    )

    save_evaluation_outputs(
        output_dir=config_dir,
        prefix="test",
        metrics=test_metrics,
        cm_array=test_cm,
        report_df=test_report,
    )

    epochs_run = int(len(history.epoch))

    if "val_macro_f1" in hist_df.columns:
        best_val_macro_f1_history = float(hist_df["val_macro_f1"].max())
    else:
        best_val_macro_f1_history = float("nan")

    result_row = {
        "config_id": config_id,
        "config_name": config_name,
        "config_dir": str(config_dir),

        "image_size": image_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "dense_units": dense_units,
        "dropout": dropout,
        "train_base": train_base,
        "unfreeze_last_block": unfreeze_last_block,
        "unfreeze_from_block": unfreeze_from_block,

        "epochs_run": epochs_run,
        "steps_per_epoch": steps_per_epoch,
        "train_seconds": train_seconds,
        "seconds_per_epoch": train_seconds / max(epochs_run, 1),

        "best_val_macro_f1_history": best_val_macro_f1_history,

        "val_accuracy": val_metrics["accuracy"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_weighted_f1": val_metrics["weighted_f1"],
        "val_recall_II": val_metrics["recall_II"],
        "val_recall_III": val_metrics["recall_III"],
        "val_recall_IV": val_metrics["recall_IV"],

        "test_accuracy": test_metrics["accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "test_precision_II": test_metrics["precision_II"],
        "test_precision_III": test_metrics["precision_III"],
        "test_precision_IV": test_metrics["precision_IV"],
        "test_recall_II": test_metrics["recall_II"],
        "test_recall_III": test_metrics["recall_III"],
        "test_recall_IV": test_metrics["recall_IV"],
        "test_f1_II": test_metrics["f1_II"],
        "test_f1_III": test_metrics["f1_III"],
        "test_f1_IV": test_metrics["f1_IV"],

        "best_model_path": str(best_model_path),
        "last_model_path": str(last_model_path),
    }

    write_json(config_dir / "result_summary.json", result_row)

    return result_row


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Grid search VGG16 multiclase II/III/IV con máscaras, "
            "balanceo por muestreo y aumento selectivo de clases minoritarias."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Carpeta raíz del dataset generado. Debe contener labels.csv.",
    )

    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help="Carpeta donde se guardarán los resultados de entrenamiento.",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="pchip_vgg16_gridsearch",
        help="Nombre del experimento. Ej: faithful_vgg16_gridsearch o pchip_vgg16_gridsearch.",
    )

    parser.add_argument(
        "--mask-column",
        type=str,
        default="mask_path",
        help=(
            "Columna de labels.csv que contiene la máscara a usar. "
            "Default: mask_path."
        ),
    )

    parser.add_argument(
        "--image-sizes",
        type=str,
        default="224",
        help="Lista separada por comas. Ej: 224,320,512",
    )

    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="8,16",
        help="Lista separada por comas. Ej: 4,8,16",
    )

    parser.add_argument(
        "--learning-rates",
        type=str,
        default="1e-5,1e-6,1e-7",
        help="Lista separada por comas. Ej: 1e-5,1e-6,1e-7",
    )

    parser.add_argument(
        "--dense-units",
        type=str,
        default="128,256",
        help="Lista separada por comas. Ej: 64,128,256",
    )

    parser.add_argument(
        "--dropouts",
        type=str,
        default="0.4,0.5",
        help="Lista separada por comas. Ej: 0.4,0.5",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Épocas máximas por configuración. Default: 200.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Paciencia de EarlyStopping sobre val_macro_f1. Default: 20.",
    )

    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Fracción total para validación. Default: 0.15.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Fracción total para test. Default: 0.15.",
    )

    parser.add_argument(
        "--steps-multiplier",
        type=float,
        default=1.0,
        help=(
            "Multiplicador de pasos por época. "
            "steps_per_epoch = ceil(len(train) * multiplier / batch_size). Default: 1.0."
        ),
    )

    parser.add_argument(
        "--class-sample-weights",
        type=str,
        default="0.333,0.333,0.333",
        help=(
            "Pesos de muestreo por clase II,III,IV para el train balanceado. "
            "Default: 0.333,0.333,0.333."
        ),
    )

    parser.add_argument(
        "--gaussian-noise-std",
        type=float,
        default=3.0,
        help="Desviación estándar del ruido gaussiano para II/IV. Default: 3.0.",
    )

    parser.add_argument(
        "--brightness-delta",
        type=float,
        default=8.0,
        help="Variación máxima de brillo para II/IV en escala 0-255. Default: 8.",
    )

    parser.add_argument(
        "--contrast-low",
        type=float,
        default=0.9,
        help="Factor mínimo de contraste para II/IV. Default: 0.9.",
    )

    parser.add_argument(
        "--contrast-high",
        type=float,
        default=1.1,
        help="Factor máximo de contraste para II/IV. Default: 1.1.",
    )

    parser.add_argument(
        "--translation-width-factor",
        type=float,
        default=0.10,
        help="Translación horizontal máxima para II/IV. Default: 0.10.",
    )

    parser.add_argument(
        "--train-base",
        action="store_true",
        help="Si se activa, entrena todo VGG16 desde el inicio. No recomendado inicialmente.",
    )

    parser.add_argument(
        "--unfreeze-last-block",
        action="store_true",
        help="Si se activa, congela VGG16 excepto block5. Ignorado si --train-base está activo.",
    )
    
    parser.add_argument(
      "--unfreeze-from-block",
      type=str,
      default="none",
      choices=["none", "block5", "block4"],
      help=(
          "Permite descongelar bloques superiores de VGG16. "
          "'none' congela toda la base; 'block5' entrena solo block5; "
          "'block4' entrena block4 y block5. "
          "Si se usa --train-base, este argumento se ignora."
          ),
    )

    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Para pruebas rápidas. Ej: 100 procesa máximo 100 muestras por clase.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Semilla global. Default: 7.",
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Activa determinismo TensorFlow. Puede hacer más lento el entrenamiento.",
    )

    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Activa mixed_float16. Recomendado para RTX 3090.",
    )

    args = parser.parse_args()
    
    unfreeze_from_block = str(args.unfreeze_from_block).strip().lower()

    if bool(args.unfreeze_last_block) and unfreeze_from_block == "none":
      unfreeze_from_block = "block5"

    dataset_root: Path = args.dataset_root
    runs_root: Path = args.runs_root
    experiment_name: str = args.experiment_name

    image_sizes = parse_int_list(args.image_sizes)
    batch_sizes = parse_int_list(args.batch_sizes)
    learning_rates = parse_float_list(args.learning_rates)
    dense_units_list = parse_int_list(args.dense_units)
    dropout_list = parse_float_list(args.dropouts)
    class_sample_weights = parse_class_sample_weights(args.class_sample_weights)

    set_reproducibility(seed=args.seed, deterministic=bool(args.deterministic))
    configure_gpu_memory_growth()
    configure_mixed_precision(enabled=bool(args.mixed_precision))

    run_dir = runs_root / experiment_name
    ensure_dir(run_dir)

    print()
    print("[INFO] Entrenamiento VGG16 multiclase con máscaras")
    print(f"[INFO] TensorFlow version: {tf.__version__}")
    print(f"[INFO] Dataset root: {dataset_root}")
    print(f"[INFO] Runs root: {runs_root}")
    print(f"[INFO] Experiment name: {experiment_name}")
    print(f"[INFO] Run dir: {run_dir}")
    print(f"[INFO] Mask column: {args.mask_column}")
    print(f"[INFO] Clases: {CLASS_NAMES}")
    print(f"[INFO] Class sample weights II/III/IV: {class_sample_weights}")
    print(f"[INFO] Mixed precision: {bool(args.mixed_precision)}")
    print()

    df = load_labels_dataframe(
        dataset_root=dataset_root,
        mask_column=args.mask_column,
        max_samples_per_class=args.max_samples_per_class,
        seed=args.seed,
    )

    print("[INFO] Dataset cargado:")
    print(df["final_label"].value_counts().reindex(CLASS_NAMES, fill_value=0).to_string())
    print()

    train_df, val_df, test_df = split_train_val_test(
        df=df,
        val_size=float(args.val_size),
        test_size=float(args.test_size),
        seed=int(args.seed),
    )

    print_split_summary("Train", train_df)
    print_split_summary("Validation", val_df)
    print_split_summary("Test", test_df)

    train_df.to_csv(run_dir / "split_train.csv", index=False, encoding="utf-8")
    val_df.to_csv(run_dir / "split_val.csv", index=False, encoding="utf-8")
    test_df.to_csv(run_dir / "split_test.csv", index=False, encoding="utf-8")

    experiment_config = {
        "dataset_root": str(dataset_root),
        "runs_root": str(runs_root),
        "experiment_name": experiment_name,
        "mask_column": args.mask_column,
        "image_sizes": image_sizes,
        "batch_sizes": batch_sizes,
        "learning_rates": learning_rates,
        "dense_units": dense_units_list,
        "dropouts": dropout_list,
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "val_size": float(args.val_size),
        "test_size": float(args.test_size),
        "steps_multiplier": float(args.steps_multiplier),
        "class_sample_weights_II_III_IV": class_sample_weights,
        "gaussian_noise_std": float(args.gaussian_noise_std),
        "brightness_delta": float(args.brightness_delta),
        "contrast_low": float(args.contrast_low),
        "contrast_high": float(args.contrast_high),
        "translation_width_factor": float(args.translation_width_factor),
        "train_base": bool(args.train_base),
        "unfreeze_last_block": bool(args.unfreeze_last_block),
        "unfreeze_from_block": unfreeze_from_block,
        "max_samples_per_class": args.max_samples_per_class,
        "seed": int(args.seed),
        "deterministic": bool(args.deterministic),
        "mixed_precision": bool(args.mixed_precision),
        "n_total": int(len(df)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
    }

    write_json(run_dir / "experiment_config.json", experiment_config)

    configs = list(itertools.product(
        image_sizes,
        batch_sizes,
        learning_rates,
        dense_units_list,
        dropout_list,
    ))

    print()
    print(f"[INFO] Total de configuraciones de grid search: {len(configs)}")
    print()

    grid_rows: list[dict[str, Any]] = []
    grid_csv = run_dir / "grid_search_results.csv"

    for idx, (image_size, batch_size, learning_rate, dense_units, dropout) in enumerate(
        tqdm(configs, desc="Grid search", unit="config"),
        start=1,
    ):
        result_row = run_one_configuration(
            config_id=idx,
            total_configs=len(configs),
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            run_dir=run_dir,
            image_size=int(image_size),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            dense_units=int(dense_units),
            dropout=float(dropout),
            epochs=int(args.epochs),
            patience=int(args.patience),
            steps_multiplier=float(args.steps_multiplier),
            class_sample_weights=class_sample_weights,
            seed=int(args.seed),
            train_base=bool(args.train_base),
            unfreeze_last_block=bool(args.unfreeze_last_block),
            unfreeze_from_block=unfreeze_from_block,
            gaussian_noise_std=float(args.gaussian_noise_std),
            brightness_delta=float(args.brightness_delta),
            contrast_low=float(args.contrast_low),
            contrast_high=float(args.contrast_high),
            translation_width_factor=float(args.translation_width_factor),
        )

        grid_rows.append(result_row)

        grid_df = pd.DataFrame(grid_rows)
        grid_df = grid_df.sort_values(
            ["val_macro_f1", "val_balanced_accuracy", "val_recall_II", "val_recall_IV"],
            ascending=False,
        )
        grid_df.to_csv(grid_csv, index=False, encoding="utf-8")

        best_so_far = grid_df.iloc[0]
        print()
        print("[INFO] Mejor configuración hasta ahora:")
        print(best_so_far[[
            "config_name",
            "val_macro_f1",
            "val_balanced_accuracy",
            "val_recall_II",
            "val_recall_IV",
            "test_macro_f1",
            "test_balanced_accuracy",
            "test_recall_II",
            "test_recall_IV",
        ]].to_string())
        print()

    final_grid_df = pd.DataFrame(grid_rows)

    if not final_grid_df.empty:
        final_grid_df = final_grid_df.sort_values(
            ["val_macro_f1", "val_balanced_accuracy", "val_recall_II", "val_recall_IV"],
            ascending=False,
        )
        final_grid_df.to_csv(grid_csv, index=False, encoding="utf-8")

        best_row = final_grid_df.iloc[0].to_dict()
        write_json(run_dir / "best_grid_config.json", best_row)

        print()
        print("=" * 80)
        print("[INFO] GRID SEARCH TERMINADO")
        print("=" * 80)
        print("[INFO] Mejor configuración:")
        print(pd.Series(best_row)[[
            "config_name",
            "image_size",
            "batch_size",
            "learning_rate",
            "dense_units",
            "dropout",
            "val_macro_f1",
            "val_balanced_accuracy",
            "val_recall_II",
            "val_recall_IV",
            "test_macro_f1",
            "test_balanced_accuracy",
            "test_recall_II",
            "test_recall_III",
            "test_recall_IV",
            "best_model_path",
        ]].to_string())
        print()
        print(f"[INFO] Resultados completos: {grid_csv}")
        print(f"[INFO] Mejor configuración JSON: {run_dir / 'best_grid_config.json'}")
    else:
        print("[WARN] No se generaron resultados de grid search.")


if __name__ == "__main__":
    main()