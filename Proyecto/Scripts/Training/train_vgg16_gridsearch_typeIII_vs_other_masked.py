#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import shutil
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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
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

# Se mantienen las mismas rutas del entrenamiento multiclase actual.
DEFAULT_DATASET_ROOT = Path(
    "/home/igonzalez/ecallisto/dataset_images_656_top6_pchip_gapfill_meanfreq_positive_50_870"
)

DEFAULT_RUNS_ROOT = Path(
    "/home/igonzalez/ecallisto/ecallisto-srb-ai/training_runs"
)

# Modelo 2 de la cascada:
#   TYPE III      -> 1 (clase positiva)
#   OTHER TYPES   -> 0 (en este dataset: II + IV)
CLASS_NAMES = ["OTHER_TYPES", "TYPE_III"]
CLASS_TO_ID = {
    "II": 0,
    "III": 1,
    "IV": 0,
}
ID_TO_CLASS = {
    0: "OTHER_TYPES",
    1: "TYPE_III",
}

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


def parse_binary_sample_weights(text: str) -> list[float]:
    values = parse_float_list(text)

    if len(values) != 2:
        raise ValueError(
            "class-sample-weights debe tener 2 valores: peso_OTHER_TYPES,peso_TYPE_III"
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
    """Convierte rutas antiguas/relativas a rutas válidas dentro del dataset."""
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
        return str(dataset_root.joinpath(*parts[idx + 1:]))

    for anchor in ["images", "masks", "scientific_masks", "gapfill_masks"]:
        if anchor in parts:
            idx = parts.index(anchor)
            return str(dataset_root.joinpath(*parts[idx:]))

    return str(dataset_root / normalized)


# ============================================================
# LECTURA DEL DATASET Y ETIQUETADO BINARIO
# ============================================================

def load_labels_dataframe(
    dataset_root: Path,
    mask_column: str,
    max_samples_per_binary_class: int | None,
    seed: int,
) -> pd.DataFrame:
    """
    Reutiliza labels.csv del dataset II/III/IV.

    Etiquetado para el segundo modelo de la cascada:
      III       -> TYPE_III     -> label_id = 1
      II + IV   -> OTHER_TYPES  -> label_id = 0

    Se preserva también original_label para poder auditar qué parte de
    OTHER_TYPES corresponde a II y cuál a IV.
    """
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

    df["original_label"] = df["final_label"]
    df["label_id"] = df["final_label"].map(CLASS_TO_ID).astype(np.int32)
    df["binary_label"] = df["label_id"].map(ID_TO_CLASS)

    df["image_path"] = df["image_path"].astype(str).map(
        lambda p: resolve_dataset_file_path(p, dataset_root)
    )
    df["mask_path_used"] = df[mask_column].astype(str).map(
        lambda p: resolve_dataset_file_path(p, dataset_root)
    )

    exists_image = df["image_path"].map(lambda p: Path(p).exists())
    exists_mask = df["mask_path_used"].map(lambda p: Path(p).exists())
    missing_count = int((~exists_image | ~exists_mask).sum())

    if missing_count > 0:
        print(f"[WARN] Se excluirán {missing_count} filas con imagen o máscara inexistente.")
        df = df[exists_image & exists_mask].copy()

    if df.empty:
        raise ValueError("No quedan muestras válidas después de verificar paths.")

    # El límite se aplica DESPUÉS de agrupar II+IV como OTHER_TYPES, para que
    # una prueba rápida mantenga el problema binario balanceable.
    if max_samples_per_binary_class is not None:
        sampled = []
        for label_id, sub in df.groupby("label_id"):
            n = min(len(sub), max_samples_per_binary_class)
            sampled.append(sub.sample(n=n, random_state=seed + int(label_id)))

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
    counts = df["binary_label"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    total = int(len(df))

    print(f"[INFO] {name}: {total} muestras")
    for cls in CLASS_NAMES:
        n = int(counts.loc[cls])
        pct = 100.0 * n / max(total, 1)
        print(f"       {cls}: {n} ({pct:.2f}%)")

    original = df["original_label"].value_counts().reindex(["II", "III", "IV"], fill_value=0)
    print(
        "       composición original: "
        f"II={int(original['II'])}, III={int(original['III'])}, IV={int(original['IV'])}"
    )


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


def apply_augmentation(
    image: tf.Tensor,
    mask: tf.Tensor,
    translation_layer: keras.Model,
    gaussian_noise_std: float,
    brightness_delta: float,
    contrast_low: float,
    contrast_high: float,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Aumento compatible con la morfología del espectrograma.

    Conserva la idea del pipeline multiclase: translación horizontal, ruido,
    contraste y brillo; no usa flips ni rotaciones.
    """
    combined = tf.concat([image, mask], axis=-1)
    combined = translation_layer(combined, training=True)

    image_aug = tf.cast(combined[..., :3], tf.float32)
    mask_aug = tf.cast(combined[..., 3:4], tf.float32)
    mask_aug = tf.where(mask_aug > 0.5, 1.0, 0.0)
    mask_rgb = tf.repeat(mask_aug, repeats=3, axis=-1)

    if gaussian_noise_std > 0:
        noise = tf.random.normal(
            shape=tf.shape(image_aug),
            mean=0.0,
            stddev=float(gaussian_noise_std),
            dtype=tf.float32,
        )
        image_aug = image_aug + noise * mask_rgb

    image_aug = tf.image.random_contrast(
        image_aug,
        lower=float(contrast_low),
        upper=float(contrast_high),
    )

    if brightness_delta > 0:
        delta = tf.random.uniform(
            shape=[],
            minval=-float(brightness_delta),
            maxval=float(brightness_delta),
            dtype=tf.float32,
        )
        image_aug = image_aug + delta

    image_aug = tf.clip_by_value(image_aug, 0.0, 255.0)
    image_aug = image_aug * tf.cast(mask_rgb, tf.float32)
    return image_aug, mask_aug


def preprocess_for_vgg16(
    image: tf.Tensor,
    mask: tf.Tensor,
    label: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Aplica máscara + preprocess_input de VGG16 y devuelve etiqueta binaria."""
    image = tf.cast(image, tf.float32)
    mask = tf.cast(mask, tf.float32)
    mask_rgb = tf.repeat(mask, repeats=3, axis=-1)

    image = image * mask_rgb
    image = preprocess_input(image)
    image = tf.cast(image, tf.float32)
    image = image * mask_rgb

    label = tf.cast(label, tf.float32)
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
    labels = sub_df["label_id"].to_numpy(dtype=np.float32)

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
            image, mask = apply_augmentation(
                image=image,
                mask=mask,
                translation_layer=translation_layer,
                gaussian_noise_std=gaussian_noise_std,
                brightness_delta=brightness_delta,
                contrast_low=contrast_low,
                contrast_high=contrast_high,
            )

        return preprocess_for_vgg16(image, mask, label_out)

    return ds.map(_map, num_parallel_calls=AUTOTUNE)


def make_balanced_train_dataset(
    train_df: pd.DataFrame,
    image_size: int,
    batch_size: int,
    class_sample_weights: list[float],
    augment_other_types: bool,
    augment_type_iii: bool,
    seed: int,
    gaussian_noise_std: float,
    brightness_delta: float,
    contrast_low: float,
    contrast_high: float,
    translation_width_factor: float,
) -> tf.data.Dataset:
    class_datasets = []

    for class_id in [0, 1]:
        sub = train_df[train_df["label_id"].eq(class_id)].copy()
        if sub.empty:
            raise ValueError(f"No hay muestras de {ID_TO_CLASS[class_id]} en train.")

        augment = augment_other_types if class_id == 0 else augment_type_iii

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
    return ds.prefetch(AUTOTUNE)


def make_eval_dataset(
    df: pd.DataFrame,
    image_size: int,
    batch_size: int,
) -> tf.data.Dataset:
    image_paths = df["image_path"].to_numpy(dtype=str)
    mask_paths = df["mask_path_used"].to_numpy(dtype=str)
    labels = df["label_id"].to_numpy(dtype=np.float32)

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
    return ds.prefetch(AUTOTUNE)


def collect_labels_from_dataset(ds: tf.data.Dataset) -> np.ndarray:
    labels = []
    for _, y_batch in ds:
        labels.append(y_batch.numpy())

    if not labels:
        return np.array([], dtype=np.int32)

    return np.concatenate(labels).astype(np.int32)


# ============================================================
# MODELO VGG16 BINARIO: TYPE III vs OTHER TYPES
# ============================================================

def build_vgg16_binary_model(
    image_size: int,
    dense_units: int,
    dropout: float,
    learning_rate: float,
    train_base: bool,
    unfreeze_last_block: bool,
    unfreeze_from_block: str,
    head_type: str,
) -> keras.Model:
    """
    VGG16 preentrenada + clasificador binario.

    head_type='thesis': reproduce la cabeza de model.py/TesisIvan:
        AveragePooling2D(3x3) -> Flatten -> Dense -> Dropout -> Sigmoid.

    head_type='gap': conserva la cabeza compacta del script multiclase actual:
        GlobalAveragePooling2D -> Dense -> Dropout -> Sigmoid.
    """
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
    print(f"       head_type: {head_type}")

    n_trainable = sum(int(layer.trainable) for layer in base_model.layers)
    print(f"       capas entrenables VGG16: {n_trainable}/{len(base_model.layers)}")

    x = base_model.output

    if head_type == "thesis":
        # Igual a model.py en espíritu. Con input 112, la salida de VGG16 es 3x3,
        # por lo que pool_size=3 sigue siendo válido.
        x = layers.AveragePooling2D(pool_size=(3, 3), name="avg_pool_3x3")(x)
        x = layers.Flatten(name="flatten")(x)
    elif head_type == "gap":
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    else:
        raise ValueError("head_type debe ser 'thesis' o 'gap'.")

    x = layers.Dense(int(dense_units), activation="relu", name="dense_head")(x)
    x = layers.Dropout(float(dropout), name="dropout_head")(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
        dtype="float32",
        name="type_iii_probability",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(learning_rate)),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
            keras.metrics.Precision(name="precision", thresholds=0.5),
            keras.metrics.Recall(name="recall", thresholds=0.5),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


# ============================================================
# CALLBACK F1 BINARIO PARA GRID SEARCH
# ============================================================

class ValidationBinaryMetricsCallback(keras.callbacks.Callback):
    def __init__(
        self,
        val_ds: tf.data.Dataset,
        threshold: float,
    ) -> None:
        super().__init__()
        self.val_ds = val_ds
        self.threshold = float(threshold)
        self.y_true = collect_labels_from_dataset(val_ds)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_prob = self.model.predict(self.val_ds, verbose=0).reshape(-1)
        y_pred = (y_prob >= self.threshold).astype(np.int32)

        macro_f1 = f1_score(
            self.y_true,
            y_pred,
            labels=[0, 1],
            average="macro",
            zero_division=0,
        )
        bal_acc = balanced_accuracy_score(self.y_true, y_pred)
        type_iii_f1 = f1_score(
            self.y_true,
            y_pred,
            pos_label=1,
            average="binary",
            zero_division=0,
        )

        logs["val_macro_f1"] = float(macro_f1)
        logs["val_balanced_accuracy"] = float(bal_acc)
        logs["val_type_iii_f1"] = float(type_iii_f1)

        print(
            f" - val_macro_f1: {macro_f1:.4f}"
            f" - val_balanced_accuracy: {bal_acc:.4f}"
            f" - val_type_iii_f1: {type_iii_f1:.4f}"
        )


# ============================================================
# EVALUACIÓN Y EXPORTACIÓN DE PREDICCIONES
# ============================================================

def evaluate_model_on_dataset(
    model: keras.Model,
    ds: tf.data.Dataset,
    threshold: float,
) -> tuple[dict[str, Any], np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    y_true = collect_labels_from_dataset(ds)
    y_prob = model.predict(ds, verbose=0).reshape(-1)
    y_pred = (y_prob >= threshold).astype(np.int32)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)
    weighted_f1 = f1_score(
        y_true, y_pred, labels=[0, 1], average="weighted", zero_division=0
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )

    cm_array = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).T

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    tn, fp, fn, tp = cm_array.ravel()
    fnr = fn / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "roc_auc": float(auc),
        "precision_OTHER_TYPES": float(precision[0]),
        "precision_TYPE_III": float(precision[1]),
        "recall_OTHER_TYPES": float(recall[0]),
        "recall_TYPE_III": float(recall[1]),
        "f1_OTHER_TYPES": float(f1[0]),
        "f1_TYPE_III": float(f1[1]),
        "support_OTHER_TYPES": int(support[0]),
        "support_TYPE_III": int(support[1]),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fnr_TYPE_III": float(fnr),
        "fpr_TYPE_III": float(fpr),
    }

    return metrics, cm_array, report_df, y_prob, y_pred


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
        title=f"{prefix}: TYPE III vs OTHER TYPES",
    )
    report_df.to_csv(
        output_dir / f"{prefix}_classification_report.csv",
        encoding="utf-8",
    )


def build_predictions_dataframe(
    source_df: pd.DataFrame,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    if len(source_df) != len(y_prob):
        raise ValueError(
            f"Cantidad de filas ({len(source_df)}) != predicciones ({len(y_prob)})."
        )

    out = source_df.copy().reset_index(drop=True)
    out["prob_TYPE_III"] = y_prob.astype(float)
    out["prob_OTHER_TYPES"] = 1.0 - out["prob_TYPE_III"]
    out["predicted_id"] = y_pred.astype(np.int32)
    out["predicted_label"] = out["predicted_id"].map(ID_TO_CLASS)
    out["is_correct"] = out["predicted_id"].eq(out["label_id"])
    return out


def export_predictions_in_two_parts(
    predictions_df: pd.DataFrame,
    output_dir: Path,
    export_images: bool,
) -> None:
    """
    Guarda la salida pedida para el segundo modelo en dos partes:

      predictions/TYPE_III/predictions.csv
      predictions/OTHER_TYPES/predictions.csv

    Opcionalmente copia también las PNG predichas a cada carpeta.
    """
    predictions_root = output_dir / "predictions"
    type_iii_dir = predictions_root / "TYPE_III"
    other_dir = predictions_root / "OTHER_TYPES"
    ensure_dir(type_iii_dir)
    ensure_dir(other_dir)

    predictions_df.to_csv(
        predictions_root / "all_predictions.csv", index=False, encoding="utf-8"
    )

    grouped = {
        "TYPE_III": predictions_df[predictions_df["predicted_label"].eq("TYPE_III")],
        "OTHER_TYPES": predictions_df[
            predictions_df["predicted_label"].eq("OTHER_TYPES")
        ],
    }

    for class_name, class_df in grouped.items():
        class_dir = predictions_root / class_name
        class_df.to_csv(class_dir / "predictions.csv", index=False, encoding="utf-8")

        if not export_images:
            continue

        images_dir = class_dir / "images"
        ensure_dir(images_dir)

        for row_idx, row in class_df.reset_index(drop=True).iterrows():
            src = Path(str(row["image_path"]))
            if not src.exists():
                continue

            # Prefijo numérico para evitar colisiones si dos estaciones usan el mismo nombre.
            dst = images_dir / f"{row_idx:06d}_{src.name}"
            shutil.copy2(src, dst)


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
    augment_other_types: bool,
    augment_type_iii: bool,
    seed: int,
    train_base: bool,
    unfreeze_last_block: bool,
    unfreeze_from_block: str,
    gaussian_noise_std: float,
    brightness_delta: float,
    contrast_low: float,
    contrast_high: float,
    translation_width_factor: float,
    threshold: float,
    head_type: str,
    export_images: bool,
) -> dict[str, Any]:
    config_name = (
        f"is{image_size}_bs{batch_size}_lr{slug_float(learning_rate)}_"
        f"du{dense_units}_do{slug_float(dropout)}_{head_type}"
    )
    config_dir = run_dir / config_name
    ensure_dir(config_dir)

    print()
    print("=" * 80)
    print(f"[INFO] Configuración {config_id}/{total_configs}: {config_name}")
    print("=" * 80)

    tf.keras.backend.clear_session()

    train_ds = make_balanced_train_dataset(
        train_df=train_df,
        image_size=image_size,
        batch_size=batch_size,
        class_sample_weights=class_sample_weights,
        augment_other_types=augment_other_types,
        augment_type_iii=augment_type_iii,
        seed=seed,
        gaussian_noise_std=gaussian_noise_std,
        brightness_delta=brightness_delta,
        contrast_low=contrast_low,
        contrast_high=contrast_high,
        translation_width_factor=translation_width_factor,
    )
    val_ds = make_eval_dataset(val_df, image_size, batch_size)
    test_ds = make_eval_dataset(test_df, image_size, batch_size)

    model = build_vgg16_binary_model(
        image_size=image_size,
        dense_units=dense_units,
        dropout=dropout,
        learning_rate=learning_rate,
        train_base=train_base,
        unfreeze_last_block=unfreeze_last_block,
        unfreeze_from_block=unfreeze_from_block,
        head_type=head_type,
    )

    model.summary(print_fn=lambda x: print(f"[MODEL] {x}"))

    # balanced sampling produce muestras con reemplazo; una época equivalente
    # recorre aproximadamente el tamaño del train original.
    steps_per_epoch = max(
        1,
        int(math.ceil(len(train_df) / batch_size) * float(steps_multiplier)),
    )

    best_model_path = config_dir / "best_model.keras"
    last_model_path = config_dir / "last_model.keras"

    val_metrics_callback = ValidationBinaryMetricsCallback(
        val_ds=val_ds,
        threshold=threshold,
    )

    callbacks = [
        val_metrics_callback,
        keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_macro_f1",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    start = time.monotonic()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        verbose=1,
    )
    train_seconds = time.monotonic() - start

    model.save(last_model_path)

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(config_dir / "history.csv", index=False, encoding="utf-8")

    # Garantiza que la evaluación final use el checkpoint seleccionado por macro-F1.
    if best_model_path.exists():
        best_model = keras.models.load_model(best_model_path)
    else:
        best_model = model

    val_metrics, val_cm, val_report, val_prob, val_pred = evaluate_model_on_dataset(
        best_model, val_ds, threshold=threshold
    )
    test_metrics, test_cm, test_report, test_prob, test_pred = evaluate_model_on_dataset(
        best_model, test_ds, threshold=threshold
    )

    save_evaluation_outputs(config_dir, "val", val_metrics, val_cm, val_report)
    save_evaluation_outputs(config_dir, "test", test_metrics, test_cm, test_report)

    val_predictions = build_predictions_dataframe(val_df, val_prob, val_pred)
    test_predictions = build_predictions_dataframe(test_df, test_prob, test_pred)
    val_predictions.to_csv(
        config_dir / "val_predictions.csv", index=False, encoding="utf-8"
    )
    test_predictions.to_csv(
        config_dir / "test_predictions.csv", index=False, encoding="utf-8"
    )

    # La división en carpetas TYPE_III / OTHER_TYPES se hace para TEST.
    export_predictions_in_two_parts(
        predictions_df=test_predictions,
        output_dir=config_dir,
        export_images=export_images,
    )

    epochs_run = int(len(history.epoch))
    best_val_macro_f1_history = (
        float(hist_df["val_macro_f1"].max())
        if "val_macro_f1" in hist_df.columns
        else float("nan")
    )

    result_row = {
        "config_id": config_id,
        "config_name": config_name,
        "config_dir": str(config_dir),
        "image_size": image_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "dense_units": dense_units,
        "dropout": dropout,
        "head_type": head_type,
        "threshold": threshold,
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
        "val_roc_auc": val_metrics["roc_auc"],
        "val_precision_TYPE_III": val_metrics["precision_TYPE_III"],
        "val_recall_TYPE_III": val_metrics["recall_TYPE_III"],
        "val_f1_TYPE_III": val_metrics["f1_TYPE_III"],
        "val_f1_OTHER_TYPES": val_metrics["f1_OTHER_TYPES"],
        "test_accuracy": test_metrics["accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_precision_TYPE_III": test_metrics["precision_TYPE_III"],
        "test_recall_TYPE_III": test_metrics["recall_TYPE_III"],
        "test_f1_TYPE_III": test_metrics["f1_TYPE_III"],
        "test_precision_OTHER_TYPES": test_metrics["precision_OTHER_TYPES"],
        "test_recall_OTHER_TYPES": test_metrics["recall_OTHER_TYPES"],
        "test_f1_OTHER_TYPES": test_metrics["f1_OTHER_TYPES"],
        "test_fnr_TYPE_III": test_metrics["fnr_TYPE_III"],
        "test_fpr_TYPE_III": test_metrics["fpr_TYPE_III"],
        "best_model_path": str(best_model_path),
        "last_model_path": str(last_model_path),
        "prediction_type_iii_csv": str(
            config_dir / "predictions" / "TYPE_III" / "predictions.csv"
        ),
        "prediction_other_types_csv": str(
            config_dir / "predictions" / "OTHER_TYPES" / "predictions.csv"
        ),
    }

    write_json(config_dir / "result_summary.json", result_row)

    print(f"[INFO] Tiempo de entrenamiento: {timedelta(seconds=train_seconds)}")
    print(
        f"[INFO] TEST: acc={test_metrics['accuracy']:.4f}, "
        f"macro_f1={test_metrics['macro_f1']:.4f}, "
        f"recall_TYPE_III={test_metrics['recall_TYPE_III']:.4f}, "
        f"f1_OTHER_TYPES={test_metrics['f1_OTHER_TYPES']:.4f}"
    )

    return result_row


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Grid search VGG16 binario con máscaras para el segundo modelo de la cascada: "
            "TYPE III vs OTHER TYPES (II + IV)."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Misma carpeta raíz del dataset multiclase; debe contener labels.csv.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help="Misma carpeta base de resultados de entrenamiento.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="pchip_vgg16_typeIII_vs_other_gridsearch",
        help="Nombre del experimento binario.",
    )
    parser.add_argument(
        "--mask-column",
        type=str,
        default="mask_path",
        help="Columna de labels.csv que contiene la máscara. Default: mask_path.",
    )

    # Defaults alineados al script multiclase actual para comparar ambos enfoques.
    parser.add_argument("--image-sizes", type=str, default="224")
    parser.add_argument("--batch-sizes", type=str, default="8,16")
    parser.add_argument("--learning-rates", type=str, default="1e-5,1e-6,1e-7")
    parser.add_argument("--dense-units", type=str, default="128,256")
    parser.add_argument("--dropouts", type=str, default="0.4,0.5")

    parser.add_argument(
        "--head-type",
        type=str,
        default="thesis",
        choices=["thesis", "gap"],
        help=(
            "'thesis' usa AveragePooling2D(3x3)+Flatten como model.py/TesisIvan; "
            "'gap' usa GlobalAveragePooling como el script multiclase actual."
        ),
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--steps-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--class-sample-weights",
        type=str,
        default="0.5,0.5",
        help="Pesos de muestreo OTHER_TYPES,TYPE_III. Default: 0.5,0.5.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Umbral de probabilidad para clasificar TYPE_III. Default: 0.5.",
    )

    # La tesis usó ruido gaussiano std=3 y translación temporal 10%.
    parser.add_argument("--gaussian-noise-std", type=float, default=3.0)
    parser.add_argument("--brightness-delta", type=float, default=8.0)
    parser.add_argument("--contrast-low", type=float, default=0.9)
    parser.add_argument("--contrast-high", type=float, default=1.1)
    parser.add_argument("--translation-width-factor", type=float, default=0.10)

    parser.add_argument(
        "--augment-other-types",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Aplica augmentation a OTHER_TYPES (II+IV). Default: true.",
    )
    parser.add_argument(
        "--augment-type-iii",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Aplica augmentation también a TYPE III. Default: false.",
    )

    parser.add_argument(
        "--train-base",
        action="store_true",
        help="Entrena toda VGG16 desde el inicio.",
    )
    parser.add_argument(
        "--unfreeze-last-block",
        action="store_true",
        help="Compatibilidad: descongela block5 si unfreeze-from-block=none.",
    )
    parser.add_argument(
        "--unfreeze-from-block",
        type=str,
        default="none",
        choices=["none", "block5", "block4"],
    )

    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Para pruebas rápidas: máximo por clase BINARIA.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument(
        "--export-predicted-images",
        action="store_true",
        help=(
            "Además de CSV, copia las imágenes de test a predictions/TYPE_III/images "
            "y predictions/OTHER_TYPES/images."
        ),
    )

    args = parser.parse_args()

    if not 0.0 < float(args.threshold) < 1.0:
        raise ValueError("threshold debe estar entre 0 y 1.")

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
    class_sample_weights = parse_binary_sample_weights(args.class_sample_weights)

    # AveragePooling 3x3 requiere al menos salida 3x3 del backbone (112 funciona).
    if args.head_type == "thesis" and any(size < 96 for size in image_sizes):
        raise ValueError(
            "Con --head-type thesis use image_size >= 96; 112/224/512 son seguros."
        )

    set_reproducibility(seed=args.seed, deterministic=bool(args.deterministic))
    configure_gpu_memory_growth()
    configure_mixed_precision(enabled=bool(args.mixed_precision))

    run_dir = runs_root / experiment_name
    ensure_dir(run_dir)

    print()
    print("[INFO] VGG16 binario: TYPE III vs OTHER TYPES (II + IV)")
    print(f"[INFO] TensorFlow version: {tf.__version__}")
    print(f"[INFO] Dataset root: {dataset_root}")
    print(f"[INFO] Runs root: {runs_root}")
    print(f"[INFO] Experiment name: {experiment_name}")
    print(f"[INFO] Run dir: {run_dir}")
    print(f"[INFO] Mask column: {args.mask_column}")
    print(f"[INFO] Binary classes: {CLASS_NAMES}")
    print(f"[INFO] Mapping original: II->OTHER_TYPES, III->TYPE_III, IV->OTHER_TYPES")
    print(f"[INFO] Class sample weights OTHER_TYPES/TYPE_III: {class_sample_weights}")
    print(f"[INFO] Head type: {args.head_type}")
    print(f"[INFO] Threshold TYPE_III: {args.threshold}")
    print()

    df = load_labels_dataframe(
        dataset_root=dataset_root,
        mask_column=args.mask_column,
        max_samples_per_binary_class=args.max_samples_per_class,
        seed=args.seed,
    )

    print("[INFO] Dataset binario cargado:")
    print(df["binary_label"].value_counts().reindex(CLASS_NAMES, fill_value=0).to_string())
    print("[INFO] Composición original:")
    print(df["original_label"].value_counts().reindex(["II", "III", "IV"], fill_value=0).to_string())
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
        "task": "TYPE_III_vs_OTHER_TYPES",
        "other_types_definition": ["II", "IV"],
        "positive_class": "TYPE_III",
        "dataset_root": str(dataset_root),
        "runs_root": str(runs_root),
        "experiment_name": experiment_name,
        "mask_column": args.mask_column,
        "image_sizes": image_sizes,
        "batch_sizes": batch_sizes,
        "learning_rates": learning_rates,
        "dense_units": dense_units_list,
        "dropouts": dropout_list,
        "head_type": args.head_type,
        "threshold": float(args.threshold),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "val_size": float(args.val_size),
        "test_size": float(args.test_size),
        "steps_multiplier": float(args.steps_multiplier),
        "class_sample_weights_OTHER_TYPES_TYPE_III": class_sample_weights,
        "augment_other_types": bool(args.augment_other_types),
        "augment_type_iii": bool(args.augment_type_iii),
        "gaussian_noise_std": float(args.gaussian_noise_std),
        "brightness_delta": float(args.brightness_delta),
        "contrast_low": float(args.contrast_low),
        "contrast_high": float(args.contrast_high),
        "translation_width_factor": float(args.translation_width_factor),
        "train_base": bool(args.train_base),
        "unfreeze_last_block": bool(args.unfreeze_last_block),
        "unfreeze_from_block": unfreeze_from_block,
        "max_samples_per_binary_class": args.max_samples_per_class,
        "seed": int(args.seed),
        "deterministic": bool(args.deterministic),
        "mixed_precision": bool(args.mixed_precision),
        "export_predicted_images": bool(args.export_predicted_images),
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
        tqdm(configs, desc="Grid search TYPE III vs OTHER", unit="config"),
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
            augment_other_types=bool(args.augment_other_types),
            augment_type_iii=bool(args.augment_type_iii),
            seed=int(args.seed),
            train_base=bool(args.train_base),
            unfreeze_last_block=bool(args.unfreeze_last_block),
            unfreeze_from_block=unfreeze_from_block,
            gaussian_noise_std=float(args.gaussian_noise_std),
            brightness_delta=float(args.brightness_delta),
            contrast_low=float(args.contrast_low),
            contrast_high=float(args.contrast_high),
            translation_width_factor=float(args.translation_width_factor),
            threshold=float(args.threshold),
            head_type=str(args.head_type),
            export_images=bool(args.export_predicted_images),
        )

        grid_rows.append(result_row)
        grid_df = pd.DataFrame(grid_rows).sort_values(
            [
                "val_macro_f1",
                "val_balanced_accuracy",
                "val_recall_TYPE_III",
                "val_f1_OTHER_TYPES",
            ],
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
            "val_recall_TYPE_III",
            "val_f1_OTHER_TYPES",
            "test_macro_f1",
            "test_recall_TYPE_III",
            "test_f1_OTHER_TYPES",
        ]].to_string())
        print()

    final_grid_df = pd.DataFrame(grid_rows)
    if not final_grid_df.empty:
        final_grid_df = final_grid_df.sort_values(
            [
                "val_macro_f1",
                "val_balanced_accuracy",
                "val_recall_TYPE_III",
                "val_f1_OTHER_TYPES",
            ],
            ascending=False,
        )
        final_grid_df.to_csv(grid_csv, index=False, encoding="utf-8")

        best_row = final_grid_df.iloc[0].to_dict()
        write_json(run_dir / "best_grid_config.json", best_row)

        print()
        print("=" * 80)
        print("[INFO] GRID SEARCH TERMINADO")
        print("=" * 80)
        print(pd.Series(best_row)[[
            "config_name",
            "image_size",
            "batch_size",
            "learning_rate",
            "dense_units",
            "dropout",
            "head_type",
            "val_macro_f1",
            "val_balanced_accuracy",
            "val_recall_TYPE_III",
            "val_f1_OTHER_TYPES",
            "test_macro_f1",
            "test_recall_TYPE_III",
            "test_f1_OTHER_TYPES",
            "best_model_path",
            "prediction_type_iii_csv",
            "prediction_other_types_csv",
        ]].to_string())
        print()
        print(f"[INFO] Resultados completos: {grid_csv}")
        print(f"[INFO] Mejor configuración JSON: {run_dir / 'best_grid_config.json'}")
    else:
        print("[WARN] No se generaron resultados de grid search.")


if __name__ == "__main__":
    main()
