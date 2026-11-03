#!/usr/bin/env python3
"""
ETAPA 2 del nuevo Modelo 1: SRB vs NO_SRB.

ETAPA 1:
    VGG16 completamente congelada + cabeza entrenada.
    Se parte de su best_model.keras.

ETAPA 2:
    - Carga el best_model.keras de la Etapa 1.
    - Mantiene congelados blocks 1-4.
    - Descongela solamente block5.
    - Mantiene entrenable la cabeza binaria.
    - Recompila con un learning rate pequeño (default: 1e-7).
    - Conserva el mismo split train/validation/test usando el mismo seed.
    - Augmentation por defecto solo para NO_SRB.
    - Selecciona el mejor checkpoint por val_macro_f1.
    - Guarda métricas, matrices, predicciones y breakdown por II/III/IV/NO_SRB.

IMPORTANTE:
Este script debe estar en la misma carpeta que:
    train_vgg16_gridsearch_model1_srb_vs_no_srb.py
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Permite importar el script base si ambos están en scripts/training/
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import train_vgg16_gridsearch_model1_srb_vs_no_srb as base


DEFAULT_DATASET_ROOT = Path(
    "/home/igonzalez/ecallisto/dataset_images_balanced_model_1_srb_vs_no_srb"
)

DEFAULT_STAGE1_MODEL = Path(
    "/home/igonzalez/ecallisto/ecallisto-srb-ai/training_runs/"
    "pchip_vgg16_model1_srb_vs_no_srb_frozen_w050_050_v2/"
    "is224_bs8_lr1em05_du128_do0p5_thesis/"
    "best_model.keras"
)

DEFAULT_RUNS_ROOT = Path(
    "/home/igonzalez/ecallisto/ecallisto-srb-ai/training_runs"
)

DEFAULT_EXPERIMENT_NAME = (
    "pchip_vgg16_model1_srb_vs_no_srb_stage2_block5_lr1em07_v3"
)


def configure_stage2_model(
    model: keras.Model,
    learning_rate: float,
) -> keras.Model:
    """
    Congela blocks 1-4 y descongela solamente block5 + cabeza.
    """
    head_names = {
        "avg_pool_3x3",
        "global_avg_pool",
        "flatten",
        "dense_head",
        "dropout_head",
        "srb_probability",
    }

    for layer in model.layers:
        if layer.name.startswith("block5"):
            layer.trainable = True
        elif layer.name in head_names:
            layer.trainable = True
        else:
            layer.trainable = False

    print("\n[INFO] Estado de capas para ETAPA 2:")
    for block in ["block1", "block2", "block3", "block4", "block5"]:
        layers_block = [
            layer for layer in model.layers
            if layer.name.startswith(block)
        ]
        trainable = sum(int(layer.trainable) for layer in layers_block)
        print(
            f"  {block}: {trainable}/{len(layers_block)} capas entrenables"
        )

    head_trainable = [
        layer.name
        for layer in model.layers
        if layer.name in head_names and layer.trainable
    ]
    print(f"  cabeza entrenable: {head_trainable}")

    trainable_params = int(
        sum(
            np.prod(v.shape)
            for v in model.trainable_weights
        )
    )
    non_trainable_params = int(
        sum(
            np.prod(v.shape)
            for v in model.non_trainable_weights
        )
    )

    print(f"  parámetros entrenables: {trainable_params:,}")
    print(f"  parámetros no entrenables: {non_trainable_params:,}")

    # Es obligatorio recompilar después de cambiar trainable.
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=float(learning_rate)
        ),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(
                name="accuracy",
                threshold=0.5,
            ),
            keras.metrics.Precision(
                name="precision",
                thresholds=0.5,
            ),
            keras.metrics.Recall(
                name="recall",
                thresholds=0.5,
            ),
            keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def save_original_label_breakdown(
    test_predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    rows = []

    for original_label in base.ORIGINAL_LABELS:
        sub = test_predictions[
            test_predictions["original_label"].eq(original_label)
        ]
        n = int(len(sub))

        if n == 0:
            rows.append(
                {
                    "original_label": original_label,
                    "support": 0,
                    "predicted_NO_SRB": 0,
                    "predicted_SRB": 0,
                    "accuracy_binary": float("nan"),
                }
            )
            continue

        rows.append(
            {
                "original_label": original_label,
                "support": n,
                "predicted_NO_SRB": int(
                    sub["predicted_label"].eq("NO_SRB").sum()
                ),
                "predicted_SRB": int(
                    sub["predicted_label"].eq("SRB").sum()
                ),
                "accuracy_binary": float(
                    sub["is_correct"].mean()
                ),
            }
        )

    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Etapa 2 SRB vs NO_SRB: carga el mejor modelo frozen "
            "de Etapa 1 y hace fine-tuning solamente de block5."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )
    parser.add_argument(
        "--stage1-model",
        type=Path,
        default=DEFAULT_STAGE1_MODEL,
        help="Ruta al best_model.keras de la Etapa 1.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=DEFAULT_EXPERIMENT_NAME,
    )

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-7)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--steps-multiplier", type=float, default=1.0)

    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)

    parser.add_argument(
        "--class-sample-weights",
        type=str,
        default="0.5,0.5",
        help="Orden: NO_SRB,SRB. Default: 0.5,0.5.",
    )

    parser.add_argument(
        "--augment-no-srb",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--augment-srb",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--gaussian-noise-std", type=float, default=3.0)
    parser.add_argument("--brightness-delta", type=float, default=8.0)
    parser.add_argument("--contrast-low", type=float, default=0.9)
    parser.add_argument("--contrast-high", type=float, default=1.1)
    parser.add_argument(
        "--translation-width-factor",
        type=float,
        default=0.10,
    )

    parser.add_argument("--mask-column", type=str, default="mask_path")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument(
        "--export-predicted-images",
        action="store_true",
    )

    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    stage1_model_path = args.stage1_model.expanduser().resolve()
    runs_root = args.runs_root.expanduser().resolve()

    if not stage1_model_path.exists():
        raise FileNotFoundError(
            f"No existe el modelo de Etapa 1:\n{stage1_model_path}"
        )

    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold debe estar entre 0 y 1.")

    class_sample_weights = base.parse_binary_sample_weights(
        args.class_sample_weights
    )

    base.set_reproducibility(
        seed=args.seed,
        deterministic=bool(args.deterministic),
    )
    base.configure_gpu_memory_growth()
    base.configure_mixed_precision(
        enabled=bool(args.mixed_precision)
    )

    run_dir = runs_root / args.experiment_name
    base.ensure_dir(run_dir)

    print("\n" + "=" * 80)
    print("[INFO] ETAPA 2 - Modelo 1 SRB vs NO_SRB")
    print("=" * 80)
    print(f"[INFO] Dataset: {dataset_root}")
    print(f"[INFO] Modelo Etapa 1: {stage1_model_path}")
    print(f"[INFO] Salida: {run_dir}")
    print(f"[INFO] LR fine-tuning: {args.learning_rate}")
    print(f"[INFO] Batch size: {args.batch_size}")
    print(f"[INFO] Epochs máx.: {args.epochs}")
    print(f"[INFO] Patience: {args.patience}")
    print(
        "[INFO] Pesos de muestreo NO_SRB/SRB: "
        f"{class_sample_weights}"
    )
    print(
        "[INFO] Augmentation: "
        f"NO_SRB={args.augment_no_srb}, SRB={args.augment_srb}"
    )

    # ========================================================
    # MISMO DATASET Y MISMO SPLIT QUE ETAPA 1
    # ========================================================
    df = base.load_labels_dataframe(
        dataset_root=dataset_root,
        mask_column=args.mask_column,
        max_samples_per_binary_class=None,
        seed=args.seed,
    )

    train_df, val_df, test_df = base.split_train_val_test(
        df=df,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    base.print_split_summary("TRAIN", train_df)
    base.print_split_summary("VALIDATION", val_df)
    base.print_split_summary("TEST", test_df)

    train_ds = base.make_balanced_train_dataset(
        train_df=train_df,
        image_size=args.image_size,
        batch_size=args.batch_size,
        class_sample_weights=class_sample_weights,
        augment_no_srb=bool(args.augment_no_srb),
        augment_srb=bool(args.augment_srb),
        seed=args.seed,
        gaussian_noise_std=args.gaussian_noise_std,
        brightness_delta=args.brightness_delta,
        contrast_low=args.contrast_low,
        contrast_high=args.contrast_high,
        translation_width_factor=args.translation_width_factor,
    )

    val_ds = base.make_eval_dataset(
        val_df,
        args.image_size,
        args.batch_size,
    )
    test_ds = base.make_eval_dataset(
        test_df,
        args.image_size,
        args.batch_size,
    )

    # ========================================================
    # CARGAR MEJOR CHECKPOINT DE ETAPA 1
    # ========================================================
    print("\n[INFO] Cargando best_model.keras de Etapa 1...")
    model = keras.models.load_model(stage1_model_path)

    model = configure_stage2_model(
        model=model,
        learning_rate=args.learning_rate,
    )

    model.summary(
        print_fn=lambda x: print(f"[MODEL] {x}")
    )

    steps_per_epoch = max(
        1,
        int(
            math.ceil(len(train_df) / args.batch_size)
            * float(args.steps_multiplier)
        ),
    )

    best_model_path = run_dir / "best_model.keras"
    last_model_path = run_dir / "last_model.keras"

    val_metrics_callback = base.ValidationBinaryMetricsCallback(
        val_ds=val_ds,
        threshold=args.threshold,
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
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_macro_f1",
            mode="max",
            factor=0.5,
            patience=max(3, args.patience // 3),
            min_lr=1e-9,
            verbose=1,
        ),
    ]

    # ========================================================
    # FINE-TUNING
    # ========================================================
    start = time.monotonic()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        verbose=1,
    )

    train_seconds = time.monotonic() - start

    model.save(last_model_path)

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(
        run_dir / "history.csv",
        index=False,
        encoding="utf-8",
    )

    # ========================================================
    # EVALUAR SIEMPRE EL MEJOR CHECKPOINT DE ETAPA 2
    # ========================================================
    if best_model_path.exists():
        best_model = keras.models.load_model(best_model_path)
    else:
        best_model = model

    (
        val_metrics,
        val_cm,
        val_report,
        val_prob,
        val_pred,
    ) = base.evaluate_model_on_dataset(
        best_model,
        val_ds,
        threshold=args.threshold,
    )

    (
        test_metrics,
        test_cm,
        test_report,
        test_prob,
        test_pred,
    ) = base.evaluate_model_on_dataset(
        best_model,
        test_ds,
        threshold=args.threshold,
    )

    base.save_evaluation_outputs(
        run_dir,
        "val",
        val_metrics,
        val_cm,
        val_report,
    )

    base.save_evaluation_outputs(
        run_dir,
        "test",
        test_metrics,
        test_cm,
        test_report,
    )

    val_predictions = base.build_predictions_dataframe(
        val_df,
        val_prob,
        val_pred,
    )
    test_predictions = base.build_predictions_dataframe(
        test_df,
        test_prob,
        test_pred,
    )

    val_predictions.to_csv(
        run_dir / "val_predictions.csv",
        index=False,
        encoding="utf-8",
    )
    test_predictions.to_csv(
        run_dir / "test_predictions.csv",
        index=False,
        encoding="utf-8",
    )

    breakdown_path = (
        run_dir / "test_breakdown_by_original_label.csv"
    )
    save_original_label_breakdown(
        test_predictions,
        breakdown_path,
    )

    base.export_predictions_in_two_parts(
        predictions_df=test_predictions,
        output_dir=run_dir,
        export_images=bool(args.export_predicted_images),
    )

    epochs_run = len(history.epoch)

    best_val_macro_f1_history = (
        float(hist_df["val_macro_f1"].max())
        if "val_macro_f1" in hist_df.columns
        else float("nan")
    )

    result = {
        "stage": 2,
        "task": "SRB_vs_NO_SRB",
        "source_stage1_model": str(stage1_model_path),
        "dataset_root": str(dataset_root),
        "run_dir": str(run_dir),
        "image_size": int(args.image_size),
        "batch_size": int(args.batch_size),
        "learning_rate_initial": float(args.learning_rate),
        "fine_tuned_blocks": ["block5"],
        "frozen_blocks": [
            "block1",
            "block2",
            "block3",
            "block4",
        ],
        "class_sample_weights_NO_SRB_SRB": class_sample_weights,
        "augment_NO_SRB": bool(args.augment_no_srb),
        "augment_SRB": bool(args.augment_srb),
        "threshold": float(args.threshold),
        "epochs_run": int(epochs_run),
        "steps_per_epoch": int(steps_per_epoch),
        "train_seconds": float(train_seconds),
        "best_val_macro_f1_history": best_val_macro_f1_history,

        "val_accuracy": val_metrics["accuracy"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_roc_auc": val_metrics["roc_auc"],
        "val_precision_NO_SRB": val_metrics["precision_NO_SRB"],
        "val_recall_NO_SRB": val_metrics["recall_NO_SRB"],
        "val_f1_NO_SRB": val_metrics["f1_NO_SRB"],
        "val_precision_SRB": val_metrics["precision_SRB"],
        "val_recall_SRB": val_metrics["recall_SRB"],
        "val_f1_SRB": val_metrics["f1_SRB"],

        "test_accuracy": test_metrics["accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_precision_NO_SRB": test_metrics["precision_NO_SRB"],
        "test_recall_NO_SRB": test_metrics["recall_NO_SRB"],
        "test_f1_NO_SRB": test_metrics["f1_NO_SRB"],
        "test_precision_SRB": test_metrics["precision_SRB"],
        "test_recall_SRB": test_metrics["recall_SRB"],
        "test_f1_SRB": test_metrics["f1_SRB"],
        "test_fnr_SRB": test_metrics["fnr_SRB"],
        "test_fpr_SRB": test_metrics["fpr_SRB"],

        "best_model_path": str(best_model_path),
        "last_model_path": str(last_model_path),
        "test_breakdown_by_original_label_csv": str(
            breakdown_path
        ),
    }

    base.write_json(
        run_dir / "result_summary.json",
        result,
    )

    print("\n" + "=" * 80)
    print("[OK] ETAPA 2 FINALIZADA")
    print("=" * 80)
    print(
        f"[INFO] Tiempo: "
        f"{timedelta(seconds=train_seconds)}"
    )
    print(
        f"[INFO] VAL  -> "
        f"macro_f1={val_metrics['macro_f1']:.4f}, "
        f"bal_acc={val_metrics['balanced_accuracy']:.4f}, "
        f"auc={val_metrics['roc_auc']:.4f}"
    )
    print(
        f"[INFO] TEST -> "
        f"macro_f1={test_metrics['macro_f1']:.4f}, "
        f"bal_acc={test_metrics['balanced_accuracy']:.4f}, "
        f"auc={test_metrics['roc_auc']:.4f}"
    )
    print(
        f"[INFO] TEST recall NO_SRB="
        f"{test_metrics['recall_NO_SRB']:.4f}"
    )
    print(
        f"[INFO] TEST recall SRB="
        f"{test_metrics['recall_SRB']:.4f}"
    )
    print(f"[INFO] Mejor modelo: {best_model_path}")


if __name__ == "__main__":
    main()
