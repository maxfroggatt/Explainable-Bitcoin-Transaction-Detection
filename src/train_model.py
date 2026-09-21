from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from elliptic_preprocessing import (
    EllipticConfig,
    load_elliptic_dataset,
)


RANDOM_STATE = 42

TRAIN_END_TIME = 29
VALIDATION_START_TIME = 30
VALIDATION_END_TIME = 34
TEST_START_TIME = 35


def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float, float, float]:
    precision, recall, thresholds = precision_recall_curve(
        labels,
        probabilities,
    )

    if len(thresholds) == 0:
        return 0.5, 0.0, 0.0, 0.0

    precision = precision[:-1]
    recall = recall[:-1]

    denominator = precision + recall

    f1_scores = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )

    best_index = int(np.argmax(f1_scores))

    return (
        float(thresholds[best_index]),
        float(precision[best_index]),
        float(recall[best_index]),
        float(f1_scores[best_index]),
    )


def create_candidate_models() -> dict[str, Any]:
    return {
        "Random Forest - unweighted": (
            RandomForestClassifier(
                n_estimators=600,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                max_features="sqrt",
                min_samples_leaf=1,
                class_weight=None,
            )
        ),

        "Random Forest - weight 2": (
            RandomForestClassifier(
                n_estimators=600,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                max_features="sqrt",
                min_samples_leaf=1,
                class_weight={0: 1, 1: 2},
            )
        ),

        "Random Forest - weight 5": (
            RandomForestClassifier(
                n_estimators=600,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                max_features="sqrt",
                min_samples_leaf=1,
                class_weight={0: 1, 1: 5},
            )
        ),

        "Random Forest - balanced": (
            RandomForestClassifier(
                n_estimators=600,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                max_features="sqrt",
                min_samples_leaf=1,
                class_weight="balanced_subsample",
            )
        ),

        "Random Forest - balanced leaf 3": (
            RandomForestClassifier(
                n_estimators=600,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                max_features="sqrt",
                min_samples_leaf=3,
                class_weight="balanced_subsample",
            )
        ),

        "Extra Trees - balanced": (
            ExtraTreesClassifier(
                n_estimators=600,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                max_features="sqrt",
                min_samples_leaf=1,
                class_weight="balanced",
            )
        ),

        "Extra Trees - balanced leaf 3": (
            ExtraTreesClassifier(
                n_estimators=600,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                max_features="sqrt",
                min_samples_leaf=3,
                class_weight="balanced",
            )
        ),

        "Histogram Gradient Boosting": (
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=500,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=1.0,
                class_weight="balanced",
                early_stopping=True,
                random_state=RANDOM_STATE,
            )
        ),
    }


def class_summary(
    name: str,
    labels: pd.Series,
) -> None:
    licit_count = int((labels == 0).sum())
    illicit_count = int((labels == 1).sum())

    print(
        f"{name}: {len(labels):,} transactions "
        f"({licit_count:,} licit, "
        f"{illicit_count:,} illicit)"
    )


def serialise(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            str(key): serialise(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            serialise(item)
            for item in value
        ]

    return value


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Elliptic transaction dataset...")

    (
        features,
        labels,
        time_steps,
        transaction_ids,
    ) = load_elliptic_dataset(EllipticConfig())

    train_mask = time_steps <= TRAIN_END_TIME

    validation_mask = (
        (time_steps >= VALIDATION_START_TIME)
        & (time_steps <= VALIDATION_END_TIME)
    )

    test_mask = time_steps >= TEST_START_TIME

    X_train = features.loc[train_mask].reset_index(
        drop=True
    )
    y_train = labels.loc[train_mask].reset_index(
        drop=True
    )

    X_validation = features.loc[
        validation_mask
    ].reset_index(drop=True)

    y_validation = labels.loc[
        validation_mask
    ].reset_index(drop=True)

    X_test = features.loc[test_mask].reset_index(
        drop=True
    )
    y_test = labels.loc[test_mask].reset_index(
        drop=True
    )

    test_times = time_steps.loc[
        test_mask
    ].reset_index(drop=True)

    test_transaction_ids = transaction_ids.loc[
        test_mask
    ].reset_index(drop=True)

    if any(
        len(split) == 0
        for split in [
            X_train,
            X_validation,
            X_test,
        ]
    ):
        raise ValueError(
            "One or more chronological dataset splits "
            "are empty."
        )

    print()
    class_summary("Training", y_train)
    class_summary("Validation", y_validation)
    class_summary("Testing", y_test)

    candidate_models = create_candidate_models()

    validation_results: list[dict[str, Any]] = []

    best_model_name: str | None = None
    best_model_template = None
    best_threshold = 0.5
    best_validation_f1 = -1.0
    best_validation_ap = -1.0

    print("\nComparing candidate models...")

    for model_name, model_template in (
        candidate_models.items()
    ):
        print(f"\nTraining: {model_name}")

        model = clone(model_template)
        model.fit(X_train, y_train)

        illicit_position = list(
            model.classes_
        ).index(1)

        validation_probabilities = model.predict_proba(
            X_validation
        )[:, illicit_position]

        (
            threshold,
            validation_precision,
            validation_recall,
            validation_f1,
        ) = select_threshold(
            y_validation.to_numpy(),
            validation_probabilities,
        )

        validation_ap = average_precision_score(
            y_validation,
            validation_probabilities,
        )

        validation_roc_auc = roc_auc_score(
            y_validation,
            validation_probabilities,
        )

        print(f"  Threshold: {threshold:.4f}")
        print(
            f"  Illicit precision: "
            f"{validation_precision:.4f}"
        )
        print(
            f"  Illicit recall:    "
            f"{validation_recall:.4f}"
        )
        print(
            f"  Illicit F1:        "
            f"{validation_f1:.4f}"
        )
        print(
            f"  Average precision: "
            f"{validation_ap:.4f}"
        )
        print(
            f"  ROC-AUC:           "
            f"{validation_roc_auc:.4f}"
        )

        validation_results.append(
            {
                "model": model_name,
                "threshold": threshold,
                "illicit_precision": (
                    validation_precision
                ),
                "illicit_recall": validation_recall,
                "illicit_f1": validation_f1,
                "average_precision": validation_ap,
                "roc_auc": validation_roc_auc,
            }
        )

        # Select primarily by illicit F1, then AP
        candidate_score = (
            validation_f1,
            validation_ap,
        )

        best_score = (
            best_validation_f1,
            best_validation_ap,
        )

        if candidate_score > best_score:
            best_model_name = model_name
            best_model_template = model_template
            best_threshold = threshold
            best_validation_f1 = validation_f1
            best_validation_ap = validation_ap

    if (
        best_model_name is None
        or best_model_template is None
    ):
        raise RuntimeError(
            "No candidate model completed successfully."
        )

    print()
    print("=" * 60)
    print(f"Selected model: {best_model_name}")
    print(
        f"Validation illicit F1: "
        f"{best_validation_f1:.4f}"
    )
    print(
        f"Validation average precision: "
        f"{best_validation_ap:.4f}"
    )
    print(
        f"Selected threshold: "
        f"{best_threshold:.4f}"
    )
    print("=" * 60)

    # Refit using all development time steps 1-34
    development_mask = (
        time_steps <= VALIDATION_END_TIME
    )

    X_development = features.loc[
        development_mask
    ].reset_index(drop=True)

    y_development = labels.loc[
        development_mask
    ].reset_index(drop=True)

    print(
        f"\nRefitting selected model on "
        f"{len(X_development):,} development rows..."
    )

    final_model = clone(best_model_template)
    final_model.fit(
        X_development,
        y_development,
    )

    illicit_position = list(
        final_model.classes_
    ).index(1)

    test_probabilities = final_model.predict_proba(
        X_test
    )[:, illicit_position]

    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        test_predictions,
    )

    baseline_accuracy = float(
        np.mean(y_test == 0)
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        test_predictions,
    )

    illicit_precision = precision_score(
        y_test,
        test_predictions,
        pos_label=1,
        zero_division=0,
    )

    illicit_recall = recall_score(
        y_test,
        test_predictions,
        pos_label=1,
        zero_division=0,
    )

    illicit_f1 = f1_score(
        y_test,
        test_predictions,
        pos_label=1,
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_test,
        test_predictions,
        average="macro",
        zero_division=0,
    )

    average_precision = average_precision_score(
        y_test,
        test_probabilities,
    )

    roc_auc = roc_auc_score(
        y_test,
        test_probabilities,
    )

    report = classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["licit", "illicit"],
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1],
    )

    metrics_path = outputs_dir / "metrics.txt"

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            f"Selected model:           "
            f"{best_model_name}\n"
        )
        file.write(
            f"Decision threshold:       "
            f"{best_threshold:.4f}\n\n"
        )

        file.write(
            f"Accuracy:                  "
            f"{accuracy:.4f}\n"
        )
        file.write(
            f"Majority baseline:         "
            f"{baseline_accuracy:.4f}\n"
        )
        file.write(
            f"Balanced accuracy:         "
            f"{balanced_accuracy:.4f}\n"
        )
        file.write(
            f"Illicit precision:         "
            f"{illicit_precision:.4f}\n"
        )
        file.write(
            f"Illicit recall:            "
            f"{illicit_recall:.4f}\n"
        )
        file.write(
            f"Illicit F1-score:          "
            f"{illicit_f1:.4f}\n"
        )
        file.write(
            f"Macro F1-score:            "
            f"{macro_f1:.4f}\n"
        )
        file.write(
            f"Average precision:         "
            f"{average_precision:.4f}\n"
        )
        file.write(
            f"ROC-AUC:                   "
            f"{roc_auc:.4f}\n\n"
        )

        file.write("Classification report:\n")
        file.write(report)

        file.write("\nConfusion matrix:\n")
        file.write(str(matrix))

    with (
        outputs_dir / "validation_results.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            serialise(validation_results),
            file,
            indent=2,
        )

    # Confusion matrix
    confusion_figure, confusion_axis = plt.subplots(
        figsize=(6, 5)
    )

    confusion_display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["licit", "illicit"],
    )

    confusion_display.plot(
        ax=confusion_axis,
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )

    confusion_axis.set_title(
        f"{best_model_name} Confusion Matrix"
    )

    confusion_figure.tight_layout()

    confusion_figure.savefig(
        outputs_dir / "confusion_matrix.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(confusion_figure)

    # Precision-recall curve
    precision_values, recall_values, _ = (
        precision_recall_curve(
            y_test,
            test_probabilities,
        )
    )

    pr_figure, pr_axis = plt.subplots(
        figsize=(7, 5)
    )

    pr_axis.plot(
        recall_values,
        precision_values,
        label=(
            f"{best_model_name} "
            f"(AP = {average_precision:.3f})"
        ),
    )

    prevalence = float(np.mean(y_test))

    pr_axis.axhline(
        prevalence,
        color="grey",
        linestyle="--",
        label=(
            f"Random baseline "
            f"({prevalence:.3f})"
        ),
    )

    pr_axis.scatter(
        illicit_recall,
        illicit_precision,
        color="red",
        zorder=3,
        label=(
            f"Selected threshold "
            f"({best_threshold:.3f})"
        ),
    )

    pr_axis.set_xlabel("Recall")
    pr_axis.set_ylabel("Precision")
    pr_axis.set_title(
        "Illicit-Transaction Precision-Recall Curve"
    )
    pr_axis.set_xlim(0, 1)
    pr_axis.set_ylim(0, 1)
    pr_axis.grid(alpha=0.25)
    pr_axis.legend()

    pr_figure.tight_layout()

    pr_figure.savefig(
        outputs_dir / "precision_recall_curve.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(pr_figure)

    # Save model and test data for SHAP
    joblib.dump(
        final_model,
        outputs_dir / "best_model.joblib",
    )

    X_test.to_csv(
        outputs_dir / "X_test.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "transaction_id": test_transaction_ids,
            "time_step": test_times,
            "true_label": y_test,
            "illicit_probability": test_probabilities,
            "predicted_label": test_predictions,
        }
    ).to_csv(
        outputs_dir / "y_test.csv",
        index=False,
    )

    metadata = {
        "selected_model": best_model_name,
        "decision_threshold": best_threshold,
        "train_time_steps": "1-29",
        "validation_time_steps": "30-34",
        "test_time_steps": "35-49",
        "metrics": {
            "accuracy": accuracy,
            "majority_baseline": baseline_accuracy,
            "balanced_accuracy": balanced_accuracy,
            "illicit_precision": illicit_precision,
            "illicit_recall": illicit_recall,
            "illicit_f1": illicit_f1,
            "macro_f1": macro_f1,
            "average_precision": average_precision,
            "roc_auc": roc_auc,
        },
    }

    with (
        outputs_dir / "model_metadata.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            serialise(metadata),
            file,
            indent=2,
        )

    print("\nFinal chronological test results:")
    print(f"Selected model:      {best_model_name}")
    print(f"Accuracy:            {accuracy:.4f}")
    print(
        f"Balanced accuracy:   "
        f"{balanced_accuracy:.4f}"
    )
    print(
        f"Illicit precision:   "
        f"{illicit_precision:.4f}"
    )
    print(
        f"Illicit recall:      "
        f"{illicit_recall:.4f}"
    )
    print(
        f"Illicit F1-score:    "
        f"{illicit_f1:.4f}"
    )
    print(
        f"Average precision:   "
        f"{average_precision:.4f}"
    )
    print(f"ROC-AUC:             {roc_auc:.4f}")
    print(f"Metrics saved to:    {metrics_path}")


if __name__ == "__main__":
    main()