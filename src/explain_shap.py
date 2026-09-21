from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


GLOBAL_SAMPLE_SIZE = 500
RANDOM_STATE = 42
MAX_DISPLAY_FEATURES = 20


def extract_illicit_shap_values(
    raw_shap_values,
) -> np.ndarray:
    """Extract SHAP values for the illicit class."""
    if isinstance(raw_shap_values, list):
        values = raw_shap_values[1]
    elif (
        isinstance(raw_shap_values, np.ndarray)
        and raw_shap_values.ndim == 3
    ):
        values = raw_shap_values[:, :, 1]
    else:
        values = raw_shap_values

    values = np.asarray(values)

    if values.ndim != 2:
        raise ValueError(
            "Expected two-dimensional SHAP values, "
            f"but received shape {values.shape}."
        )

    return values


def extract_base_value(
    explainer: shap.TreeExplainer,
) -> float:
    expected_values = np.asarray(
        explainer.expected_value,
        dtype=float,
    ).reshape(-1)

    if expected_values.size > 1:
        return float(expected_values[1])

    return float(expected_values[0])


def save_local_explanation(
    explainer: shap.TreeExplainer,
    sample: pd.DataFrame,
    probability: float,
    description: str,
    output_path: Path,
) -> None:
    raw_values = explainer.shap_values(sample)

    values = extract_illicit_shap_values(
        raw_values
    )[0]

    explanation = shap.Explanation(
        values=values,
        base_values=extract_base_value(explainer),
        data=sample.iloc[0].to_numpy(),
        feature_names=list(sample.columns),
    )

    shap.plots.waterfall(
        explanation,
        max_display=15,
        show=False,
    )

    figure = plt.gcf()
    figure.set_size_inches(10, 6)

    figure.suptitle(
        (
            f"{description}\n"
            f"Predicted illicit probability: "
            f"{probability:.1%}"
        ),
        fontsize=13,
    )

    figure.subplots_adjust(top=0.84)

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {output_path}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    outputs_dir = project_root / "outputs"

    model_path = outputs_dir / "best_model.joblib"
    test_features_path = outputs_dir / "X_test.csv"
    test_labels_path = outputs_dir / "y_test.csv"
    metadata_path = outputs_dir / "model_metadata.json"

    required_files = [
        model_path,
        test_features_path,
        test_labels_path,
        metadata_path,
    ]

    missing_files = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        missing_list = "\n".join(
            f"  - {path}"
            for path in missing_files
        )

        raise FileNotFoundError(
            "Required model files were not found:\n"
            f"{missing_list}\n"
            "Run train_model.py first."
        )

    print("Loading trained model and test data...")

    model = joblib.load(model_path)
    X_test = pd.read_csv(test_features_path)
    test_results = pd.read_csv(test_labels_path)

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    decision_threshold = float(
        metadata["decision_threshold"]
    )

    print(
        f"Model: {metadata['selected_model']}"
    )
    print(
        f"Decision threshold: "
        f"{decision_threshold:.4f}"
    )

    if len(X_test) != len(test_results):
        raise ValueError(
            "X_test.csv and y_test.csv contain "
            "different numbers of rows."
        )

    sample_size = min(
        GLOBAL_SAMPLE_SIZE,
        len(X_test),
    )

    X_sample = X_test.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    )

    print(
        f"Calculating global SHAP values for "
        f"{sample_size} transactions..."
    )

    explainer = shap.TreeExplainer(model)
    raw_shap_values = explainer.shap_values(
        X_sample
    )

    shap_values = extract_illicit_shap_values(
        raw_shap_values
    )

    # Global SHAP beeswarm plot
    summary_path = outputs_dir / "shap_summary.png"

    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=list(X_sample.columns),
        max_display=MAX_DISPLAY_FEATURES,
        plot_size=(9, 7),
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        summary_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {summary_path}")

    # Global mean absolute SHAP importance
    importance_path = (
        outputs_dir / "shap_feature_importance.png"
    )

    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=list(X_sample.columns),
        plot_type="bar",
        max_display=MAX_DISPLAY_FEATURES,
        plot_size=(9, 7),
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        importance_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {importance_path}")

    # Correctly detected illicit transaction
    true_positive_rows = test_results[
        (test_results["true_label"] == 1)
        & (test_results["predicted_label"] == 1)
    ]

    if not true_positive_rows.empty:
        true_positive_index = (
            true_positive_rows[
                "illicit_probability"
            ].idxmax()
        )

        save_local_explanation(
            explainer=explainer,
            sample=X_test.loc[
                [true_positive_index]
            ],
            probability=float(
                test_results.loc[
                    true_positive_index,
                    "illicit_probability",
                ]
            ),
            description=(
                "Correctly detected illicit transaction"
            ),
            output_path=(
                outputs_dir
                / "shap_true_positive.png"
            ),
        )

    # Missed illicit transaction closest to the threshold
    false_negative_rows = test_results[
        (test_results["true_label"] == 1)
        & (test_results["predicted_label"] == 0)
    ]

    if not false_negative_rows.empty:
        false_negative_index = (
            false_negative_rows[
                "illicit_probability"
            ].idxmax()
        )

        save_local_explanation(
            explainer=explainer,
            sample=X_test.loc[
                [false_negative_index]
            ],
            probability=float(
                test_results.loc[
                    false_negative_index,
                    "illicit_probability",
                ]
            ),
            description=(
                "Missed illicit transaction "
                "closest to the threshold"
            ),
            output_path=(
                outputs_dir
                / "shap_false_negative.png"
            ),
        )

    # Incorrectly flagged licit transaction
    false_positive_rows = test_results[
        (test_results["true_label"] == 0)
        & (test_results["predicted_label"] == 1)
    ]

    if not false_positive_rows.empty:
        false_positive_index = (
            false_positive_rows[
                "illicit_probability"
            ].idxmax()
        )

        save_local_explanation(
            explainer=explainer,
            sample=X_test.loc[
                [false_positive_index]
            ],
            probability=float(
                test_results.loc[
                    false_positive_index,
                    "illicit_probability",
                ]
            ),
            description=(
                "Incorrectly flagged licit transaction"
            ),
            output_path=(
                outputs_dir
                / "shap_false_positive.png"
            ),
        )

    print("\nSHAP analysis completed successfully.")


if __name__ == "__main__":
    main()