from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class EllipticConfig:
    dataset_dir: str = "dataset"


def load_elliptic_dataset(
    config: EllipticConfig,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    project_root = Path(__file__).resolve().parents[1]
    dataset_dir = project_root / config.dataset_dir

    features_path = (
        dataset_dir / "elliptic_txs_features.csv"
    )
    classes_path = (
        dataset_dir / "elliptic_txs_classes.csv"
    )
    edges_path = (
        dataset_dir / "elliptic_txs_edgelist.csv"
    )

    required_files = [
        features_path,
        classes_path,
        edges_path,
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
            "Required Elliptic dataset files "
            "were not found:\n"
            f"{missing_list}"
        )

    print("Loading transaction features...")

    # The features file does not contain column headings
    raw_features = pd.read_csv(
        features_path,
        header=None,
    )

    if raw_features.shape[1] < 3:
        raise ValueError(
            "The transaction feature file does not "
            "contain the expected columns."
        )

    feature_count = raw_features.shape[1] - 2

    feature_names = [
        f"feature_{index:03d}"
        for index in range(1, feature_count + 1)
    ]

    raw_features.columns = [
        "transaction_id",
        "time_step",
        *feature_names,
    ]

    print("Loading transaction labels...")

    classes = pd.read_csv(classes_path)

    required_class_columns = {
        "txId",
        "class",
    }

    if not required_class_columns.issubset(
        classes.columns
    ):
        raise ValueError(
            "The classes file must contain txId "
            "and class columns. "
            f"Found: {list(classes.columns)}"
        )

    raw_features["transaction_id"] = (
        raw_features["transaction_id"].astype(str)
    )

    classes["txId"] = classes["txId"].astype(str)

    classes = classes.rename(
        columns={
            "txId": "transaction_id",
            "class": "transaction_class",
        }
    )

    data = raw_features.merge(
        classes[
            [
                "transaction_id",
                "transaction_class",
            ]
        ],
        on="transaction_id",
        how="inner",
    )

    data["transaction_class"] = (
        data["transaction_class"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Elliptic labels:
    # 1 = illicit
    # 2 = licit
    # unknown = unlabelled
    labelled_data = data[
        data["transaction_class"].isin(["1", "2"])
    ].copy()

    labelled_data["label"] = (
        labelled_data["transaction_class"]
        .map(
            {
                "1": 1,
                "2": 0,
            }
        )
        .astype(int)
    )

    features = (
        labelled_data[feature_names]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    labels = labelled_data["label"].reset_index(
        drop=True
    )

    time_steps = pd.to_numeric(
        labelled_data["time_step"],
        errors="raise",
    ).reset_index(drop=True)

    transaction_ids = (
        labelled_data["transaction_id"]
        .reset_index(drop=True)
    )

    features = features.reset_index(drop=True)

    print(f"Total transactions: {len(data):,}")
    print(
        f"Labelled transactions: "
        f"{len(labelled_data):,}"
    )
    print(
        f"Licit transactions: "
        f"{int((labels == 0).sum()):,}"
    )
    print(
        f"Illicit transactions: "
        f"{int((labels == 1).sum()):,}"
    )
    print(f"Model features: {features.shape[1]}")
    print(
        f"Time steps: "
        f"{time_steps.min()}–{time_steps.max()}"
    )

    return (
        features,
        labels,
        time_steps,
        transaction_ids,
    )


def main() -> None:
    config = EllipticConfig()
    load_elliptic_dataset(config)

    print("Dataset loaded successfully.")


if __name__ == "__main__":
    main()