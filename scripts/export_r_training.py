"""Recreate the R input using the SUPPORT2 notebook's cohort and split.

Run from any directory: python scripts/export_r_training.py
Uses the repository's included CSV; no network or model fitting is required.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "support2.csv"
    df = pd.read_csv(source)
    if not df["id"].is_unique:
        raise ValueError("Expected one source row per patient ID.")
    day3 = df.loc[df["d.time"] > 3].copy()
    day3["time"] = day3["d.time"] - 3
    day3["event"] = day3["death"]
    excluded = [
        "id", "death", "d.time", "hospdead", "slos", "sfdm2",
        "aps", "sps", "surv2m", "surv6m", "prg2m", "prg6m",
        "dnr", "dnrday", "charges", "totcst", "totmcst", "avtisst",
    ]
    X = df.drop(columns=excluded).loc[day3.index].copy()
    y = day3.loc[X.index, ["time", "event"]]
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y["event"]
    )
    reduced = X_train.drop(columns=["scoma", "adlsc"])
    export = pd.concat([reduced, y_train], axis=1)
    if len(reduced.columns) != 28 or export[["time", "event"]].isna().any().any():
        raise ValueError("Unexpected export schema or missing outcomes.")
    export_path = root / "support2_trained.csv"
    export.to_csv(export_path, index=False)

    manifests = []
    for label, rows in [("train", X_train.index), ("test", X_test.index)]:
        manifests.append(pd.DataFrame({
            "id": df.loc[rows, "id"].to_numpy(),
            "source_row_1based": rows.to_numpy() + 1,
            "set": label,
            "split_order": np.arange(1, len(rows) + 1),
        }))
    pd.concat(manifests, ignore_index=True).to_csv(
        root / "support2_split_manifest.csv", index=False
    )
    provenance = {
        "source": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "export_sha256": hashlib.sha256(export_path.read_bytes()).hexdigest(),
        "landmark_days": 3,
        "eligibility": "d.time > 3",
        "time_definition": "d.time - 3",
        "event_definition": "death (1=observed death, 0=censored)",
        "split": {"random_state": 42, "test_size": 0.2, "stratify": "event"},
        "source_patients": len(df),
        "landmark_patients": len(day3),
        "training_patients": len(X_train),
        "test_patients": len(X_test),
        "training_events": int(y_train["event"].sum()),
        "excluded_source_columns": excluded,
        "reduced_feature_exclusions": ["scoma", "adlsc"],
        "predictors": list(reduced.columns),
        "versions": {"pandas": pd.__version__, "numpy": np.__version__,
                     "scikit-learn": sklearn.__version__},
    }
    (root / "support2_export_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(f"Exported {len(export):,} training patients and {len(reduced.columns)} predictors.")
    print(f"Held-out test patients: {len(X_test):,}. Split manifest and provenance saved.")


if __name__ == "__main__":
    main()
