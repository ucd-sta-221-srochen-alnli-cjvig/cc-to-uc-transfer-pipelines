"""
Data split for Q3
- Load the master panel
- Construct high-melt label from melt_rate
- Split into train / main-test / covid-postcovid sets
- Build a preprocessing pipeline for numeric + categorical
"""

from pathlib import Path
from typing import Tuple, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ------------------------------------------------------
DATA_PROC = Path("ml/q3_enrollment_melt/processed_data")
MASTER_PATH = DATA_PROC / "melt_panel_master.csv"
RANDOM_STATE = 42
POS_QUANTILE = 0.75


# ------------------------------------------------------
# 1. Load data + construct high-melt label
# ------------------------------------------------------
def load_data(
    master_path: Path = MASTER_PATH,
    pos_quantile: float = POS_QUANTILE,
) -> Tuple[pd.DataFrame, float]:
    """
    Load data, drop rows with missing melt_rate,
    and construct high-melt binary label based on melt_rate quantile.
    """
    df = pd.read_csv(master_path)
    df = df.dropna(subset=["melt_rate"])

    threshold = df["melt_rate"].quantile(pos_quantile)
    df["y_high_melt"] = (df["melt_rate"] >= threshold).astype(int)

    print(f"High-melt threshold at quantile {pos_quantile:.2f}: {threshold:.3f}")
    print("Overall class balance:")
    print(df["y_high_melt"].value_counts(normalize=True))
    print("Available years:", sorted(df["year"].unique()))

    return df, threshold


# ------------------------------------------------------
# 2. train / main test / covid-postcovid test
# ------------------------------------------------------
def make_splits(
    df: pd.DataFrame,
    train_end_year: int = 2018,
    main_test_year: int = 2019,
    covid_start_year: int = 2020,
):
    """
    Train: year <= train_end_year (pre-COVID)
    Main test: year == main_test_year
    COVID/post-COVID stress test: year >= covid_start_year
    """
    target_col = "y_high_melt"

    # non-feature columns
    drop_cols = [
        target_col,
        "melt_rate",
        "melt_count",
        "n_app",
        "n_admit",
        "n_enroll",
        "cc_name",
        "uc_campus",
    ]

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    y = df[target_col]

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    years = df["year"]

    train_mask = years <= train_end_year
    main_test_mask = years == main_test_year
    covid_mask = years >= covid_start_year

    X_train, y_train = X[train_mask], y[train_mask]
    X_test_main, y_test_main = X[main_test_mask], y[main_test_mask]
    X_test_covid, y_test_covid = X[covid_mask], y[covid_mask]

    print("\n=== Split summary (before dropping NaN) ===")
    print("Train years:", sorted(df.loc[train_mask, "year"].unique()))
    print("Main test years:", sorted(df.loc[main_test_mask, "year"].unique()))
    print("COVID/post-COVID years:", sorted(df.loc[covid_mask, "year"].unique()))
    print(
        "Train size:", X_train.shape,
        " Main test size:", X_test_main.shape,
        " COVID size:", X_test_covid.shape,
    )

    # ---- Drop NaN ----
    def dropna_xy(X_sub, y_sub, label):
        n_before = X_sub.shape[0]
        mask = ~X_sub.isna().any(axis=1)
        X_clean = X_sub.loc[mask].copy()
        y_clean = y_sub.loc[mask].copy()
        n_after = X_clean.shape[0]
        print(
            f"{label}: dropped {n_before - n_after} rows with NaN "
            f"({n_after} remaining)"
        )
        return X_clean, y_clean

    X_train, y_train = dropna_xy(X_train, y_train, "Train")
    X_test_main, y_test_main = dropna_xy(X_test_main, y_test_main, "Main test (2019)")
    X_test_covid, y_test_covid = dropna_xy(
        X_test_covid, y_test_covid, "COVID/post-COVID (>=2020)"
    )

    return (
        X_train,
        y_train,
        X_test_main,
        y_test_main,
        X_test_covid,
        y_test_covid,
        numeric_cols,
        categorical_cols,
    )


# ------------------------------------------------------
# 3. Preprocessing
# ------------------------------------------------------
def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> ColumnTransformer:
    """
    ColumnTransformer：
    - Numeric：median imputation + StandardScaler
    - Categorical：most_frequent imputation + OneHotEncoder
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor


# ------------------------------------------------------
# 4. Convenience wrapper for train_models.py
# ------------------------------------------------------
def get_splits_and_preprocessor():
    """
    One-stop helper for training scripts.

    Returns:
    - X_train, y_train
    - X_test_main, y_test_main
    - X_test_covid, y_test_covid
    - preprocessor
    - threshold (used to define high-melt)
    """
    df, threshold = load_data()
    (
        X_train,
        y_train,
        X_test_main,
        y_test_main,
        X_test_covid,
        y_test_covid,
        numeric_cols,
        categorical_cols,
    ) = make_splits(df)

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    return (
        X_train,
        y_train,
        X_test_main,
        y_test_main,
        X_test_covid,
        y_test_covid,
        preprocessor,
        threshold,
    )