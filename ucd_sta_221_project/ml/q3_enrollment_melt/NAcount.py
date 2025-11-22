import pandas as pd
from pathlib import Path

DATA_PROC = Path("ml/q3_enrollment_melt/processed_data")
MASTER_PATH = DATA_PROC / "melt_panel_master.csv"


def main():
    df = pd.read_csv(MASTER_PATH)
    print("Master shape:", df.shape)
    print("Years:", sorted(df["year"].unique()))

    target_col = "y_high_melt" if "y_high_melt" in df.columns else None
    drop_cols = [
        "melt_rate",
        "melt_count",
        "n_app",
        "n_admit",
        "n_enroll",
        "cc_name",
        "uc_campus",
    ]
    if target_col:
        drop_cols.append(target_col)

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]

    years = df["year"]
    masks = {
        "Train (<=2018)": years <= 2018,
        "Main test (2019)": years == 2019,
        "COVID/post-COVID (>=2020)": years >= 2020,
    }

    for label, m in masks.items():
        X_sub = X[m]
        n_total = X_sub.shape[0]
        if n_total == 0:
            print(f"\n[{label}] 0 rows in this subset, skip.")
            continue
        nan_row_mask = X_sub.isna().any(axis=1)
        n_nan_rows = nan_row_mask.sum()
        n_after = X_sub.dropna().shape[0]

        print(f"\n===== {label} =====")
        print(f"Total rows: {n_total}")
        print(f"Rows with >=1 NaN: {n_nan_rows}")
        print(f"Rows remaining after dropna(): {n_after}")

        na_counts = X_sub.isna().sum()
        na_counts = na_counts[na_counts > 0].sort_values(ascending=False)
        if na_counts.empty:
            print("No NaN in any feature column.")
        else:
            print("\nColumns with NaN (top 15):")
            print(na_counts.head(15))


if __name__ == "__main__":
    main()
