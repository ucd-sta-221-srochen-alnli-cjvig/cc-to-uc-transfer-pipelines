"""
Main training for Q3
- Load split data
- Fit GLM (logistic + elastic net) and RandomForest baselines
- Evaluate on 2019 and COVID/post-COVID
"""

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from datasplit import (
    get_splits_and_preprocessor,
    DATA_PROC,
    RANDOM_STATE,
)

# ------------------------------------------------------
# 1. GLM (logistic + elastic net)
# ------------------------------------------------------
def fit_glm(X_train, y_train, preprocessor):
    """
    Logistic + elastic net = GLM(L1+L2) baseline.
    """
    base_model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        max_iter=5000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", base_model),
        ]
    )

    param_grid = {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__l1_ratio": [0.1, 0.5, 0.9],
    }

    clf = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    clf.fit(X_train, y_train)
    print("\n[GLM] Best params:", clf.best_params_)
    print("[GLM] Best CV AUC:", clf.best_score_)

    return clf.best_estimator_

# ------------------------------------------------------
# 2. RF classifier baseline
# ------------------------------------------------------
def fit_rf(X_train, y_train, preprocessor):
    """
    RandomForestClassifier as nonlinear baseline.
    """
    base_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", base_model),
        ]
    )

    param_grid = {
        "model__max_depth": [5, 10, None],
        "model__min_samples_leaf": [5, 10, 20],
    }

    clf = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    clf.fit(X_train, y_train)
    print("\n[RF] Best params:", clf.best_params_)
    print("[RF] Best CV AUC:", clf.best_score_)

    return clf.best_estimator_

# ------------------------------------------------------
# 3. Evaluation
# ------------------------------------------------------
def evaluate_model(model, X, y, label="dataset"):
    """
    Print AUC, PR-AUC, confusion matrix, and classification report.
    """
    if X.shape[0] == 0:
        print(f"\n==== {label}: empty, skipped ====")
        return

    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)

    auc = roc_auc_score(y, proba)
    ap = average_precision_score(y, proba)
    print(f"\n==== Performance on {label} ====")
    print(f"AUC:     {auc:.3f}")
    print(f"PR-AUC:  {ap:.3f}")
    print("Confusion matrix:\n", confusion_matrix(y, pred))
    print("Classification report:\n", classification_report(y, pred))

# ------------------------------------------------------
# 4. main
# ------------------------------------------------------
def main():
    (
        X_train,
        y_train,
        X_test,
        y_test,
        X_covid,
        y_covid,
        preprocessor,
        threshold,
    ) = get_splits_and_preprocessor()

    print(f"\nUsing high-melt threshold (quantile): {threshold:.3f}")

    # 2. GLM (elastic net logistic)
    glm_model = fit_glm(X_train, y_train, preprocessor)
    evaluate_model(glm_model, X_test, y_test, label="Main test (2019)")
    evaluate_model(glm_model, X_covid, y_covid, label="COVID/post-COVID (>=2020)")

    # 3. Random Forest
    rf_model = fit_rf(X_train, y_train, preprocessor)
    evaluate_model(rf_model, X_test, y_test, label="Main test (2019)")
    evaluate_model(rf_model, X_covid, y_covid, label="COVID/post-COVID (>=2020)")

    # 4. Save results
    results_main = X_test.copy()
    results_main["y_true"] = y_test.values
    results_main["glm_proba"] = glm_model.predict_proba(X_test)[:, 1]
    results_main["rf_proba"] = rf_model.predict_proba(X_test)[:, 1]
    results_main.to_csv(
        DATA_PROC / "melt_model_main_test_predictions.csv",
        index=False,
    )

    results_covid = X_covid.copy()
    results_covid["y_true"] = y_covid.values
    if X_covid.shape[0] > 0:
        results_covid["glm_proba"] = glm_model.predict_proba(X_covid)[:, 1]
        results_covid["rf_proba"] = rf_model.predict_proba(X_covid)[:, 1]
    results_covid.to_csv(
        DATA_PROC / "melt_model_covid_predictions.csv",
        index=False,
    )

    print("\nSaved predictions for main test and COVID/post-COVID sets.")


if __name__ == "__main__":
    main()