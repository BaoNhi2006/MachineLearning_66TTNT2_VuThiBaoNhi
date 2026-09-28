"""
train_regularization.py
------------------------
Khac phuc Overfitting bang REGULARIZATION (Ridge - L2, Lasso - L1).

Y tuong:
- Mo hinh overfit (Polynomial degree=6, khong regularization) co cac he so
  (coefficient) rat lon va bat thuong, vi no co dang "vat lon" de khop
  chinh xac tung diem nhieu trong tap train.
- Regularization them mot so hang "phat" (penalty) vao ham mat mat, ty le
  voi do lon cua he so:
    * Ridge (L2): phat theo tong binh phuong he so  -> keo TAT CA he so
      ve gan 0 (nhung khong bang 0 hoan toan), giup mo hinh "muot" hon.
    * Lasso (L1): phat theo tong tri tuyet doi he so -> co the dua HAN
      mot so he so ve dung bang 0, tuong duong voi tu dong loai bo bot
      dac trung khong quan trong (feature selection tu dong).
- He so regularization (alpha) cang lon thi mo hinh cang bi "ep" don gian.
  alpha qua nho -> van overfit; alpha qua lon -> underfit. Vi vay dung
  Cross-Validation (GridSearchCV) de chon alpha toi uu mot cach khach quan.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "house_prices.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
DEGREE = 6  # giu nguyen do phuc tap cao nhu mo hinh overfit ban dau


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["gia_nha_trieu_vnd"])
    y = df["gia_nha_trieu_vnd"].values
    return X, y


def evaluate(model, X_train, y_train, X_test, y_test):
    y_tr_pred = model.predict(X_train)
    y_te_pred = model.predict(X_test)
    return {
        "train_r2": r2_score(y_train, y_tr_pred),
        "test_r2": r2_score(y_test, y_te_pred),
        "train_mse": mean_squared_error(y_train, y_tr_pred),
        "test_mse": mean_squared_error(y_test, y_te_pred),
    }


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    results = {}

    # -----------------------------------------------------------------
    # (0) MO HINH GOC BI OVERFITTING - lam moc so sanh
    # -----------------------------------------------------------------
    overfit = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=DEGREE, include_bias=False)),
        ("linreg", LinearRegression()),
    ])
    overfit.fit(X_train, y_train)
    results["Overfit\n(khong regularization)"] = evaluate(overfit, X_train, y_train, X_test, y_test)
    overfit_coefs = overfit.named_steps["linreg"].coef_

    # -----------------------------------------------------------------
    # (1) RIDGE (L2) - dung GridSearchCV (Cross-Validation) de chon alpha
    # -----------------------------------------------------------------
    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=DEGREE, include_bias=False)),
        ("ridge", Ridge()),
    ])
    ridge_grid = GridSearchCV(
        ridge_pipe,
        param_grid={"ridge__alpha": [0.1, 1, 10, 50, 100, 300, 500, 1000]},
        cv=kf, scoring="r2", n_jobs=-1,
    )
    ridge_grid.fit(X_train, y_train)
    best_ridge = ridge_grid.best_estimator_
    print(f"[Ridge] alpha tot nhat (chon boi 5-Fold CV) = {ridge_grid.best_params_['ridge__alpha']}")
    results["Ridge (L2)"] = evaluate(best_ridge, X_train, y_train, X_test, y_test)
    ridge_coefs = best_ridge.named_steps["ridge"].coef_

    # -----------------------------------------------------------------
    # (2) LASSO (L1) - dung GridSearchCV de chon alpha
    # -----------------------------------------------------------------
    lasso_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=DEGREE, include_bias=False)),
        ("lasso", Lasso(max_iter=3000, tol=1e-2)),
    ])
    lasso_grid = GridSearchCV(
        lasso_pipe,
        param_grid={"lasso__alpha": [0.5, 1, 5, 10, 30, 50]},
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        scoring="r2", n_jobs=-1,
    )
    lasso_grid.fit(X_train, y_train)
    best_lasso = lasso_grid.best_estimator_
    print(f"[Lasso] alpha tot nhat (chon boi 3-Fold CV) = {lasso_grid.best_params_['lasso__alpha']}")
    results["Lasso (L1)"] = evaluate(best_lasso, X_train, y_train, X_test, y_test)
    lasso_coefs = best_lasso.named_steps["lasso"].coef_
    n_zero = int(np.sum(lasso_coefs == 0))
    n_total = len(lasso_coefs)
    print(f"[Lasso] da dua {n_zero}/{n_total} he so ve dung 0 -> tu dong loai bo dac trung khong quan trong")

    # -----------------------------------------------------------------
    # In bang so sanh
    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"{'Mo hinh':30s}{'Train R2':>12s}{'Test R2':>12s}{'Chenh lech':>14s}")
    print("=" * 78)
    for name, r in results.items():
        gap = r["train_r2"] - r["test_r2"]
        print(f"{name.replace(chr(10), ' '):30s}{r['train_r2']:>12.4f}{r['test_r2']:>12.4f}{gap:>14.4f}")

    pd.DataFrame(results).T.to_csv(os.path.join(RESULTS_DIR, "regularization_summary.csv"))

    # -----------------------------------------------------------------
    # Bieu do 1: so sanh Train/Test R2 - Overfit vs Ridge vs Lasso
    # -----------------------------------------------------------------
    labels = list(results.keys())
    train_vals = [results[k]["train_r2"] for k in labels]
    test_vals = [results[k]["test_r2"] for k in labels]
    x = np.arange(len(labels))
    width = 0.32
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.bar(x - width/2, train_vals, width, label="Train R2", color="tab:blue")
    ax.bar(x + width/2, test_vals, width, label="Test R2", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("R2 score")
    ax.set_title("Khac phuc Overfitting bang Regularization (Ridge / Lasso)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    out1 = os.path.join(RESULTS_DIR, "regularization_before_after.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"\nDa luu bieu do so sanh -> {out1}")

    # -----------------------------------------------------------------
    # Bieu do 2: so sanh do lon he so (coefficient) truoc/sau regularization
    # -> minh hoa truc quan viec regularization "keo" he so ve gan 0
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.sort(np.abs(overfit_coefs))[::-1], label="Overfit (khong regularization)", color="tab:red", alpha=0.8)
    ax.plot(np.sort(np.abs(ridge_coefs))[::-1], label="Ridge (L2)", color="tab:blue", alpha=0.8)
    ax.plot(np.sort(np.abs(lasso_coefs))[::-1], label="Lasso (L1)", color="tab:green", alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Chi so he so (da sap xep giam dan theo do lon)")
    ax.set_ylabel("|He so| (thang log)")
    ax.set_title("Regularization lam giam do lon cac he so cua mo hinh\n"
                 "(Lasso con dua nhieu he so ve dung 0)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    out2 = os.path.join(RESULTS_DIR, "regularization_coefficients.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Da luu bieu do so sanh he so -> {out2}")


if __name__ == "__main__":
    main()
