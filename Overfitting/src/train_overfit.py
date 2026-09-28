"""
train_overfit.py
-----------------
Buoc 1: XAY DUNG MO HINH BI OVERFITTING tren bo du lieu gia nha.

Cach co tinh gay overfitting (nhung nguyen nhan kinh dien):
1. Dung Polynomial Features bac CAO (degree=6) -> so luong tham so tang
   theo cap so nhan so voi so mau du lieu (120 mau, ~12 dac trung goc).
2. Khong dung regularization (Linear Regression thuong, khong co L1/L2).
3. Bo du lieu nho (120 mau) trong khi so chieu dac trung sau khi mo rong
   polynomial rat lon -> mo hinh "hoc thuoc long" nhieu cua tap train.
4. Co nhieu dac trung nhieu (nhieu_1..nhieu_5) khong thuc su lien quan
   den gia nha nhung mo hinh bac cao van co tim quan he voi chung.

Ket qua mong doi:
- MSE / R2 tren tap TRAIN rat tot (gan nhu hoan hao).
- MSE / R2 tren tap TEST te hon nhieu -> khoang cach lon giua train va test
  chinh la dau hieu cua overfitting.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "house_prices.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["gia_nha_trieu_vnd"])
    y = df["gia_nha_trieu_vnd"].values
    return X, y


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    degree = 6
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)

    print(f"So dac trung goc: {X_train.shape[1]}")
    print(f"So dac trung sau khi mo rong da thuc bac {degree}: {X_train_poly.shape[1]}")
    print(f"So mau train: {X_train_poly.shape[0]}")

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("\n===== KET QUA MO HINH OVERFITTING (Polynomial degree=6, khong regularization) =====")
    print(f"Train MSE : {train_mse:,.2f}   |  Train R2 : {train_r2:.4f}")
    print(f"Test  MSE : {test_mse:,.2f}   |  Test  R2 : {test_r2:.4f}")
    print(f"Chenh lech R2 (Train - Test) = {train_r2 - test_r2:.4f}  <-- cang lon thi cang overfitting")

    degrees = list(range(1, 9))
    train_scores, test_scores = [], []
    for d in degrees:
        p = PolynomialFeatures(degree=d, include_bias=False)
        Xtr = p.fit_transform(X_train_scaled)
        Xte = p.transform(X_test_scaled)
        m = LinearRegression().fit(Xtr, y_train)
        train_scores.append(r2_score(y_train, m.predict(Xtr)))
        test_scores.append(r2_score(y_test, m.predict(Xte)))

    plt.figure(figsize=(8, 5))
    plt.plot(degrees, train_scores, "o-", label="Train R2")
    plt.plot(degrees, test_scores, "o-", label="Test R2")
    plt.axvline(degree, color="red", linestyle="--", alpha=0.5, label=f"Mo hinh overfit (degree={degree})")
    plt.xlabel("Bac da thuc (do phuc tap mo hinh)")
    plt.ylabel("R2 score")
    plt.title("Overfitting: Train R2 tang nhung Test R2 giam khi do phuc tap tang")
    plt.legend()
    plt.grid(alpha=0.3)
    out_fig = os.path.join(RESULTS_DIR, "overfitting_curve.png")
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"\nDa luu bieu do minh hoa overfitting -> {out_fig}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].scatter(y_train, y_train_pred, alpha=0.6, color="tab:blue")
    axes[0].plot([y.min(), y.max()], [y.min(), y.max()], "k--")
    axes[0].set_title(f"TRAIN (R2={train_r2:.3f})")
    axes[0].set_xlabel("Gia thuc te")
    axes[0].set_ylabel("Gia du doan")

    axes[1].scatter(y_test, y_test_pred, alpha=0.6, color="tab:orange")
    axes[1].plot([y.min(), y.max()], [y.min(), y.max()], "k--")
    axes[1].set_title(f"TEST (R2={test_r2:.3f})")
    axes[1].set_xlabel("Gia thuc te")
    axes[1].set_ylabel("Gia du doan")

    plt.suptitle("Mo hinh OVERFITTING: khop gan hoan hao tren Train nhung lech nhieu tren Test")
    out_fig2 = os.path.join(RESULTS_DIR, "overfit_train_vs_test.png")
    plt.savefig(out_fig2, dpi=150, bbox_inches="tight")
    print(f"Da luu bieu do train vs test -> {out_fig2}")

    return {
        "train_mse": train_mse, "test_mse": test_mse,
        "train_r2": train_r2, "test_r2": test_r2,
    }


if __name__ == "__main__":
    main()
