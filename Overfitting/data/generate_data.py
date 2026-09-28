"""
generate_data.py
-----------------
Sinh bo du lieu gia lap "Du bao gia nha" (House Price Prediction).

Muc dich: tao du lieu THUC TE (co nhieu, co dac trung du thua/nhieu, so luong
mau khong qua lon) de khi xay mo hinh phuc tap, mo hinh de bi OVERFITTING.

Cac dac trung:
- dien_tich       : dien tich nha (m2)          -> co anh huong that den gia
- so_phong_ngu    : so phong ngu                -> co anh huong that
- so_phong_tam    : so phong tam                -> co anh huong that
- tuoi_nha        : tuoi cua can nha (nam)      -> co anh huong that (nghich)
- khoang_cach_tt  : khoang cach den trung tam (km) -> co anh huong that (nghich)
- diem_an_ninh    : diem an ninh khu vuc (0-10) -> co anh huong that
- so_tang         : so tang                      -> anh huong nhe
- nhieu_1..nhieu_5: cac dac trung NHIEU, khong that su lien quan den gia nha
                    (them vao de mo phong du lieu thuc te va lam mo hinh de
                    "hoc vet" neu qua phuc tap)

Gia nha = ham phi tuyen cua cac dac trung that + nhieu ngau nhien.
"""

import numpy as np
import pandas as pd
import os

RANDOM_STATE = 42
N_SAMPLES = 120  # so mau CO TINH de nho -> it du lieu la mot nguyen nhan gay overfitting


def generate_house_price_data(n_samples: int = N_SAMPLES, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.RandomState(random_state)

    dien_tich = rng.normal(90, 30, n_samples).clip(25, 250)
    so_phong_ngu = rng.randint(1, 6, n_samples)
    so_phong_tam = rng.randint(1, 4, n_samples)
    tuoi_nha = rng.randint(0, 40, n_samples)
    khoang_cach_tt = rng.uniform(0.5, 25, n_samples)
    diem_an_ninh = rng.uniform(3, 10, n_samples)
    so_tang = rng.randint(1, 4, n_samples)

    nhieu_1 = rng.normal(0, 1, n_samples)
    nhieu_2 = rng.normal(0, 1, n_samples)
    nhieu_3 = rng.normal(0, 1, n_samples)
    nhieu_4 = rng.normal(0, 1, n_samples)
    nhieu_5 = rng.normal(0, 1, n_samples)

    gia = (
        15 * dien_tich
        + 80 * so_phong_ngu
        + 60 * so_phong_tam
        - 5 * tuoi_nha
        - 25 * khoang_cach_tt
        + 40 * diem_an_ninh
        + 10 * so_tang
        + 0.05 * dien_tich * diem_an_ninh
        + rng.normal(0, 250, n_samples)
    )
    gia = gia.clip(min=200)

    df = pd.DataFrame({
        "dien_tich": dien_tich.round(1),
        "so_phong_ngu": so_phong_ngu,
        "so_phong_tam": so_phong_tam,
        "tuoi_nha": tuoi_nha,
        "khoang_cach_tt": khoang_cach_tt.round(2),
        "diem_an_ninh": diem_an_ninh.round(2),
        "so_tang": so_tang,
        "nhieu_1": nhieu_1.round(3),
        "nhieu_2": nhieu_2.round(3),
        "nhieu_3": nhieu_3.round(3),
        "nhieu_4": nhieu_4.round(3),
        "nhieu_5": nhieu_5.round(3),
        "gia_nha_trieu_vnd": gia.round(1),
    })
    return df


if __name__ == "__main__":
    df = generate_house_price_data()
    out_path = os.path.join(os.path.dirname(__file__), "house_prices.csv")
    df.to_csv(out_path, index=False)
    print(f"Da sinh {len(df)} mau du lieu -> {out_path}")
    print(df.head())
