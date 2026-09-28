# Overfitting — Du bao gia nha

Bai tap: Xay dung mo hinh du bao gia nha, co tinh lam cho mo hinh **overfitting**,
sau do ap dung cac ky thuat da hoc de **khac phuc overfitting**.

## 1. Cau truc thu muc

```
Overfitting/
├── data/
│   ├── generate_data.py     # sinh bo du lieu gia lap gia nha
│   └── house_prices.csv     # du lieu da sinh (120 mau, 12 dac trung + gia nha)
├── src/
│   ├── train_overfit.py         # Buoc 1: xay mo hinh bi overfitting
│   └── train_regularization.py  # Khac phuc rieng bang Regularization (Ridge & Lasso)
├── results/                 # cac bieu do & bang ket qua (tu sinh khi chay code)
├── requirements.txt
└── README.md
```

## 2. Bo du lieu

Du lieu gia nha duoc **sinh gia lap** (`data/generate_data.py`) voi:
- 12 dac trung dau vao: dien tich, so phong ngu, so phong tam, tuoi nha,
  khoang cach toi trung tam, diem an ninh, so tang, va **5 dac trung nhieu**
  (`nhieu_1`...`nhieu_5`) khong thuc su lien quan toi gia nha.
- Chi **120 mau** — co tinh de it, vi tap du lieu nho la mot trong nhung
  nguyen nhan pho bien gay overfitting.
- Gia nha = ham (gan) tuyen tinh cua cac dac trung that + nhieu ngau nhien.

Chay lai de tai tao du lieu (khong bat buoc, file csv da co san):
```bash
python3 data/generate_data.py
```

## 3. Buoc 1 — Lam mo hinh bi Overfitting

File: `src/train_overfit.py`

Cach co tinh gay overfitting:
- Mo rong dac trung bang `PolynomialFeatures(degree=6)` → tu 12 dac trung goc
  thanh **18.563 dac trung**, trong khi chi co 90 mau train.
- Dung `LinearRegression` thuong, **khong co regularization**.

Chay:
```bash
python3 src/train_overfit.py
```

Ket qua thu duoc:

| Tap du lieu | R2 | MSE |
|---|---|---|
| Train | **1.0000** | ~0 |
| Test  | **-0.0439** | 279,921.61 |

→ Mo hinh khop **hoan hao tuyet doi** tren tap train nhung gan nhu vo dung
tren tap test (R2 am nghia la du doan con te hon ca viec lay trung binh).
Day la dau hieu overfitting dien hinh: mo hinh "hoc thuoc long" nhieu cua tap
train thay vi hoc quy luat tong quat.

Bieu do sinh ra trong `results/`:
- `overfitting_curve.png`: Train R2 luon ~1 trong khi Test R2 dao dong manh
  va giam khi bac da thuc tang.
- `overfit_train_vs_test.png`: so sanh truc quan du doan vs thuc te tren
  train (khop hoan hao) va test (lech nhieu).

## 4. Buoc 2 — Khac phuc Overfitting

## Khac phuc bang REGULARIZATION (Ridge & Lasso) — chi tiet rieng

File: `src/train_regularization.py`

Day la phien ban tap trung **chi vao ky thuat Regularization**, theo dung
yeu cau cua bai tap. Van giu nguyen do phuc tap cao (Polynomial degree=6)
nhu mo hinh overfit ban dau, nhung them so hang phat (penalty) vao ham mat
mat de ep cac he so nho lai:

- **Ridge (L2)**: phat theo tong binh phuong he so → keo TAT CA he so ve
  gan 0 (nhung khong ve dung 0), giup duong du doan "muot" hon, bot nhay
  cam voi nhieu.
- **Lasso (L1)**: phat theo tong tri tuyet doi he so → co the dua HAN mot
  so he so ve DUNG BANG 0, tuong duong voi tu dong loai bo dac trung khong
  quan trong (feature selection tu dong).
- He so `alpha` (muc do phat) duoc chon bang **Cross-Validation
  (GridSearchCV)** de dam bao khach quan, khong "doan mo".

Chay:
```bash
python3 src/train_regularization.py
```

### Ket qua

| Mo hinh | Train R2 | Test R2 | Chenh lech |
|---|---|---|---|
| Overfit (khong regularization) | 1.0000 | -0.0439 | 1.0439 |
| Ridge (L2), alpha=1000 (chon boi CV) | 0.9896 | 0.1238 | 0.8658 |
| **Lasso (L1), alpha=0.5 (chon boi CV)** | **0.9999** | **0.6158** | **0.3841** |

Lasso da tu dong dua **15.166 / 18.563 he so ve dung 0** — nghia la no tu
loai bo hon 80% dac trung khong that su can thiet (bao gom cac dac trung
nhieu va cac to hop da thuc thua), giu lai chi nhung dac trung thuc su co
anh huong den gia nha.

**Nhan xet:**
- Ridge cai thien nhe (Test R2 tu -0.04 len 0.12) vi no chi lam "yeu" cac
  he so chu khong loai bo hoan toan — voi so luong dac trung qua lon
  (18.563) so voi so mau (90), Ridge mot minh chua du manh.
- **Lasso hieu qua ro ret hon** (Test R2 len 0.62) nho kha nang tu dong
  loai bo dac trung — day chinh la diem manh cua L1 regularization khi
  du lieu co nhieu dac trung du thua/nhieu.

Bieu do sinh ra trong `results/`:
- `regularization_before_after.png`: so sanh Train/Test R2 giua Overfit,
  Ridge, Lasso.
- `regularization_coefficients.png`: so sanh do lon cac he so (thang log)
  truoc va sau regularization — thay ro Lasso dua rat nhieu he so ve dung 0.
- `regularization_summary.csv`: bang so lieu chi tiet.

## 5. Cach chay toan bo

```bash
pip install -r requirements.txt
python3 data/generate_data.py         # (tuy chon) tai tao du lieu
python3 src/train_overfit.py          # xem mo hinh bi overfitting

python3 src/train_regularization.py   # khac phuc bang Regularization (Ridge & Lasso)
```

## 6. Tom tat kien thuc ap dung

- Nguyen nhan gay overfitting: mo hinh qua phuc tap so voi du lieu, du lieu
  it, co nhieu dac trung nhieu/du thua, khong co regularization.
- Cach khac phuc da dung: giam do phuc tap mo hinh, regularization (L1/L2),
- Cach danh gia overfitting: so sanh **R2/MSE tren train vs test**, khoang
  cach cang lon thi overfitting cang nang.

## 7. Dua len Git ca nhan (repo ten "Overfitting")

```bash
cd Overfitting
git init
git add .
git commit -m "Bai tap overfitting: du bao gia nha"
git branch -M main
git remote add origin https://github.com/<ten-user-cua-ban>/Overfitting.git
git push -u origin main
```
