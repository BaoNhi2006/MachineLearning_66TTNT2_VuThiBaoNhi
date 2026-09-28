from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# 1. Đọc dữ liệu
data = load_breast_cancer()

X = data.data
y = data.target


# 2. Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 3. Chuẩn hóa dữ liệu
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 4. Xây dựng mô hình Perceptron
model = Perceptron(
    max_iter=1000,
    random_state=42
)


# 5. Huấn luyện mô hình
model.fit(X_train, y_train)


# 6. Dự đoán
y_pred = model.predict(X_test)


# 7. Đánh giá mô hình

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)


# 8. In kết quả
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)