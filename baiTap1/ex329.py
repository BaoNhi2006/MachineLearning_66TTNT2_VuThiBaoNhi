import numpy as np


class Perceptron:

    def __init__(self, learning_rate=0.1, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None

    # Hàm huấn luyện
    def fit(self, X, y):

        # Khởi tạo trọng số bằng 0
        self.w = np.zeros(X.shape[1])

        # Lặp qua số epoch
        for epoch in range(self.epochs):

            # Duyệt từng mẫu dữ liệu
            for i in range(len(X)):

                # Tính w^T.x
                z = np.dot(self.w, X[i])

                # Dự đoán
                if z >= 0:
                    y_pred = 1
                else:
                    y_pred = -1

                # Nếu dự đoán sai thì cập nhật w
                if y_pred != y[i]:
                    self.w = self.w + self.learning_rate * y[i] * X[i]

    # Hàm dự đoán
    def predict(self, X):

        predictions = []

        for i in range(len(X)):

            # Tính w^T.x
            z = np.dot(self.w, X[i])

            # Dự đoán
            if z >= 0:
                predictions.append(1)
            else:
                predictions.append(-1)

        return np.array(predictions)