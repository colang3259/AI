import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Tạo dữ liệu mẫu
data = {
    'Diện tích (m2)': [50, 60, 70, 80, 90, 100],
    'Giá (triệu VND)': [300, 400, 500, 600, 700, 800]
}
df = pd.DataFrame(data)

# Chuẩn bị dữ liệu
X = df[['Diện tích (m2)']]
y = df['Giá (triệu VND)']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán và đánh giá mô hình
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'Predicted Prices: {y_pred}')
