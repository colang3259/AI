from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Tải bộ dữ liệu Iris
iris = load_iris()
X = iris.data
y = iris.target

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Khởi tạo mô hình Rừng ngẫu nhiên
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Huấn luyện mô hình
forest_clf.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = forest_clf.predict(X_test)

# Đánh giá độ chính xác của mô hình
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác của mô hình Rừng ngẫu nhiên: {accuracy:.2f}")
