import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

print("--- XÂY DỰNG MÔI TRƯỜNG GIẢ LẬP VẬT LÝ BẰNG XGBOOST ---")

# 1. Đọc dữ liệu đã mã hóa
data_path = './data/Cleaned_data_encode.csv'
data = pd.read_csv(data_path)

# 2. Xóa các biến tương lai và biến định danh để làm Đầu vào (Input)
# Tác giả đã loại bỏ Next_Indoor_Temp, Next_Indoor_RH, Date_Time, Study_ID, ID và chính biến mục tiêu
columns_to_drop = ['Next_Indoor_Temp', 'Next_Indoor_RH', 'Date_Time', 'Study_ID', 'Differ_Indoor_Temp', 'ID']

# Dùng errors='ignore' để tránh lỗi nếu một số cột đã bị loại bỏ từ trước trong file csv
X = data.drop(columns=columns_to_drop, errors='ignore')

# Biến mục tiêu (Output): Sự thay đổi nhiệt độ trong nhà ở bước thời gian tiếp theo
y = data['Differ_Indoor_Temp']

# 3. Chia tập dữ liệu (Tác giả dùng random_state=2022)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2022)
print(f"Kích thước tập huấn luyện: {X_train.shape[0]} mẫu.")
print(f"Số lượng đặc trưng đầu vào: {X_train.shape[1]} biến.")

# 4. Khởi tạo mô hình với siêu tham số (Hyperparameters) tối ưu từ bài báo
model_XGB = xgb.XGBRegressor(
    random_state=2000,
    n_jobs=-1,
    max_depth=5,
    learning_rate=0.23474,
    n_estimators=500
)

print("\nĐang huấn luyện XGBoost Simulator (Căn phòng ảo)...")
model_XGB.fit(X_train, y_train)
print("Huấn luyện hoàn tất!")

# 5. Đánh giá độ chính xác của căn phòng ảo
y_pred = model_XGB.predict(X_test)
r2 = r2_score(y_test, y_pred)
# Tính RMSE bằng căn bậc hai của MSE
rmse = (mean_squared_error(y_test, y_pred)) ** 0.5 

print("\n--- KẾT QUẢ KIỂM ĐỊNH MÔI TRƯỜNG GIẢ LẬP ---")
print(f"Độ phù hợp (R2 Score): {r2:.4f}")
print(f"Sai số căn quân phương (RMSE): {rmse:.4f}°C")

# 6. Lưu mô hình giả lập
os.makedirs('./models', exist_ok=True)
model_path = './models/xgb_env_simulator.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model_XGB, f)

print(f"\nĐã lưu Môi trường giả lập thành công tại: {model_path}")
