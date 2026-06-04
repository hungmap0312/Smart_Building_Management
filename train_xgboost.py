import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
import pickle

print("--- BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN XGBOOST ---")

data_path = './data/Cleaned_data_encode.csv'
df = pd.read_csv(data_path)
print(f"Đã tải dữ liệu thành công! Kích thước dữ liệu: {df.shape}")

# 1. Khai báo Nhãn (Mục tiêu dự đoán)
label_column = 'Window_Status' 

# 2. Lựa chọn Đặc trưng (Feature Selection)
# Chỉ chọn các thông số môi trường hiện tại và thời gian, tuyệt đối không dùng dữ liệu tương lai
feature_columns = [
    'Indoor_Temp', 'Indoor_RH',    # Nhiệt độ và độ ẩm trong nhà
    'Outdoor_Temp', 'Outdoor_RH',  # Nhiệt độ và độ ẩm ngoài trời
    'Rain', 'Cloud', 'Windspeed',  # Thời tiết
    'Month', 'Hour',               # Thời gian (ảnh hưởng thói quen sinh hoạt)
    'AC_Status'                    # Trạng thái máy lạnh hiện tại
]

X = df[feature_columns]
y = df[label_column]

# 3. Chia tập dữ liệu (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Đã chia dữ liệu: {len(X_train)} mẫu huấn luyện, {len(X_test)} mẫu kiểm thử.")
print(f"Các đặc trưng đầu vào (Input): {feature_columns}")

# 4. Cấu hình và Huấn luyện mô hình
model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=150,      # Tăng số cây quyết định lên một chút
    learning_rate=0.05,    
    max_depth=6,           
    random_state=42,
    eval_metric='logloss'
)

print("\nĐang huấn luyện mô hình (có thể mất vài giây)...")
model.fit(X_train, y_train)
print("Huấn luyện hoàn tất!")

# 5. Đánh giá mô hình
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n--- KẾT QUẢ ĐÁNH GIÁ (WINDOW STATUS) ---")
print(f"Độ chính xác (Accuracy): {accuracy * 100:.2f}%")
print("\nBáo cáo chi tiết:")
print(classification_report(y_test, y_pred))

# 6. Lưu mô hình
model_dir = './models'
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'xgboost_window_model.pkl')

with open(model_path, 'wb') as f:
    pickle.dump(model, f)
    
print(f"\nĐã lưu mô hình thành công tại: {model_path}")
