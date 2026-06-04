import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import os
import pickle

print("--- BẮT ĐẦU HUẤN LUYỆN XGBOOST (HVAC TARGET TEMP - BẢN THỰC CHẤT) ---")

data_path = './data/Cleaned_data_encode.csv'
df = pd.read_csv(data_path)

# ----------------- 3 DÒNG CODE SỬA ĐỔI NẰM Ở ĐÂY -----------------

# 1. Chỉ giữ lại những dữ liệu khi máy lạnh ĐANG BẬT
df_ac_on = df[df['AC_Status'] == 1].copy()
print(f"Số lượng mẫu sau khi lọc bỏ lúc tắt máy lạnh: {len(df_ac_on)} (Giảm từ {len(df)})")

label_column = 'Target_Temp'

# 2. Xóa 'AC_Status' khỏi danh sách đầu vào
feature_columns = [
    'Indoor_Temp', 'Indoor_RH', 'Outdoor_Temp', 'Outdoor_RH',
    'Rain', 'Cloud', 'Windspeed', 'Month', 'Hour', 'Window_Status'
]

# 3. Sử dụng DataFrame đã lọc (df_ac_on) thay vì df gốc
X = df_ac_on[feature_columns]
y = df_ac_on[label_column]

# -----------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Đã chia dữ liệu: {len(X_train)} mẫu huấn luyện, {len(X_test)} mẫu kiểm thử.")

model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=150,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

print("\nĐang huấn luyện mô hình dự đoán nhiệt độ HVAC...")
model.fit(X_train, y_train)
print("Huấn luyện hoàn tất!")

# Đánh giá mô hình
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- KẾT QUẢ ĐÁNH GIÁ (HVAC TARGET TEMP) ---")
print(f"Sai số tuyệt đối trung bình (MAE): {mae:.2f}°C")
print(f"Độ phù hợp (R2 Score): {r2:.2f}")

# Lưu mô hình (Ghi đè lên file cũ ảo tưởng)
model_dir = './models'
model_path = os.path.join(model_dir, 'xgboost_hvac_model.pkl')

with open(model_path, 'wb') as f:
    pickle.dump(model, f)
    
print(f"\nĐã lưu mô hình THỰC CHẤT thành công tại: {model_path}")