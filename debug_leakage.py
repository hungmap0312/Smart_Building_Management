import pandas as pd
import xgboost as xgb
import pickle
import numpy as np

print("--- BẮT ĐẦU ĐIỀU TRA RÒ RỈ DỮ LIỆU (DATA LEAKAGE) ---")

# 1. Đọc dữ liệu
data_path = './data/Cleaned_data_encode.csv'
df = pd.read_csv(data_path)

print("\n[1] KIỂM TRA MỨC ĐỘ TƯƠNG QUAN (CORRELATION)")
# Đã sửa lỗi: Lọc lấy các cột mang giá trị số trước khi tính tương quan
corr_matrix = df.select_dtypes(include=[np.number]).corr()
target_corr = corr_matrix['Target_Temp'].abs().sort_values(ascending=False)
print("Top 5 biến có tương quan mạnh nhất với Target_Temp (1.0 là tương quan tuyệt đối):")
print(target_corr.head(6))

print("\n[2] KIỂM TRA GIẢ THUYẾT QUY LUẬT NỘI SUY KHI TẮT MÁY LẠNH")
ac_off_df = df[df['AC_Status'] == 0]
if not ac_off_df.empty:
    exact_match = (ac_off_df['Target_Temp'].round(1) == ac_off_df['Indoor_Temp'].round(1)).sum()
    total_ac_off = len(ac_off_df)
    print(f"- Số mẫu ghi nhận tắt máy lạnh (AC_Status = 0): {total_ac_off}")
    print(f"- Số mẫu mà Target_Temp BẰNG HỆT Indoor_Temp: {exact_match}")
    print(f"- Tỷ lệ trùng khớp: {(exact_match / total_ac_off) * 100:.2f}%")
else:
    print("- Không có dòng nào trong dữ liệu có AC_Status = 0.")

print("\n[3] KIỂM TRA LỜI KHAI CỦA MÔ HÌNH (FEATURE IMPORTANCE)")
model_path = './models/xgboost_hvac_model.pkl'
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Lấy tầm quan trọng của các đặc trưng
    feature_importances = model.feature_importances_
    features = [
        'Indoor_Temp', 'Indoor_RH', 'Outdoor_Temp', 'Outdoor_RH',
        'Rain', 'Cloud', 'Windspeed', 'Month', 'Hour', 'AC_Status', 'Window_Status'
    ]
    
    importance_df = pd.DataFrame({'Feature': features, 'Importance_Score': feature_importances})
    importance_df = importance_df.sort_values(by='Importance_Score', ascending=False)
    
    print("Trọng số quyết định của các biến đầu vào (càng gần 1.0 càng bị mô hình phụ thuộc):")
    print(importance_df.to_string(index=False))
except FileNotFoundError:
    print(f"- Lỗi: Không tìm thấy mô hình tại {model_path}. Vui lòng chạy lại file train_xgboost_hvac.py trước.")
