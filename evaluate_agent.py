import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from environment import SmartBuildingEnv
from train_agent import DQN # Tái sử dụng class bản vẽ não bộ

print("--- KIỂM TRA TỐT NGHIỆP: AI QUẢN TRỊ TÒA NHÀ ---")

# 1. Khởi tạo môi trường và lấy 1 ngày bất kỳ (ví dụ: ngày bắt đầu từ dòng 0)
env = SmartBuildingEnv('./data/Cleaned_data_encode.csv', './models/xgb_env_simulator.pkl')

# Lọc ra tất cả các dòng là 0:00 giờ sáng (Bắt đầu một ngày mới)
start_of_day_indices = env.data[env.data['Hour'] == 0].index.tolist()

# Đảm bảo ngày đó không bị cụt (còn đủ ít nhất 24 giờ phía sau để vẽ biểu đồ)
valid_start_indices = [idx for idx in start_of_day_indices if idx + 24 <= len(env.data)]

# Bốc thăm ngẫu nhiên 1 vị trí hợp lệ
start_index = random.choice(valid_start_indices)

# In ra thông tin để bạn kiểm tra
thong_tin_ngay = env.data.iloc[start_index]
print(f"[Hệ thống] Đang bốc thăm ngày thi ngẫu nhiên...")
print(f"[Hệ thống] => Đã chọn Dòng thứ {start_index} | Tương ứng với: Tháng {int(thong_tin_ngay['Month'])}")

NUM_ACTIONS = 24

# 2. Khởi tạo cái vỏ não và TẢI TRÍ NHỚ ĐÃ HUẤN LUYỆN LÊN
brain = DQN(NUM_ACTIONS)
dummy_state = tf.zeros((1, 8))
brain(dummy_state) # Chạy mồi để tạo cấu trúc

print("[Hệ thống] Đang tải bộ nhớ từ dqn_brain.weights.h5...")
try:
    brain.load_weights('./models/dqn_brain.weights.h5')
    print("[Hệ thống] Tải bộ nhớ THÀNH CÔNG! AI đã sẵn sàng.")
except Exception as e:
    print(f"[Lỗi] Không tìm thấy file não bộ. Vui lòng chạy train_agent.py trước. Chi tiết: {e}")
    exit()

# 3. Kịch bản đánh giá (1 Ngày = 24 Giờ)
state = env.reset(start_index=start_index)
done = False

# Dữ liệu để vẽ biểu đồ
ai_indoor_temps = []
human_indoor_temps = []
outdoor_temps = []
hours = range(24)

print("\nBắt đầu mô phỏng 24 giờ điều khiển...")
for step in range(24):
    # Lấy dữ liệu con người và thời tiết gốc để so sánh
    original_row = env.data.iloc[start_index + step]
    human_indoor_temps.append(original_row['Indoor_Temp'])
    outdoor_temps.append(original_row['Outdoor_Temp'])
    ai_indoor_temps.append(state[0]) # state[0] chính là Indoor_Temp của AI
    
    # AI RA QUYẾT ĐỊNH (EPSILON = 0: Không đoán bừa, chỉ dùng trí khôn 100%)
    state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
    q_values = brain(state_tensor)
    action = np.argmax(q_values.numpy())
    
    # Thực thi vào môi trường
    next_state, reward, done = env.step(action)
    state = next_state

# 4. Trực quan hóa kết quả so sánh
plt.figure(figsize=(12, 6))

# Vẽ đường thời tiết ngoài trời (Tham chiếu)
plt.plot(hours, outdoor_temps, label='Nhiệt độ ngoài trời (Tham chiếu)', color='gray', linestyle='--')

# Vẽ đường con người điều khiển
plt.plot(hours, human_indoor_temps, label='Con người điều khiển (Thực tế)', color='red', linewidth=2)

# Vẽ đường AI điều khiển
plt.plot(hours, ai_indoor_temps, label='AI DQN điều khiển', color='blue', linewidth=2.5)

# Cấu hình biểu đồ
plt.title('So sánh Tiện nghi nhiệt: Con Người vs. AI (1 Ngày mô phỏng)', fontsize=14)
plt.xlabel('Giờ trong ngày', fontsize=12)
plt.ylabel('Nhiệt độ (°C)', fontsize=12)
plt.xticks(hours)
plt.grid(True, alpha=0.3)
plt.legend()

# Lưu kết quả
plt.savefig('ai_vs_human_comparison.png')
print("\n--- HOÀN TẤT BÀI THI ---")
print("Đã xuất biểu đồ so sánh thành công tại: ai_vs_human_comparison.png")
