import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from environment import SmartBuildingEnv
from train_agent import DQN # Tái sử dụng class bản vẽ não bộ

print("--- KIỂM TRA TỐT NGHIỆP: AI QUẢN TRỊ TÒA NHÀ ---")

# 1. Khởi tạo môi trường
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

# Dữ liệu Nhiệt độ
ai_indoor_temps = []
human_indoor_temps = []
outdoor_temps = []

# Dữ liệu Hành vi (Thêm mới)
ai_ac_status = []
ai_window_status = []
human_ac_status = []
human_window_status = []

hours = range(24)

print("\nBắt đầu mô phỏng 24 giờ điều khiển...")
for step in range(24):
    # Lấy dữ liệu con người và thời tiết gốc để so sánh
    original_row = env.data.iloc[start_index + step]
    human_indoor_temps.append(original_row['Indoor_Temp'])
    outdoor_temps.append(original_row['Outdoor_Temp'])
    
    # Lưu lại hành động thực tế của con người
    human_ac_status.append(original_row['AC_Status'])
    human_window_status.append(original_row['Window_Status'])
    
    ai_indoor_temps.append(state[0]) # state[0] chính là Indoor_Temp của AI
    
    # AI RA QUYẾT ĐỊNH (EPSILON = 0: Không đoán bừa, chỉ dùng trí khôn 100%)
    state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
    q_values = brain(state_tensor)
    action = np.argmax(q_values.numpy())
    
    # Giải mã hành động của AI (Từ số 0-23 sang trạng thái thiết bị vật lý)
    _, ai_ac, ai_win, _, _ = env.map_action(action)
    ai_ac_status.append(ai_ac)
    ai_window_status.append(ai_win)
    
    # Thực thi vào môi trường
    next_state, reward, done = env.step(action)
    state = next_state

# 4. Trực quan hóa kết quả so sánh (Nâng cấp thành Đồ thị Đa bảng)
# Khởi tạo 3 bảng đồ thị xếp dọc, bảng trên cùng (nhiệt độ) to gấp 3 lần bảng dưới
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1, 1]}, sharex=True)

# --- BẢNG 1: ĐỒ THỊ NHIỆT ĐỘ (Giữ nguyên như cũ) ---
ax1.plot(hours, outdoor_temps, label='Nhiệt độ ngoài trời', color='gray', linestyle='--')
ax1.plot(hours, human_indoor_temps, label='Con người điều khiển', color='red', linewidth=2)
ax1.plot(hours, ai_indoor_temps, label='AI DQN điều khiển', color='blue', linewidth=2.5)

ax1.set_title('So sánh Tổng hợp: Tiện nghi Nhiệt độ và Quyết định Điều khiển', fontsize=14, fontweight='bold')
ax1.set_ylabel('Nhiệt độ (°C)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# --- BẢNG 2: ĐỒ THỊ TRẠNG THÁI MÁY LẠNH (AC) ---
# Sử dụng step-plot để vẽ dạng bậc thang nhị phân vuông vức
ax2.step(hours, human_ac_status, label='AC Con người', color='red', linestyle='--', alpha=0.6, where='mid')
ax2.step(hours, ai_ac_status, label='AC AI', color='blue', linewidth=2, where='mid')

ax2.set_ylabel('Máy Lạnh\n(1=Bật, 0=Tắt)', fontsize=10)
ax2.set_yticks([0, 1])
ax2.grid(True, alpha=0.3)
ax2.legend(loc='center right')

# --- BẢNG 3: ĐỒ THỊ TRẠNG THÁI CỬA SỔ (WINDOW) ---
ax3.step(hours, human_window_status, label='Window Con người', color='red', linestyle='--', alpha=0.6, where='mid')
ax3.step(hours, ai_window_status, label='Window AI', color='blue', linewidth=2, where='mid')

ax3.set_xlabel('Giờ trong ngày', fontsize=12)
ax3.set_ylabel('Cửa Sổ\n(1=Mở, 0=Đóng)', fontsize=10)
ax3.set_yticks([0, 1])
ax3.set_xticks(hours)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='center right')

# Tối ưu khoảng cách và lưu file
plt.tight_layout()
plt.savefig('ai_vs_human_comparison.png')
print("\n--- HOÀN TẤT BÀI THI ---")
print("Đã xuất biểu đồ so sánh thành công tại: ai_vs_human_comparison.png")