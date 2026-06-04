import tensorflow as tf
import numpy as np
import random
from environment import SmartBuildingEnv

print("--- KHỞI TẠO BỘ NÃO DQN ---")

# 1. TRÍ NHỚ CỦA AI (Experience Replay Buffer)
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, next_state, reward, done):
        """Lưu trữ một sự kiện vào trí nhớ"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, next_state, reward, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Lấy ngẫu nhiên một lô sự kiện ra để học"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# 2. MẠNG NƠ-RON CỦA AI (Deep Q-Network)
class DQN(tf.keras.Model):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        # Kiến trúc 2 lớp ẩn, mỗi lớp 64 nơ-ron theo chuẩn của tác giả
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        # Lớp đầu ra tương ứng với 24 hành động
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='linear')

    def call(self, inputs):
        """Truyền dữ liệu tiến qua mạng nơ-ron"""
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)

# Khối lệnh kiểm tra sự hoạt động của Mạng nơ-ron và Trí nhớ
if __name__ == "__main__":
    # Cấu hình cơ bản
    NUM_ACTIONS = 24
    
    # Tạo vỏ não và trí nhớ
    brain = DQN(num_actions=NUM_ACTIONS)
    memory = ReplayBuffer(capacity=1000)
    
    # Kết nối với Môi trường giả lập
    env = SmartBuildingEnv('./data/Cleaned_data_encode.csv', './models/xgb_env_simulator.pkl')
    
    # Bắt đầu thử nghiệm
    state = env.reset()
    
    # AI nhìn vào trạng thái và đoán bừa (do chưa được học)
    # Cần reshape(1, -1) để biến mảng 1D thành mảng 2D (1 mẫu x 8 đặc trưng) cho TensorFlow
    state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
    q_values = brain(state_tensor)
    
    print("\n[AI Vision] Đầu vào (Trạng thái phòng):", state)
    print(f"[AI Brain] Trọng số đánh giá 24 hành động (Q-values chưa huấn luyện):")
    print(q_values.numpy())
    
    # Chọn hành động có điểm cao nhất (Dù đang là đoán bừa)
    chosen_action = np.argmax(q_values.numpy())
    print(f"\n-> AI quyết định chọn Hành động: {chosen_action}")
    
    # Gửi hành động vào môi trường
    next_state, reward, done = env.step(chosen_action)
    
    # Lưu kinh nghiệm vào trí nhớ
    memory.push(state, chosen_action, next_state, reward, done)
    
    print(f"-> Đã lưu sự kiện vào trí nhớ. Số lượng kinh nghiệm hiện tại: {len(memory)}/1000")
