import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from environment import SmartBuildingEnv

print("--- BẮT ĐẦU HUẤN LUYỆN AI QUẢN TRỊ TÒA NHÀ ---")

# 1. CÁC THÀNH PHẦN CỐT LÕI (Từ Tuần 4)
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, next_state, reward, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, next_state, reward, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(tf.keras.Model):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='linear')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)

# 2. HÀM CẬP NHẬT TRỌNG SỐ CHO NÃO BỘ
def update_q_network(brain, target_brain, memory, optimizer, batch_size, gamma):
    """Lấy kinh nghiệm từ quá khứ để tính sai số và cập nhật mạng nơ-ron"""
    if len(memory) < batch_size:
        return 0 # Chưa đủ dữ liệu để học

    # Rút ngẫu nhiên một lô sự kiện từ trí nhớ
    transitions = memory.sample(batch_size)
    states, actions, next_states, rewards, dones = zip(*transitions)

    states = tf.convert_to_tensor(np.array(states), dtype=tf.float32)
    next_states = tf.convert_to_tensor(np.array(next_states), dtype=tf.float32)
    rewards = tf.convert_to_tensor(np.array(rewards), dtype=tf.float32)
    actions = tf.convert_to_tensor(np.array(actions), dtype=tf.int32)
    dones = tf.convert_to_tensor(np.array(dones), dtype=tf.float32)

    # Tính toán sai số Bellman (Bellman Equation)
    with tf.GradientTape() as tape:
        # Lấy giá trị Q dự đoán hiện tại
        q_values = brain(states)
        indices = tf.stack([tf.range(batch_size), actions], axis=1)
        q_action = tf.gather_nd(q_values, indices)
        
        # Dùng mạng Target để ước lượng Q cho tương lai nhằm giữ sự ổn định
        target_q_values = target_brain(next_states)
        max_target_q = tf.reduce_max(target_q_values, axis=1)
        
        # Công thức: Q_mục_tiêu = Phần_thưởng + Gamma * Q_Tương_lai (nếu chưa kết thúc ngày)
        target_q = rewards + gamma * max_target_q * (1.0 - dones)
        
        # Hàm mất mát MSE (Mean Squared Error)
        loss = tf.reduce_mean(tf.square(target_q - q_action))

    # Tối ưu hóa (Giảm loss)
    grads = tape.gradient(loss, brain.trainable_variables)
    optimizer.apply_gradients(zip(grads, brain.trainable_variables))
    
    return loss.numpy()

# 3. CHƯƠNG TRÌNH CHÍNH (MAIN TRAINING LOOP)
if __name__ == "__main__":
    # --- Siêu tham số (Hyperparameters) theo tác giả ---
    NUM_ACTIONS = 24
    GAMMA = 0.9           # Mức độ coi trọng tương lai
    EPSILON = 1.0         # Tỷ lệ thử nghiệm cái mới ban đầu (100%)
    MIN_EPSILON = 0.1     # Tỷ lệ thử nghiệm tối thiểu (10%)
    EPSILON_DECAY = 0.995 # Tốc độ giảm dần sự tò mò để tập trung làm việc hiệu quả
    LEARNING_RATE = 0.001
    MEMORY_CAP = 10000
    BATCH_SIZE = 32
    NUM_EPISODES = 1500     # Chạy 500 kịch bản huấn luyện để AI học được nhiều tình huống khác nhau
    
    # --- Khởi tạo hệ thống ---
    env = SmartBuildingEnv('./data/Cleaned_data_encode.csv', './models/xgb_env_simulator.pkl')
    brain = DQN(NUM_ACTIONS)
    target_brain = DQN(NUM_ACTIONS) # Bản sao giúp AI không bị "tẩu hỏa nhập ma" khi học
    dummy_state = tf.zeros((1, 8))
    brain(dummy_state)
    target_brain(dummy_state)
    optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)
    memory = ReplayBuffer(MEMORY_CAP)
    
    reward_history = []
    
    print("\n[AI] Bắt đầu vào quá trình huấn luyện...")
    for episode in range(NUM_EPISODES):
        state = env.reset(start_index=episode * 24) # Mỗi kịch bản bắt đầu từ 1 ngày khác nhau
        total_reward = 0
        done = False
        
        while not done:
            # Chính sách Epsilon-Greedy: Đôi khi làm bừa, đôi khi dùng não
            if np.random.rand() < EPSILON:
                action = np.random.randint(NUM_ACTIONS) # Đoán bừa để tìm đường mới
            else:
                state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
                q_values = brain(state_tensor)
                action = np.argmax(q_values.numpy())    # Dùng trí khôn
                
            # Thực thi quyết định
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            # Lưu vào trí nhớ và cho AI học
            memory.push(state, action, next_state, reward, done)
            loss = update_q_network(brain, target_brain, memory, optimizer, BATCH_SIZE, GAMMA)
            
            state = next_state
            
            # Giảm tỷ lệ làm bừa xuống
            if EPSILON > MIN_EPSILON:
                EPSILON *= EPSILON_DECAY
                
        # Cuối mỗi ngày, đồng bộ mạng đích (Target Network)
        if episode % 5 == 0:
            target_brain.set_weights(brain.get_weights())
            
        reward_history.append(total_reward)
        print(f"Kịch bản {episode + 1}/{NUM_EPISODES} | Tổng điểm: {total_reward:.2f} | Tỷ lệ thử nghiệm: {EPSILON:.2f}")

    print("\n[AI] Huấn luyện hoàn tất!")

    # Lưu lại toàn bộ nơ-ron thần kinh của AI
    brain.save_weights('./models/dqn_brain.weights.h5')
    print("Đã lưu bộ não AI thành công tại: ./models/dqn_brain.weights.h5")
    
    # Vẽ biểu đồ sự tiến bộ của AI
    plt.plot(reward_history)
    plt.xlabel('Kịch bản (Episode)')
    plt.ylabel('Tổng điểm thưởng (Total Reward)')
    plt.title('Sự tiến bộ của AI qua các ngày')
    plt.savefig('dqn_training_progress.png')
    print("Đã lưu biểu đồ học tập tại: dqn_training_progress.png")
