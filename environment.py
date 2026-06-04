import numpy as np
import pandas as pd
import pickle

class SmartBuildingEnv:
    def __init__(self, data_path, simulator_path):
        """Khởi tạo môi trường với dữ liệu thời tiết và bộ giả lập XGBoost"""
        self.data = pd.read_csv(data_path)
        
        with open(simulator_path, 'rb') as f:
            self.simulator = pickle.load(f)
            
        self.standard = 'ASHRAE'
        self.coords = {
            'ASHRAE': [[17.4, 18.4, 23.4, 24.4], 
                       [23.6, 24.6, 29.6, 30.6], 
                       [10, 30]]
        }
        
        # Danh sách các cột đầu vào cho DQN quan sát (8 đặc trưng)
        self.state_columns = ['Indoor_Temp', 'Indoor_RH', 'Outdoor_Temp', 'Outdoor_RH', 'Rain', 'Cloud', 'Windspeed', 'Hour']
        
        # Danh sách các cột cần loại bỏ khi đưa vào dự đoán XGBoost
        self.xgb_drop_columns = ['Next_Indoor_Temp', 'Next_Indoor_RH', 'Date_Time', 'Study_ID', 'Differ_Indoor_Temp', 'ID']

    def map_action(self, action):
        """Chuyển đổi 1 trong 24 quyết định của AI thành các thiết lập vật lý"""
        action = int(action)
        Target_Temp, AC_Status, Window_Status, CLast_Time, WLast_Time = 0, 0, 0, 0, 0
        
        if action == 0:                 # HÀNH ĐỘNG 0: Không làm gì cả (Tắt máy lạnh, Đóng cửa sổ)
            pass
        elif 0 < action < 12:           # HÀNH ĐỘNG 1 đến 11: Bật Điều hòa, Đóng cửa sổ
            Target_Temp = 19 + action 
            AC_Status = 1
            CLast_Time = 60
        elif action == 12:              # HÀNH ĐỘNG 12: Đóng máy lạnh, Mở cửa sổ (Tận dụng gió trời nếu có thể)
            Window_Status = 1
            WLast_Time = 60
        else:                           # HÀNH ĐỘNG 13 đến 23: Bật Điều hòa, Mở cửa sổ (Tình huống cực xấu, nhưng AI vẫn phải học cách xử lý)
            Target_Temp = action + 7
            AC_Status = 1
            Window_Status = 1
            CLast_Time = 60
            WLast_Time = 60
            
        return Target_Temp, AC_Status, Window_Status, CLast_Time, WLast_Time

    def calculate_reward(self, InT, OutT, action):
        """Hàm phần thưởng: Cân bằng giữa Tiện nghi và Năng lượng"""
        reward = 0
        L11, L12, U12, U11 = self.coords[self.standard][0]
        L21, L22, U22, U21 = self.coords[self.standard][1]
        T1, T2 = self.coords[self.standard][2]
        
        if OutT <= T1:
            L1, L2, U2, U1 = L11, L12, U12, U11
        elif OutT >= T2:
            L1, L2, U2, U1 = L21, L22, U22, U21
        else:
            increase = (OutT - T1) * (L21 - L11) / (T2 - T1)
            L1, L2, U2, U1 = L11 + increase, L12 + increase, U12 + increase, U11 + increase
            
        if L2 <= InT <= U2:
            reward += 0 
        elif InT < L2:
            reward -= (InT - L2) ** 2 
        elif InT > U2:
            reward -= (InT - U2) ** 2 
            
        energy_penalty_base = 60 * 0.87 * 1 
        
        if action == 0 or action == 12:
            reward -= 0 
        elif action > 12:
            reward -= 2 * energy_penalty_base 
        else:
            reward -= energy_penalty_base 
            
        return reward

    def reset(self, start_index=0):
        """Khởi tạo lại môi trường: Đưa AI vào một ngày mới"""
        self.current_step = start_index
        self.done = False
        
        # Lấy dòng dữ liệu tại thời điểm bắt đầu
        self.current_data_row = self.data.iloc[self.current_step].copy()
        
        # Trích xuất State (Trạng thái) để AI nhìn thấy
        self.state = self.current_data_row[self.state_columns].values.astype(np.float32)
        
        return self.state

    def step(self, action):
        """Thực thi hành động, gọi XGBoost tính toán và bước sang giờ tiếp theo"""
        if self.done:
            return self.state, 0, self.done
            
        # 1. Chuyển hành động thành thông số máy móc
        Target_Temp, AC_Status, Window_Status, CLast_Time, WLast_Time = self.map_action(action)
        
        # Cập nhật thông số máy móc vào dữ liệu hiện tại
        self.current_data_row['Target_Temp'] = Target_Temp
        self.current_data_row['AC_Status'] = AC_Status
        self.current_data_row['Window_Status'] = Window_Status
        self.current_data_row['CLast_Time'] = CLast_Time
        self.current_data_row['WLast_Time'] = WLast_Time
        
        # Tính điểm thưởng tại thời điểm hiện tại
        InT = self.current_data_row['Indoor_Temp']
        OutT = self.current_data_row['Outdoor_Temp']
        reward = self.calculate_reward(InT, OutT, action)
        
        # 2. GỌI XGBOOST: Dự đoán sự thay đổi nhiệt độ sau 1 giờ
        xgb_input = self.current_data_row.drop(labels=self.xgb_drop_columns, errors='ignore').to_frame().T
        xgb_input = xgb_input.astype(float)
        differ_temp = self.simulator.predict(xgb_input)[0]
        next_indoor_temp = InT + differ_temp
        
        # 3. Chuyển sang giờ tiếp theo
        self.current_step += 1
        
        # Nếu đã đi hết 1 ngày (24 giờ) hoặc hết dữ liệu, kết thúc kịch bản
        if self.current_step >= len(self.data) - 1 or self.current_step % 24 == 0:
            self.done = True
            
        self.current_data_row = self.data.iloc[self.current_step].copy()
        
        # Ghi đè nhiệt độ phòng thực tế bằng nhiệt độ XGBoost vừa mô phỏng
        self.current_data_row['Indoor_Temp'] = next_indoor_temp 
        
        # Trích xuất State mới
        self.state = self.current_data_row[self.state_columns].values.astype(np.float32)
        
        return self.state, reward, self.done

# Khối lệnh kiểm tra toàn bộ quy trình
if __name__ == "__main__":
    print("--- KIỂM TRA QUY TRÌNH HỌC TĂNG CƯỜNG (RL LOOP) ---")
    
    env = SmartBuildingEnv('./data/Cleaned_data_encode.csv', './models/xgb_env_simulator.pkl')
    
    # 1. Bắt đầu một ngày (Reset)
    state = env.reset(start_index=0)
    print(f"[Reset] Khởi tạo môi trường. Nhiệt độ phòng ban đầu: {state[0]:.2f}°C")
    
    # 2. AI thử nghiệm một hành động (Step)
    action = 5 # Bật máy lạnh
    print(f"-> AI quyết định Hành động: {action}")
    
    next_state, reward, done = env.step(action)
    print(f"[Step] Nhiệt độ phòng giờ tiếp theo (do XGBoost dự đoán): {next_state[0]:.2f}°C")
    print(f"[Step] Điểm thưởng: {reward:.2f} điểm")
    print(f"[Step] Kết thúc ngày?: {done}")
