**🏢 Smart Building Management: XGB-DQN for HVAC & Window Control**  
   
**📖 Giới thiệu (Abstract)**  
Dự án tái triển khai hệ thống Trí tuệ Nhân tạo kết hợp giữa **XGBoost** và  **Deep Q-Network (DQN)** nhằm tối ưu hóa việc điều khiển hệ thống điều hòa (HVAC) và cửa sổ. Mục tiêu cốt lõi là đạt được sự cân bằng hoàn hảo giữa việc duy trì tiện nghi nhiệt (Thermal Comfort) theo chuẩn quốc tế ASHRAE 55 và tối thiểu hóa lượng điện năng tiêu thụ dựa trên hành vi người dùng.  
   
    
**🧠 Kiến trúc Hệ thống**  
Dự án giải quyết bài toán bằng phương pháp tiếp cận hai giai đoạn:  
1. **XGBoost (Surrogate Environment):** Đóng vai trò là "Căn phòng ảo". Mô hình học các quy luật nhiệt động lực học từ bộ dữ liệu lịch sử để dự đoán sự thay đổi nhiệt độ phòng sau mỗi hành động. (RMSE ~ 0.34°C).  
2. **Deep Q-Network (DQN):** Tác tử AI (Agent) học cách ra quyết định chọn 1 trong 24 hành động (nhiệt độ cài đặt, đóng/mở cửa sổ, bật/tắt HVAC) dựa trên không gian trạng thái môi trường và hàm phần thưởng đa mục tiêu.  
   
    
**📂 Cấu trúc thư mục**  
├── environment.py           # Định nghĩa Môi trường RL và hàm Reward (chuẩn ASHRAE)    
 ├── train_env_simulator.py   # Huấn luyện mô hình vật lý giả lập (XGBoost)    
 ├── train_dqn.py             # Khởi tạo kiến trúc mạng nơ-ron và Replay Buffer    
 ├── train_agent.py           # Vòng lặp huấn luyện chính của AI (Agent Training)    
 ├── evaluate_agent.py        # Kịch bản kiểm thử (Stress Test) và so sánh AI vs Baseline    
 ├── .gitignore               # Cấu hình bỏ qua file rác và file dung lượng lớn    
 └── README.md                # Tài liệu dự án    
   
# **⚙️ Hướng dẫn Cài đặt (Installation)**    
 Dự án được thực thi tốt nhất trong môi trường ảo Conda để tránh xung đột thư viện.    
 1. Clone kho lưu trữ này về máy:    
 Bash    
 git clone <link-github-cua-ban> cd Hvac-Window-based-XGB-DQN     
 2. Kích hoạt môi trường Conda (Ví dụ: smart_building):    
 Bash    
 conda activate smart_building    
      
 3. Cài đặt các thư viện phụ thuộc:    
 Bash    
 pip install pandas numpy xgboost tensorflow matplotlib scikit-learn    
      
 # **📥 Tải Dữ liệu & Trọng số (Data & Models)**    
 Để tuân thủ giới hạn dung lượng của GitHub, tập dữ liệu gốc và các file trọng số AI đã được lưu trữ riêng biệt trên Google Drive. Để chạy được dự án, vui lòng thực hiện:    
 4. Tải file dữ liệu data.zip tại đây: [Tải Data](https://drive.google.com/file/d/1g5D1cCQjDc3EZwcGBta7qHy_mhP-ydCs/view?usp=sharing "https://drive.google.com/file/d/1g5D1cCQjDc3EZwcGBta7qHy_mhP-ydCs/view?usp=sharing")    
 5. Tải file trí tuệ nhân tạo models.zip tại đây: [Tải Models](https://drive.google.com/file/d/1mCZ1WJtRtAm3d-4gIb_2a_0xYUZ2ec0z/view?usp=drive_link "https://drive.google.com/file/d/1mCZ1WJtRtAm3d-4gIb_2a_0xYUZ2ec0z/view?usp=drive_link")    
 6. Giải nén 2 file trên và đặt vào thư mục gốc của dự án sao cho xuất hiện 2 thư mục data/ và models/.    
   
# **🚀 Hướng dẫn Sử dụng (How to Run)**    
 Dự án có thể chạy trực tiếp trên terminal với 3 kịch bản chính:    
 **1. Huấn luyện lại Căn phòng ảo (XGBoost):**    
 Bash    
 python train_env_simulator.py    
      
 **2. Huấn luyện lại Tác tử AI (DQN Agent):**    
 Bash    
 python train_agent.py    
      
 *(Lưu ý: Quá trình này sẽ chạy qua 500 episodes và ghi đè file * *dqn_brain.weights.h5* * mới).*    
 **3. Đánh giá Tốt nghiệp (Inference & Evaluation):**    
 Bash    
 python evaluate_agent.py    
      
 *(Script này sẽ bốc thăm ngẫu nhiên một ngày trong tập dữ liệu, nạp bộ não AI lên, tắt chế độ thăm dò (Epsilon=0) và xuất ra biểu đồ * *ai_vs_human_comparison.png* * để so sánh hiệu suất với con người).*    
   
# **👥 Nhóm phát triển (Contributors)**    
 - Nguyễn Thanh Hải    
 - Nguyễn Hải Long    
 - Lê Tuấn Hưng    
      
   
