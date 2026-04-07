import cv2
import os
import time
import csv
import numpy as np

# Sử dụng thư viện điều khiển phần cứng gốc của UBTECH
from oneai.robogo.robogo_device_solver import RoboGoDeviceSolver

# ==========================================
# 1. CẤU HÌNH THƯ MỤC LƯU TRỮ (DATASET)
# ==========================================
DATA_DIR = "my_dataset"
IMG_DIR = os.path.join(DATA_DIR, "images")
CSV_FILE = os.path.join(DATA_DIR, "driving_log.csv")

# Tự động tạo thư mục nếu chưa có
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

# Tạo/Mở file CSV để ghi đè hoặc nối tiếp
with open(CSV_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "steering_angle", "speed"])

# ==========================================
# 2. KHỞI TẠO ĐỘNG CƠ VÀ CAMERA
# ==========================================
print("Đang kết nối với động cơ RoboGo...")
robot = RoboGoDeviceSolver()
if not robot.load():
    print("❌ LỖI: Không thể kết nối với động cơ! Vui lòng kiểm tra lại cáp.")
    exit()

print("Đang khởi động Camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ LỖI: Không thể mở Camera!")
    robot.unload()
    exit()

# Thông số vật lý của xe
SPEED = 30
MAX_ANGLE = 20
STEER_STEP = 4  # Gia tốc vô lăng: Tăng/giảm 4 độ mỗi nhịp giữ phím

current_angle = 0 # Biến lưu trữ góc vô lăng hiện tại

print("\n==================================================")
print(" 🚗 SẴN SÀNG THU THẬP DỮ LIỆU (SMOOTH STEERING)")
print("==================================================")
print("📌 LƯU Ý: Phải CLICK CHUỘT vào cửa sổ Camera để nhận phím!")
print("🎮 ĐIỀU KHIỂN:")
print("   - Giữ 'W': Đi thẳng (Vô lăng tự trả thẳng)")
print("   - Giữ 'A': Cua trái (Vô lăng bẻ dần sang trái)")
print("   - Giữ 'D': Cua phải (Vô lăng bẻ dần sang phải)")
print("   - Buông tay: Xe tự phanh an toàn (Ngừng ghi data)")
print("   - Phím 'Q': Thoát chương trình")
print("==================================================\n")

# ==========================================
# 3. VÒNG LẶP CHÍNH (ĐIỀU KHIỂN & CHỤP ẢNH)
# ==========================================
last_key_time = 0
active_key = 255 # Mã ASCII 255 nghĩa là không bấm gì

try:
    while True:
        ret, frame = cap.read()
        if not ret: 
            print("❌ Lỗi: Bị mất tín hiệu camera!")
            break

        # Resize ảnh về 320x240 để model học nhanh và xe chạy mượt
        frame = cv2.resize(frame, (320, 240))
        
        # Nhận diện phím bấm
        key = cv2.waitKey(30) & 0xFF
        
        # --- BỘ LỌC CHỐNG GIẬT PHÍM (DEBOUNCE) ---
        if key != 255:
            active_key = key
            last_key_time = time.time()
        else:
            # Nếu hệ điều hành lỡ ngắt tín hiệu dưới 0.2s, vẫn giữ lệnh cũ cho xe chạy mượt
            if time.time() - last_key_time > 0.2:
                active_key = 255

        is_moving = False

        # --- LOGIC ĐIỀU KHIỂN VÔ LĂNG MƯỢT MÀ ---
        if active_key == ord('q'):
            print("\nĐã nhận lệnh thoát từ người dùng!")
            break
            
        elif active_key == ord('w'):
            current_angle = 0
            robot.servo_comeback_center()
            robot.drive_forward(SPEED)
            is_moving = True
            
        elif active_key == ord('a'):
            current_angle -= STEER_STEP
            if current_angle < -MAX_ANGLE: 
                current_angle = -MAX_ANGLE
            
            # API của xe nhận số dương cho cả trái và phải, nên dùng abs()
            robot.drive_left(abs(current_angle))
            robot.drive_forward(SPEED)
            is_moving = True
            
        elif active_key == ord('d'):
            current_angle += STEER_STEP
            if current_angle > MAX_ANGLE: 
                current_angle = MAX_ANGLE
                
            robot.drive_right(abs(current_angle))
            robot.drive_forward(SPEED)
            is_moving = True
            
        else:
            # Buông phím -> Xe dừng ngay lập tức
            robot.drive_stop()
            is_moving = False

        # --- LOGIC GHI DỮ LIỆU (CHỈ GHI KHI ĐANG CHẠY) ---
        if is_moving:
            timestamp = str(time.time()).replace('.', '')
            img_filename = f"img_{timestamp}.jpg"
            img_filepath = os.path.join(IMG_DIR, img_filename)
            
            cv2.imwrite(img_filepath, frame)
            
            # Chuẩn hóa góc lái về khoảng [-1.0, 1.0] cho Mạng Nơ-ron
            normalized_angle = current_angle / float(MAX_ANGLE)
            
            with open(CSV_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([img_filename, normalized_angle, SPEED])

        # --- HIỂN THỊ MÀN HÌNH HUD ---
        display_frame = frame.copy()
        
        # Đổi màu hiển thị: Xanh khi đang ghi, Đỏ khi đang dừng
        color = (0, 255, 0) if is_moving else (0, 0, 255)
        status_text = "RECORDING" if is_moving else "PAUSED"
        
        # In thông số lên góc trái màn hình
        cv2.putText(display_frame, f"Angle: {current_angle} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(display_frame, f"Status: {status_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Vẽ một cái vô lăng ảo ở dưới đáy màn hình
        center_x = 160
        steering_offset = int((current_angle / MAX_ANGLE) * 100) # Biến góc lái thành pixel dịch chuyển
        cv2.line(display_frame, (160, 220), (160, 240), (255, 255, 255), 2) # Tâm vạch mốc
        cv2.circle(display_frame, (center_x + steering_offset, 220), 8, color, -1) # Chấm vô lăng
        
        cv2.imshow("RoboGo Data Collection - CLICK HERE TO CONTROL", display_frame)

except Exception as e:
    print(f"❌ Đã xảy ra lỗi hệ thống: {e}")

finally:
    # ==========================================
    # 4. DỌN DẸP AN TOÀN
    # ==========================================
    cap.release()
    cv2.destroyAllWindows()
    robot.servo_comeback_center()
    robot.drive_stop()
    robot.unload()
    
    # Đếm số lượng ảnh đã gom được
    try:
        num_images = len(os.listdir(IMG_DIR))
        print("\n==================================================")
        print("✅ ĐÃ TẮT ĐỘNG CƠ VÀ CAMERA AN TOÀN!")
        print(f"📁 Tổng số ảnh thu thập được: {num_images} ảnh.")
        print(f"💾 File log: {CSV_FILE}")
        print("🚀 Nhớ nén thư mục 'my_dataset' rồi chép sang Pop!_OS nhé!")
        print("==================================================")
    except FileNotFoundError:
        pass