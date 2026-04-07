import cv2
import os
import time
import csv
import numpy as np

# Sử dụng thư viện điều khiển phần cứng của UBTECH
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

# Tạo file CSV và ghi dòng tiêu đề (Header)
# Nếu file đã có, nó sẽ tạo file mới đè lên để tránh rác data cũ
with open(CSV_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "steering_angle", "speed"])

# ==========================================
# 2. KHỞI TẠO ĐỘNG CƠ VÀ CAMERA
# ==========================================
print("Đang kết nối với động cơ RoboGo...")
robot = RoboGoDeviceSolver()
if not robot.load():
    print("❌ LỖI: Không thể kết nối với động cơ! Vui lòng khởi động lại AI Box.")
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

print("\n==================================================")
print(" 🚗 SẴN SÀNG THU THẬP DỮ LIỆU (BEHAVIORAL CLONING)")
print("==================================================")
print("📌 LƯU Ý QUAN TRỌNG: Bạn PHẢI click chuột vào cửa sổ Camera để nó nhận phím!")
print("🎮 ĐIỀU KHIỂN:")
print("   - Giữ phím 'W': Đi thẳng")
print("   - Giữ phím 'A': Cua trái")
print("   - Giữ phím 'D': Cua phải")
print("   - Buông tay: Xe tự phanh an toàn")
print("   - Phím 'Q': Thoát chương trình và Lưu dữ liệu")
print("==================================================\n")

# ==========================================
# 3. VÒNG LẶP CHÍNH (ĐIỀU KHIỂN & CHỤP ẢNH)
# ==========================================
# ==========================================
# 3. VÒNG LẶP CHÍNH (ĐIỀU KHIỂN & CHỤP ẢNH)
# ==========================================
last_key_time = 0
active_key = 255 # 255 nghĩa là không có phím nào được bấm

try:
    while True:
        ret, frame = cap.read()
        if not ret: 
            print("❌ Lỗi: Bị mất tín hiệu camera!")
            break

        frame = cv2.resize(frame, (320, 240))
        
        # Nhận diện phím bấm
        key = cv2.waitKey(30) & 0xFF
        
        # --- BỘ LỌC CHỐNG GIẬT CỤC KHI GIỮ PHÍM ---
        if key != 255:
            active_key = key
            last_key_time = time.time()
        else:
            # Nếu hệ điều hành bị "nghỉ nhịp", giữ nguyên lệnh cũ trong 0.2 giây
            if time.time() - last_key_time > 0.2:
                active_key = 255
        # ------------------------------------------

        current_angle = 0
        is_moving = False

        # --- LOGIC ĐIỀU KHIỂN ---
        if active_key == ord('q'):
            print("Nhận lệnh thoát từ người dùng...")
            break
            
        elif active_key == ord('w'):
            current_angle = 0
            robot.servo_comeback_center()
            robot.drive_forward(SPEED)
            is_moving = True
            
        elif active_key == ord('a'):
            current_angle = -MAX_ANGLE
            robot.drive_left(MAX_ANGLE)
            robot.drive_forward(SPEED)
            is_moving = True
            
        elif active_key == ord('d'):
            current_angle = MAX_ANGLE
            robot.drive_right(MAX_ANGLE)
            robot.drive_forward(SPEED)
            is_moving = True
            
        else:
            robot.drive_stop()
            is_moving = False

        # --- LOGIC GHI DỮ LIỆU ---
        if is_moving:
            timestamp = str(time.time()).replace('.', '')
            img_filename = f"img_{timestamp}.jpg"
            img_filepath = os.path.join(IMG_DIR, img_filename)
            
            cv2.imwrite(img_filepath, frame)
            
            normalized_angle = current_angle / float(MAX_ANGLE)
            
            with open(CSV_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([img_filename, normalized_angle, SPEED])

        # --- HIỂN THỊ MÀN HÌNH HUD ---
        display_frame = frame.copy()
        color = (0, 255, 0) if is_moving else (0, 0, 255)
        status_text = "RECORDING" if is_moving else "PAUSED"
        
        cv2.putText(display_frame, f"Angle: {current_angle} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(display_frame, f"Status: {status_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.imshow("RoboGo Data Collection - CLICK HERE", display_frame)

except Exception as e:
    print(f"❌ Đã xảy ra lỗi hệ thống: {e}")

finally:
# ... (Phần cuối giữ nguyên y hệt cũ)
    # ==========================================
    # 4. DỌN DẸP AN TOÀN
    # ==========================================
    cap.release()
    cv2.destroyAllWindows()
    robot.servo_comeback_center()
    robot.drive_stop()
    robot.unload()
    
    # Thống kê nhanh dữ liệu đã thu thập
    try:
        num_images = len(os.listdir(IMG_DIR))
        print("\n==================================================")
        print("✅ ĐÃ TẮT ĐỘNG CƠ VÀ CAMERA AN TOÀN!")
        print(f"📁 Tổng số ảnh đã thu thập: {num_images} ảnh.")
        print(f"💾 Dữ liệu được lưu tại: {os.path.abspath(DATA_DIR)}")
        print("🚀 Hãy nén thư mục này, copy sang máy Pop!_OS để chuẩn bị Train AI.")
        print("==================================================")
    except FileNotFoundError:
        print("Chưa có dữ liệu nào được lưu.")