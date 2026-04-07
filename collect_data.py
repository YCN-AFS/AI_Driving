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
try:
    while True:
        ret, frame = cap.read()
        if not ret: 
            print("❌ Lỗi: Bị mất tín hiệu camera!")
            break

        # Resize ảnh về 320x240. 
        # (Ảnh nhỏ giúp AI train nhanh hơn và chạy mượt hơn trên ARM64 sau này)
        frame = cv2.resize(frame, (320, 240))
        
        # Nhận diện phím bấm (delay 50ms)
        key = cv2.waitKey(50) & 0xFF
        
        current_angle = 0
        is_moving = False

        # --- LOGIC ĐIỀU KHIỂN ---
        if key == ord('q'):
            print("Nhận lệnh thoát từ người dùng...")
            break
            
        elif key == ord('w'):
            current_angle = 0
            robot.servo_comeback_center()
            robot.drive_forward(SPEED)
            is_moving = True
            
        elif key == ord('a'):
            current_angle = -MAX_ANGLE
            robot.drive_left(MAX_ANGLE)
            robot.drive_forward(SPEED)
            is_moving = True
            
        elif key == ord('d'):
            current_angle = MAX_ANGLE
            robot.drive_right(MAX_ANGLE)
            robot.drive_forward(SPEED)
            is_moving = True
            
        else:
            # Không bấm gì -> Phanh lại
            robot.drive_stop()
            is_moving = False

        # --- LOGIC GHI DỮ LIỆU ---
        # Chỉ chụp ảnh và lưu log khi xe ĐANG DI CHUYỂN
        if is_moving:
            # Đặt tên ảnh theo timestamp để không bao giờ bị trùng
            timestamp = str(time.time()).replace('.', '')
            img_filename = f"img_{timestamp}.jpg"
            img_filepath = os.path.join(IMG_DIR, img_filename)
            
            # Lưu ảnh xuống ổ cứng
            cv2.imwrite(img_filepath, frame)
            
            # Chuẩn hóa góc lái (Normalize) về khoảng [-1.0 đến 1.0]
            # Mạng Nơ-ron (Deep Learning) học cực kỳ nhanh với các số nhỏ trong khoảng này
            normalized_angle = current_angle / float(MAX_ANGLE)
            
            # Ghi thông số vào Excel
            with open(CSV_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([img_filename, normalized_angle, SPEED])

        # --- HIỂN THỊ MÀN HÌNH HUD ---
        display_frame = frame.copy()
        
        # Màu chữ: Xanh lá nếu đang ghi data, Đỏ nếu đang dừng
        color = (0, 255, 0) if is_moving else (0, 0, 255)
        status_text = "RECORDING" if is_moving else "PAUSED"
        
        cv2.putText(display_frame, f"Angle: {current_angle} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(display_frame, f"Status: {status_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
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