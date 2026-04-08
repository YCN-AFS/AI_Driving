import cv2
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

# ==========================================
# 1. IMPORT THƯ VIỆN ĐIỀU KHIỂN & MODEL
# ==========================================
from oneai.robogo.robogo_device_solver import RoboGoDeviceSolver

try:
    from train_model import PilotNet
except ImportError:
    print("❌ LỖI: Không tìm thấy file train_model.py trong cùng thư mục!")
    print("Hãy copy file train_model.py từ máy Pop!_OS sang đây.")
    exit()

# ==========================================
# 2. CẤU HÌNH THÔNG SỐ (HYPERPARAMETERS)
# ==========================================
MAX_ANGLE = 20          # Biên độ vô lăng tối đa (độ)
SPEED = 30              # Tốc độ động cơ
DEADZONE = 1.0          # Dung sai: Góc lệch dưới 1 độ sẽ bị ép đi thẳng để chống lắc
MODEL_PATH = "robogo_pilotnet.pth"

def main():
    # --- KHỞI TẠO AI TRÊN GPU ---
    print("[INFO] Đang kiểm tra sức mạnh phần cứng...")
    # Ưu tiên lấy CUDA (GPU Jetson Nano), nếu không có mới chịu xài CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 🧠 Thiết bị xử lý AI: {device}")

    print("[INFO] Đang nạp mạng Nơ-ron PilotNet...")
    model = PilotNet()
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
    except Exception as e:
        print(f"❌ LỖI NẠP MODEL: {e}")
        exit()

    # Đẩy model lên GPU và chuyển sang chế độ làm việc (không học)
    model.to(device)
    model.eval()

    # Pipeline tiền xử lý ảnh giống hệt lúc Train
    transform = T.Compose([
        T.Resize((66, 200)), # PyTorch Resize dùng (Height, Width)
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print("✅ Đã nạp thành công bộ não AI!")

    # --- KHỞI TẠO ĐỘNG CƠ & CAMERA ---
    print("[INFO] Đang kết nối với động cơ RoboGo...")
    robot = RoboGoDeviceSolver()
    if not robot.load():
        print("❌ LỖI: Không thể kết nối với mạch điều khiển động cơ!")
        exit()

    print("[INFO] Đang khởi động Camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ LỖI: Không tìm thấy Camera!")
        robot.unload()
        exit()

    print("\n==================================================")
    print(" 🚀 HỆ THỐNG TỰ LÁI (AUTOPILOT GPU) SẴN SÀNG")
    print("==================================================")
    print("- Hãy đặt xe vào sa bàn.")
    print("- Bấm phím 'Q' trên cửa sổ Camera để TẮT KHẨN CẤP.")
    print("==================================================\n")

    # --- VÒNG LẶP CHÍNH (AUTOPILOT LOOP) ---
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ LỖI: Tín hiệu camera bị ngắt đột ngột!")
                break

            # 1. TIỀN XỬ LÝ ẢNH
            # Camera đọc BGR, Model học bằng RGB -> Đổi màu
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Đổi sang Tensor, Thêm chiều Batch [1, C, H, W] và đẩy lên GPU
            input_tensor = transform(pil_image).unsqueeze(0).to(device)

            # 2. AI SUY LUẬN (INFERENCE)
            with torch.no_grad():
                pred_angle_norm = model(input_tensor).item()

            # Giới hạn góc chuẩn hóa để tránh AI ngáo sinh ra số quá lớn
            pred_angle_norm = max(-1.0, min(1.0, pred_angle_norm))
            
            # Tính toán góc đánh lái thực tế (Độ)
            steering_angle = pred_angle_norm * MAX_ANGLE

            # 3. RA LỆNH ĐỘNG CƠ (KÈM DEADZONE)
            if steering_angle < -DEADZONE:
                # Rẽ trái (Hàm hãng chỉ nhận số dương nên dùng abs)
                robot.drive_left(abs(steering_angle))
                robot.drive_forward(SPEED)
            elif steering_angle > DEADZONE:
                # Rẽ phải
                robot.drive_right(abs(steering_angle))
                robot.drive_forward(SPEED)
            else:
                # Nếu góc nhỏ hơn Deadzone -> Coi như đi thẳng để xe không bị lạng lách
                robot.servo_comeback_center()
                robot.drive_forward(SPEED)

            # 4. HIỂN THỊ HUD LÊN MÀN HÌNH
            display_frame = cv2.resize(frame, (320, 240))
            
            # Hiển thị text trạng thái
            cv2.putText(display_frame, "AUTOPILOT: ON (GPU)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Angle: {steering_angle:+.1f} deg", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Vẽ thanh vô lăng ảo
            cv2.line(display_frame, (160, 220), (160, 240), (255, 255, 255), 2)
            steer_offset = int(pred_angle_norm * 100)
            cv2.circle(display_frame, (160 + steer_offset, 230), 8, (0, 255, 0), -1)

            cv2.imshow("RoboGo Autodrive", display_frame)

            # Nhấn Q để dừng
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] Đã nhận lệnh thoát (Phím Q)...")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Đã ngắt khẩn cấp bằng Ctrl+C.")
    except Exception as e:
        print(f"\n❌ Đã xảy ra lỗi hệ thống: {e}")
    finally:
        # 5. DỌN DẸP & AN TOÀN (Cực kỳ quan trọng)
        print("[INFO] Đang đóng hệ thống phanh an toàn...")
        robot.drive_stop()
        robot.unload()
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Hoàn tất tắt máy. Chúc bạn một ngày vui vẻ!")

if __name__ == "__main__":
    main()