Gesture Control System 🖐️💻
​Dự án này cho phép bạn điều khiển các chức năng của máy tính (như chuột, âm lượng, hoặc phím tắt) thông qua cử chỉ tay từ Webcam. Sử dụng sức mạnh của MediaPipe để nhận diện bàn tay và OpenCV để xử lý hình ảnh.
​🚀 Tính năng chính
​Điều khiển chuột: Di chuyển con trỏ chuột bằng ngón trỏ.
​Click chuột: Thực hiện click trái/phải bằng cách chụm các đầu ngón tay.
​Điều khiển âm lượng: Tăng/giảm âm lượng hệ thống bằng khoảng cách giữa ngón cái và ngón trỏ.
​Phím tắt: Thực hiện các thao tác như cuộn trang hoặc chuyển tab.
​🛠️ Cài đặt (Python 3.11.x)
1. Clone repository:
git clone https://github.com/tnamr00a-cmd/gesture-ctrl.git
cd gesture-ctrl
2. Tải thư viện cần thiết:
pip install -r requirements.txt
📖 Cách sử dụng với app.py
​File app.py là trung tâm điều khiển của ứng dụng. Dưới đây là giải thích chi tiết cơ chế hoạt động:
​1. Khởi tạo luồng (Initialization)
​Khi bạn chạy python app.py, chương trình sẽ thực hiện:
​Mở Webcam thông qua cv2.VideoCapture(0).
​Khởi tạo model nhận diện bàn tay của MediaPipe (mp.solutions.hands).
​Thiết lập các thông số về độ nhạy và màn hình thông qua pyautogui.
​2. Vòng lặp xử lý (The Main Loop)
​Trong file app.py, mã nguồn sẽ lặp lại liên tục các bước:
​Đọc khung hình: Lấy dữ liệu từ camera và lật hình ảnh (flip) để tạo hiệu ứng soi gương.
​Nhận diện bàn tay: MediaPipe sẽ tìm 21 điểm mốc (landmarks) trên bàn tay của bạn.
​Trích xuất tọa độ: Lấy tọa độ (x, y) của các đầu ngón tay (thường là Landmark 8 cho ngón trỏ và Landmark 4 cho ngón cái).
​3. Logic điều khiển (Logic Mapping) [Bấm "M" để bật/tắt tính năng này]
​Di chuyển chuột: Hệ thống lấy tọa độ của ngón trỏ và ánh xạ (map) nó từ kích thước cửa sổ webcam sang độ phân giải toàn màn hình của bạn.
​Thực hiện Click: Nếu khoảng cách giữa ngón trỏ và ngón giữa nhỏ hơn một ngưỡng (threshold) nhất định, pyautogui.click() sẽ được gọi.
​Điều khiển âm lượng: Tính khoảng cách giữa ngón cái và ngón trỏ. Khoảng cách càng lớn, âm lượng càng tăng (sử dụng thư viện pycaw hoặc tương đương).
​4. Thoát ứng dụng
​Để dừng chương trình, bạn chỉ cần nhấn phím 'Esc' trên bàn phím khi cửa sổ Webcam đang hoạt động.
