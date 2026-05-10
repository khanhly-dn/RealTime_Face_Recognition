# 🎭 RealTime Face Recognition

<p align="center">
  <img src="https://img.shields.io/badge/Platform-Python%203.x-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Library-OpenCV-green?style=for-the-badge&logo=opencv" />
  <img src="https://img.shields.io/badge/Algorithm-LBPH-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Detection-Haar%20Cascade-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
</p>

<p align="center">
  Hệ thống <strong>nhận diện khuôn mặt thời gian thực</strong> sử dụng <strong>OpenCV + LBPH</strong>,  
  tích hợp thu thập dữ liệu, huấn luyện mô hình và nhận diện qua webcam với giao diện trực quan.
</p>

---

## 📌 Giới thiệu

**RealTime Face Recognition** là một hệ thống nhận diện khuôn mặt hoàn chỉnh, được xây dựng thuần bằng Python và OpenCV, không yêu cầu GPU hay deep learning framework phức tạp. Hệ thống hoạt động theo pipeline 3 bước rõ ràng:

- 📸 **Thu thập ảnh** khuôn mặt qua webcam tự động
- 🧠 **Huấn luyện mô hình** LBPH học đặc điểm từng người
- 🎯 **Nhận diện realtime** với hiển thị tên, độ tin cậy và logging tự động

Phù hợp làm dự án **portfolio / CV**, nghiên cứu học thuật, hoặc tích hợp vào hệ thống kiểm soát ra vào.

---

## ⚙️ Chức năng chính

- **Thu thập dữ liệu tự động** – Chụp 100 ảnh khuôn mặt/người qua webcam, phát hiện bằng Haar Cascade
- **Hỗ trợ nhiều người** – Thêm bao nhiêu người cũng được, mỗi người một thư mục riêng trong `dataset/`
- **Huấn luyện LBPH** – Thuật toán Local Binary Patterns Histograms, nhẹ, nhanh, chính xác cao
- **Nhận diện realtime** – Nhận diện liên tục qua webcam, hiển thị tên + độ tin cậy trực tiếp trên khung hình
- **Phân biệt người lạ** – Tự động phân loại "Nguoi la" nếu không vượt ngưỡng tin cậy
- **Lưu ảnh tự động** – Chụp và lưu ảnh người lạ mỗi 5 giây vào thư mục `captures/`
- **Ghi log chi tiết** – Lưu toàn bộ lịch sử nhận diện theo ngày vào thư mục `logs/`
- **FPS counter** – Hiển thị tốc độ xử lý thời gian thực trên màn hình

---

## 🗂️ Cấu trúc dự án

```
RealTime_Face_Recognition/
│
├── collect_faces.py       # Bước 1: Thu thập ảnh khuôn mặt
├── train.py               # Bước 2: Huấn luyện mô hình LBPH
├── recognize.py           # Bước 3: Nhận diện realtime qua webcam
│
├── dataset/               # Tự tạo sau khi chạy collect_faces.py
│   ├── khanh/             # Thư mục ảnh của từng người
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   └── ... (100 ảnh)
│   └── hoang/
│       └── ...
│
├── trainer/               # Tự tạo sau khi chạy train.py
│   ├── model.yml          # File mô hình LBPH đã huấn luyện
│   └── labels.txt         # Ánh xạ ID → Tên người
│
├── captures/              # Tự tạo khi chạy recognize.py
│   ├── nguoi_la_*.jpg     # Ảnh người lạ bị phát hiện
│   └── manual_*.jpg       # Ảnh chụp thủ công bằng phím [S]
│
└── logs/                  # Tự tạo khi chạy recognize.py
    └── log_YYYYMMDD.txt   # Log nhận diện theo ngày
```

---

## 🧩 Sơ đồ hoạt động

<p align="center">
  <img width="750" alt="Sơ đồ hoạt động" src="https://github.com/khanhly-dn/RealTime_Face_Recognition/blob/main/SDHD.png?raw=true" />
</p>

```
[Bước 1] collect_faces.py
  Nhập tên → Bật webcam → Haar Cascade phát hiện mặt
  → Tự động chụp 100 ảnh grayscale → Lưu vào dataset/TênNgười/

[Bước 2] train.py
  Đọc toàn bộ ảnh từ dataset/ → Gán ID số cho từng người
  → Train thuật toán LBPH → Lưu model.yml + labels.txt

[Bước 3] recognize.py
  Load mô hình → Bật webcam realtime
  → Phát hiện mặt → Dự đoán ID + độ tin cậy
      ├── ≥ 60% → Hiển thị tên + khung XANH
      └── < 60% → "Nguoi la" + khung CAM + tự lưu ảnh + ghi log
```

---

## 🛠️ Yêu cầu hệ thống

| Thành phần | Yêu cầu |
|---|---|
| **Ngôn ngữ** | Python 3.7+ |
| **Webcam** | Bất kỳ webcam USB hoặc tích hợp |
| **OS** | Windows / Linux / macOS |
| **RAM** | Tối thiểu 4GB |
| **GPU** | Không cần (chạy trên CPU) |

---

## 📦 Cài đặt

**1. Clone repository**
```bash
git clone https://github.com/khanhly-dn/RealTime_Face_Recognition.git
cd RealTime_Face_Recognition
```

**2. Cài đặt thư viện**
```bash
pip install opencv-python opencv-contrib-python pillow numpy
```

> ⚠️ Lưu ý: Phải cài `opencv-contrib-python` (không phải chỉ `opencv-python`) để có module `cv2.face.LBPHFaceRecognizer`.

---

## 🚀 Hướng dẫn sử dụng

### Bước 1 – Thu thập ảnh khuôn mặt

```bash
python collect_faces.py
```

- Nhập tên người cần thu thập (ví dụ: `khanh`)
- Nhìn thẳng vào webcam, hệ thống tự động chụp **100 ảnh**
- Ảnh được lưu vào `dataset/khanh/1.jpg` ... `100.jpg`
- Nhấn **[Q]** để dừng sớm nếu muốn

> 💡 Để thêm nhiều người, chạy lại bước này và nhập tên khác.

---

### Bước 2 – Huấn luyện mô hình

```bash
python train.py
```

- Đọc toàn bộ ảnh từ `dataset/`
- Gán ID số: người đầu tiên = 0, người thứ hai = 1, ...
- Huấn luyện thuật toán **LBPH (Local Binary Patterns Histograms)**
- Lưu mô hình vào `trainer/model.yml` và nhãn vào `trainer/labels.txt`

---

### Bước 3 – Nhận diện realtime

```bash
python recognize.py
```

- Load mô hình đã train, bật webcam
- Nhận diện khuôn mặt liên tục theo thời gian thực
- **Phím tắt:**
  - **[S]** – Lưu ảnh màn hình thủ công vào `captures/`
  - **[Q]** – Thoát chương trình

---

## 📊 Thông số kỹ thuật

| Thông số | Giá trị |
|---|---|
| Số ảnh thu thập / người | 100 ảnh |
| Định dạng ảnh dataset | Grayscale `.jpg` |
| Thuật toán nhận diện | LBPH (Local Binary Patterns Histograms) |
| Bộ phát hiện khuôn mặt | Haar Cascade (`haarcascade_frontalface_default.xml`) |
| Ngưỡng tin cậy nhận diện | ≥ 60% |
| Cooldown lưu ảnh người lạ | 5 giây / lần |
| Scale Factor (detect) | 1.2 – 1.3 |
| Min Neighbors (detect) | 5 |
| Min Face Size | 60 × 60 px |

---

## 📷 Kết quả chạy thực tế

<p align="center">
  <img width="700" alt="Kết quả chạy thử" src="https://github.com/khanhly-dn/RealTime_Face_Recognition/blob/main/KQCT.png?raw=true" />
</p>

Hệ thống nhận diện chính xác người dùng **"khanh"** với độ tin cậy **65%**, hiển thị khung xanh cùng tên và phần trăm tin cậy trực tiếp trên webcam. FPS ổn định ở mức **29 FPS** trên CPU thông thường.

---

## 📁 Các file tự sinh sau khi chạy

<p align="center">
  <img width="700" alt="Cấu trúc file sau khi chạy" src="https://github.com/khanhly-dn/RealTime_Face_Recognition/blob/main/FIILE_100P.png?raw=true" />
</p>

Sau khi chạy đủ 3 bước, hệ thống tự động tạo các thư mục:
- `dataset/` – Chứa ảnh khuôn mặt của từng người (100 ảnh/người)
- `trainer/` – Chứa file mô hình LBPH và file nhãn
- `captures/` – Ảnh lưu tự động (người lạ) và thủ công ([S])
- `logs/` – File log nhận diện theo từng ngày  
🎬 **Video hoạt động:** *https://drive.google.com/file/d/1-iD2clRKsPZOJNPuLhmW1Rr9t24Pwri8/view?usp=sharing*
---

## 🔬 Thuật toán LBPH

**Local Binary Patterns Histograms (LBPH)** là thuật toán nhận diện khuôn mặt cổ điển nhưng hiệu quả:

1. Chia ảnh khuôn mặt thành các ô nhỏ (grid)
2. Với mỗi pixel, so sánh với 8 pixel xung quanh → tạo chuỗi nhị phân
3. Tổng hợp thành histogram đặc trưng cho từng ô
4. Ghép toàn bộ histogram thành vector đặc trưng đại diện cho khuôn mặt
5. Khi nhận diện: so sánh khoảng cách giữa vector mới và các vector đã học

✅ **Ưu điểm:** Nhẹ, nhanh, không cần GPU, hoạt động tốt trong điều kiện ánh sáng thay đổi.

---

## 🚀 Hướng phát triển

- [ ] Tích hợp **Deep Learning** (FaceNet / ArcFace) để tăng độ chính xác
- [ ] Thêm **giao diện GUI** bằng Tkinter hoặc PyQt
- [ ] Kết nối **database** để quản lý người dùng quy mô lớn
- [ ] Tích hợp **điểm danh tự động** (xuất file Excel / Google Sheets)
- [ ] Thêm **thông báo Telegram** khi phát hiện người lạ
- [ ] Hỗ trợ **nhiều camera** cùng lúc
- [ ] Tối ưu hóa với **multi-threading** để tăng FPS

---

## 👤 Thực hiện

**Lý Gia Khánh**  
Khoa Công nghệ Thông tin – Trường Đại học Đại Nam

---

<p align="center">
  Using Python · OpenCV · LBPH · Haar Cascade
</p>
