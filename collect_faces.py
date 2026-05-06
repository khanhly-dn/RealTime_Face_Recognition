import cv2
import os
import sys

# ---- CAU HINH ----
THU_MUC_DATASET = "dataset"
SO_ANH_MO_HINH  = 100

def tao_thu_muc(ten_nguoi):
    duong_dan = os.path.join(THU_MUC_DATASET, ten_nguoi)
    os.makedirs(duong_dan, exist_ok=True)
    return duong_dan

def thu_thap_anh(ten_nguoi):
    duong_dan = tao_thu_muc(ten_nguoi)
    detector  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    camera    = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("[!] Khong mo duoc camera!")
        sys.exit(1)

    print(f"\n[*] Thu thap anh cho: {ten_nguoi}")
    print(f"[*] Se chup {SO_ANH_MO_HINH} anh. Nhin thang vao camera...\n")
    dem = 0
    while True:
        ret, khung = camera.read()
        if not ret:
            break

        xam = cv2.cvtColor(khung, cv2.COLOR_BGR2GRAY)
        khuon_mat = detector.detectMultiScale(xam, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in khuon_mat:
            dem += 1
            ten_file = os.path.join(duong_dan, f"{dem}.jpg")
            cv2.imwrite(ten_file, xam[y:y+h, x:x+w])

            cv2.rectangle(khung, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(khung, f"Da chup: {dem}/{SO_ANH_MO_HINH}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Thu thap khuon mat - Nhan Q de thoat", khung)

        if cv2.waitKey(1) & 0xFF == ord('q') or dem >= SO_ANH_MO_HINH:
            break

    camera.release()
    cv2.destroyAllWindows()
    print(f"\n[+] Da thu thap {dem} anh cho '{ten_nguoi}'")
    print(f"[+] Luu tai: {duong_dan}")

# ---- CHAY CHINH ----
if __name__ == "__main__":
    print("=" * 50)
    print("  BUOC 1: THU THAP ANH KHUON MAT")
    print("=" * 50)
    ten = input("\nNhap ten nguoi can thu thap: ").strip()
    if not ten:
        print("[!] Ten khong duoc de trong!")
        sys.exit(1)
    thu_thap_anh(ten)