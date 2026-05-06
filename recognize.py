import cv2
import os
import sys
import numpy as np
from datetime import datetime

# ---- CAU HINH ----
THU_MUC_TRAINER  = "trainer"
THU_MUC_CAPTURES = "captures"
THU_MUC_LOGS     = "logs"
FILE_MODEL       = os.path.join(THU_MUC_TRAINER, "model.yml")
FILE_NHAN        = os.path.join(THU_MUC_TRAINER, "labels.txt")
NGUONG_TIN_CAY   = 60     
FPS_HIEN_THI     = 30

# ---- MAU SAC ----
MAU_BIET    = (0,   255,  0)    
MAU_LA      = (0,   165, 255)  
MAU_TRANG   = (255, 255, 255)
MAU_DEN     = (0,   0,   0)
MAU_DO      = (0,   0,   255)

# ---- DOC MO HINH ----
def doc_nhan():
    nhan = {}
    if not os.path.exists(FILE_NHAN):
        return nhan
    with open(FILE_NHAN, "r", encoding="utf-8") as f:
        for dong in f:
            dong = dong.strip()
            if ":" in dong:
                id_, ten = dong.split(":", 1)
                nhan[int(id_)] = ten
    return nhan

def kiem_tra_mo_hinh():
    if not os.path.exists(FILE_MODEL):
        print("[!] Chua co mo hinh! Chay train.py truoc.")
        sys.exit(1)
    if not os.path.exists(FILE_NHAN):
        print("[!] Khong tim thay file nhan!")
        sys.exit(1)

# ---- LOG ----
def ghi_log(ten, do_tin_cay, ten_file_anh=""):
    os.makedirs(THU_MUC_LOGS, exist_ok=True)
    ten_log  = os.path.join(THU_MUC_LOGS, f"log_{datetime.now().strftime('%Y%m%d')}.txt")
    tg       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    noi_dung = f"[{tg}]  {ten:<20}  do_tin_cay={do_tin_cay:.1f}  anh={ten_file_anh}\n"
    with open(ten_log, "a", encoding="utf-8") as f:
        f.write(noi_dung)

# ---- LUU ANH ----
def luu_anh(khung, ten):
    os.makedirs(THU_MUC_CAPTURES, exist_ok=True)
    tg       = datetime.now().strftime("%Y%m%d_%H%M%S")
    ten_file = os.path.join(THU_MUC_CAPTURES, f"{ten}_{tg}.jpg")
    cv2.imwrite(ten_file, khung)
    return ten_file

# ---- VE UI LEN KHUNG HINH ----
def ve_hop_khuon_mat(khung, x, y, w, h, ten, do_tin_cay, la_biet):
    mau = MAU_BIET if la_biet else MAU_LA

    cv2.rectangle(khung, (x, y), (x+w, y+h), mau, 2)
    cv2.rectangle(khung, (x, y+h), (x+w, y+h+45), mau, cv2.FILLED)
    cv2.putText(khung, ten, (x+6, y+h+22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, MAU_DEN, 2)
    cv2.putText(khung, f"{do_tin_cay:.0f}%", (x+6, y+h+40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, MAU_DEN, 1)

def ve_thong_tin(khung, dem_khuon_mat, dem_biet, dem_la, fps):
    h_khung, w_khung = khung.shape[:2]

    overlay = khung.copy()
    cv2.rectangle(overlay, (0, 0), (w_khung, 70), MAU_DEN, cv2.FILLED)
    cv2.addWeighted(overlay, 0.5, khung, 0.5, 0, khung)

    tg = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(khung, f"FACE RECOGNITION  |  {tg}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, MAU_TRANG, 1)
    cv2.putText(khung, f"Khuon mat: {dem_khuon_mat}   Biet: {dem_biet}   La: {dem_la}   FPS: {fps:.0f}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, MAU_TRANG, 1)

    cv2.putText(khung, "[S] Luu anh   [Q] Thoat", (10, h_khung - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

# ---- NHAN DIEN CHINH ----
def nhan_dien():
    kiem_tra_mo_hinh()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(FILE_MODEL)
    nhan       = doc_nhan()
    detector   = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    camera     = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("[!] Khong mo duoc camera!")
        sys.exit(1)

    print("=" * 50)
    print("  BUOC 3: NHAN DIEN KHUON MAT REALTIME")
    print("=" * 50)
    print(f"\n[*] Mo hinh nhan biet {len(nhan)} nguoi: {list(nhan.values())}")
    print("[*] Nhan [S] de luu anh, [Q] de thoat\n")
    t_truoc      = cv2.getTickCount()
    lan_luu_cuoi = {}  

    while True:
        ret, khung = camera.read()
        if not ret:                                                                                                                                                                                                                                                                                                                                                                 
            break

        # Tinh FPS
        t_hien = cv2.getTickCount()
        fps    = cv2.getTickFrequency() / (t_hien - t_truoc)
        t_truoc = t_hien
        xam         = cv2.cvtColor(khung, cv2.COLOR_BGR2GRAY)
        khuon_mat   = detector.detectMultiScale(xam, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
        dem_biet = 0
        dem_la   = 0

        for (x, y, w, h) in khuon_mat:
            id_, khoang_cach = recognizer.predict(xam[y:y+h, x:x+w])
            do_tin_cay       = max(0, 100 - khoang_cach)
            if do_tin_cay >= NGUONG_TIN_CAY and id_ in nhan:
                ten    = nhan[id_]
                la_biet = True
                dem_biet += 1
            else:
                ten    = "Nguoi la"
                la_biet = False
                dem_la  += 1

            ve_hop_khuon_mat(khung, x, y, w, h, ten, do_tin_cay, la_biet)

            if not la_biet:
                t_hien_ms = cv2.getTickCount() / cv2.getTickFrequency()
                t_cuoi    = lan_luu_cuoi.get("Nguoi la", 0)
                if t_hien_ms - t_cuoi > 5:   # Luu moi 5 giay
                    ten_file = luu_anh(khung, "nguoi_la")
                    ghi_log("Nguoi la", do_tin_cay, ten_file)
                    lan_luu_cuoi["Nguoi la"] = t_hien_ms
                    print(f"  [!] Phat hien nguoi la – da luu: {ten_file}")
        ve_thong_tin(khung, len(khuon_mat), dem_biet, dem_la, fps)
        cv2.imshow("Face Recognition – Nhan Q de thoat", khung)
        phim = cv2.waitKey(1) & 0xFF
        if phim == ord('q'):
            break
        elif phim == ord('s'):
            ten_file = luu_anh(khung, "manual")
            ghi_log("Manual_save", 0, ten_file)
            print(f"  [+] Da luu anh: {ten_file}")
    camera.release()
    cv2.destroyAllWindows()
    print("\n[+] Da thoat chuong trinh.")

if __name__ == "__main__":
    nhan_dien()