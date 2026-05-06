import cv2
import os
import numpy as np
from PIL import Image

# ---- CAU HINH ----
THU_MUC_DATASET = "dataset"
THU_MUC_TRAINER = "trainer"
FILE_MODEL      = os.path.join(THU_MUC_TRAINER, "model.yml")
FILE_NHAN       = os.path.join(THU_MUC_TRAINER, "labels.txt")

def lay_anh_va_nhan():
    duong_dan_anh = []
    nhan          = []
    id_nguoi      = {}
    id_hien_tai   = 0

    for ten_nguoi in sorted(os.listdir(THU_MUC_DATASET)):
        thu_muc_nguoi = os.path.join(THU_MUC_DATASET, ten_nguoi)
        if not os.path.isdir(thu_muc_nguoi):
            continue

        id_nguoi[id_hien_tai] = ten_nguoi
        print(f"  [+] Dang xu ly: {ten_nguoi} (ID={id_hien_tai})")

        for ten_file in os.listdir(thu_muc_nguoi):
            if not ten_file.endswith(".jpg"):
                continue
            duong_dan = os.path.join(thu_muc_nguoi, ten_file)
            anh_xam   = Image.open(duong_dan).convert("L")
            mang_anh  = np.array(anh_xam, dtype="uint8")
            duong_dan_anh.append(mang_anh)
            nhan.append(id_hien_tai)
        id_hien_tai += 1
    return duong_dan_anh, nhan, id_nguoi

def luu_nhan(id_nguoi):
    os.makedirs(THU_MUC_TRAINER, exist_ok=True)
    with open(FILE_NHAN, "w", encoding="utf-8") as f:
        for id_, ten in id_nguoi.items():
            f.write(f"{id_}:{ten}\n")
    print(f"[+] Da luu nhan: {FILE_NHAN}")

def train():
    print("=" * 50)
    print("  BUOC 2: TRAIN MO HINH NHAN DIEN")
    print("=" * 50)
    if not os.path.exists(THU_MUC_DATASET) or not os.listdir(THU_MUC_DATASET):
        print("[!] Chua co du lieu! Chay collect_faces.py truoc.")
        return

    print("\n[*] Dang doc anh tu dataset...")
    danh_sach_anh, danh_sach_nhan, id_nguoi = lay_anh_va_nhan()

    if not danh_sach_anh:
        print("[!] Khong tim thay anh nao trong dataset!")
        return
    print(f"\n[*] Tong so anh: {len(danh_sach_anh)}")
    print(f"[*] So nguoi:    {len(id_nguoi)}")
    print(f"[*] Dang train mo hinh LBPH...")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(danh_sach_anh, np.array(danh_sach_nhan))

    os.makedirs(THU_MUC_TRAINER, exist_ok=True)
    recognizer.write(FILE_MODEL)
    luu_nhan(id_nguoi)
    print(f"\n[+] Train xong! Mo hinh luu tai: {FILE_MODEL}")
    print(f"[+] Danh sach nguoi:")
    for id_, ten in id_nguoi.items():
        print(f"    ID {id_} → {ten}")

# ---- CHAY CHINH ----
if __name__ == "__main__":
    train()