import numpy as np
import cv2
import os
import argparse

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_dir", type=str, required=True, help="Path to directory containing mask images")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save .npy files")
    args = parser.parse_args()

    mask_dir = args.mask_dir
    save_dir = args.save_dir

    if not os.path.isdir(mask_dir):
        raise NotADirectoryError(f"{mask_dir} is not a valid directory")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # save_dir가 없으면 생성

    # 디렉토리 내 모든 이미지 파일 처리
    for fname in os.listdir(mask_dir):
        if not fname.lower().endswith(IMG_EXTENSIONS):
            continue

        mask_path = os.path.join(mask_dir, fname)
        mask_img = cv2.imread(mask_path, 0)  # grayscale
        if mask_img is None:
            print(f"Warning: could not load {mask_path}, skipping")
            continue

        mask_img = mask_img / 255.0  # Convert to 0~1 float

        # .npy 저장 경로를 save_dir로 변경
        base_name, _ = os.path.splitext(fname)
        mask_npy_path = os.path.join(save_dir, base_name + ".npy")

        np.save(mask_npy_path, mask_img.astype(np.float32))
        print("Saved mask:", mask_npy_path)

if __name__ == "__main__":
    main()
