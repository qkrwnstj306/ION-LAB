import os

folder_path = "./data_vis/cnt"

# 폴더 내 파일 리스트
files = os.listdir(folder_path)
png_files = [f for f in files if f.endswith(".jpg")]
npy_files = [f for f in files if f.endswith(".npy")]


# 확장자 제외한 이름으로 매칭
png_stems = [os.path.splitext(f)[0] for f in png_files]

for npy_file in npy_files:
    npy_stem = os.path.splitext(npy_file)[0]
    
    # npy 파일 이름이 어떤 png 이름과 매칭되는지 확인
    for png_stem in png_stems:
        if npy_stem.startswith(png_stem):
            new_name = f"{png_stem}_mask.npy"
            old_path = os.path.join(folder_path, npy_file)
            new_path = os.path.join(folder_path, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed {npy_file} -> {new_name}")
            break