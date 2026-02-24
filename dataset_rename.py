import os

def rename_files_in_folder(folder_path, prefix):
    files = sorted(os.listdir(folder_path))
    idx = 1

    for file in files:
        ext = os.path.splitext(file)[1]
        # ext = '.npy'
        new_name = f"{prefix}_{idx:03d}_mask{ext}"
        # new_name = f"{idx:03d}{ext}"
        old_path = os.path.join(folder_path, file)
        new_path = os.path.join(folder_path, new_name)

        os.rename(old_path, new_path)
        idx += 1

    # print(f"Renamed files in {folder_path} with prefix {prefix}_###.jpg")


# 실제 경로 수정해서 사용
content_folder = "/home/qkrwnstj/Style-Transfer/data_mj/mask_npy"

rename_files_in_folder(content_folder, "content")
# rename_files_in_folder(content_folder, "")

