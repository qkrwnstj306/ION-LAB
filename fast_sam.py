import sys
import os
import torch
from PIL import Image
import numpy as np
# FastSAM 폴더를 파이썬 모듈 경로에 추가
sys.path.append(os.path.abspath("FastSAM"))

from fastsam import FastSAM, FastSAMPrompt

model = FastSAM('./sam_weights/FastSAM.pt')
IMAGE_PATH = './dataset/content/content_99.jpg'
DEVICE = 'cuda'  # 또는 'cpu'

everything_results = model(IMAGE_PATH, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)

box = [[100, 100, 400, 400]]  # 이 좌표는 사람이 있는 위치로 조정해야 합니다.

# 3. box prompt
ann = prompt_process.box_prompt(bboxes=box)
mask = ann[0].astype(np.uint8) * 255  # 0 or 255

# 4. 바이너리 마스크 이미지 저장
mask_img = Image.fromarray(mask)
mask_img.save('./test_data/mask/content_99_mask.jpg')
#prompt_process.plot(annotations=ann, output_path='./test_data/mask/content_mask.jpg')









#### 흑백 반전
# from PIL import Image, ImageOps

# # 이미지 경로
# img_path = './test_data/mask/content_99_mask.jpg'

# # 1) 이미지 로드
# img = Image.open(img_path)

# # 2) 흑백 변환 (L 모드)
# gray = img.convert("L")

# # 3) 반전
# inverted = ImageOps.invert(gray)

# # 4) 결과 저장
# inverted.save("inverted.png")