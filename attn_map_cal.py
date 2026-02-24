import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math
import os
import torch.nn.functional as F

from math import sqrt

import random



def load_mask(mask_path, h, w):
    mask = torch.tensor(np.load(mask_path), dtype=torch.float32).cuda()
    mask[mask < 0.5] = -1.0
    mask[mask > 0.5] = 1.0
    mask *= -1.0
    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = F.interpolate(mask, size=(
        h, w), mode='bilinear', align_corners=False)
    mask = mask.reshape(1, h, w, 1)
    
    mask_np = mask.squeeze().cpu().numpy()
    indices = np.where(mask_np < 0.0)
    coordinates = list(zip(indices[0], indices[1]))
    
    return coordinates
    
    #### 아래는 디버깅용 ####
    # mask_np = mask.squeeze().cpu().numpy()
    # # 3. Matplotlib으로 이미지 시각화 및 저장
    # fig, ax = plt.subplots(figsize=(8, 8))
    # im = ax.imshow(mask_np, cmap='gray', vmin=-1, vmax=1)
    
    # ax.set_title(f"Visualized Tensor (Shape: {mask_np.shape})")
    # fig.colorbar(im, ax=ax, ticks=[-1.0, 0.0, 1.0]) # 컬러바로 값 확인
    
    # plt.savefig("./latent_mask", dpi=150)
    # plt.close(fig)
    # print(mask)

def extract_random_coordinates(coordinates, num=5):
    # 리스트에서 중복 없이 num_samples 만큼의 아이템을 무작위로 선택
    random_samples = random.sample(coordinates, 5)
    return random_samples
     

def calculate_attention_map(
    q1_pkl_path=None,
    k1_pkl_path=None,
    q2_pkl_path=None,
    k2_pkl_path=None,
    sim_path=None,
    q_key="output_block_11_self_attn_q",
    k_key="output_block_11_self_attn_k",
    save_path="attention_overlay.png",
    latent_resolution=(64, 64),
    coordinates=None,
    Tau=1.5,
    original_img_path=None,
    stylized_img_path=None,
):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1) Load PKL
    with open(q1_pkl_path, "rb") as f:
        q1_feat_maps = pickle.load(f)
    with open(k1_pkl_path, "rb") as f:
        k1_feat_maps = pickle.load(f)
    q1_last_step = q1_feat_maps[-1]
    k1_last_step = k1_feat_maps[-1]
    
    q1 = q1_last_step[q_key]               # Tensor shape: (heads, N, d)
    k1 = k1_last_step[k_key]

    k1 = k1.to(device=device)           
    q1 = q1.to(device=device, dtype=k1.dtype)

    head_num = q1.shape[0]
    pixel_size = q1.shape[1]
    h = w = int(sqrt(pixel_size))
    
     # 2) Attention logit 계산
    d1 = q1.shape[-1]
    logits1 = torch.matmul(q1, k1.transpose(-1, -2)) * (h ** -0.5) * Tau # (Head, 64 * 64, 64 * 64)
    # logits1_reshaped = logits1.reshape(head_num, h, w, pixel_size)  # (Head, h, w, 64 * 64)
    
    logits2 = None
    if q2_pkl_path is not None and k2_pkl_path is not None:
        with open(q2_pkl_path, "rb") as f:
            q2_feat_maps = pickle.load(f)
        with open(k2_pkl_path, "rb") as f:
            k2_feat_maps = pickle.load(f)
        q2_last_step = q2_feat_maps[-1]
        k2_last_step = k2_feat_maps[-1]
        
        q2 = q2_last_step[q_key]               # Tensor shape: (heads, N, d)
        k2 = k2_last_step[k_key]
        
        k2 = k2.to(device=device)           
        q2 = q2.to(device=device, dtype=k2.dtype)
        
        # 2) Attention logit 계산
        d2 = q2.shape[-1]
        logits2 = torch.matmul(q2, k2.transpose(-1, -2)) * (h ** -0.5) * Tau # (Head, 64 * 64, 64 * 64)
        # logits2_reshaped = logits2.reshape(head_num, h, w, pixel_size)  # (Head, h, w, 64 * 64)
    elif sim_path is not None:
        with open(sim_path, "rb") as f:
            sim_maps = pickle.load(f) ## sim은 저장 당시에 Tau 값을 곱한 채로 저장함. 따라서 여기선 곱하지 않음. 
        logits2 = sim_maps
    
    # --- 원본 이미지 및 스타일적용된 이미지 로드 ---
    original_img = Image.open(original_img_path).convert("RGB")
    stylized_img = Image.open(stylized_img_path).convert("RGB")


    random_samples = extract_random_coordinates(coordinates) 
    
    # 3) head별 계산
    for head in range(head_num):
        # attn_all = logits_reshaped[head]  # (h, w, N)
        # attn_flat = attn_all.reshape(-1)  # flatten to 1D → shape: h * w * N
        # # print(f"[Head {head}] Total attention values: {attn_flat.shape[0]}")
        # attn_np = attn_flat.detach().cpu().numpy()
        

        for i in random_samples:
            coord_h, coord_w = i
            query_idx = coord_h * coord_w    
            
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))

            # --- subplot 1: 위치가 표시된 이미지 ---
            ax1.imshow(original_img)
            ax2.imshow(stylized_img)
            
            # 64x64 좌표를 512x512 스케일로 변환 (8배)
            # +4는 영역의 정중앙에 점을 찍기 위한것임.
            # ex) (20,10) -> (160,80) 그대로 쓰면 영역의 왼쪽 위 모서리에 찍힘.
            # 따라서, +4를 해줘야 (164,84) 영역의 정중앙에 점 찍힘.
            scaled_w = coord_w * 8 + 4
            scaled_h = coord_h * 8 + 4
            ax1.scatter(scaled_w, scaled_h, s=150, facecolors='none', edgecolors='r', linewidths=2.5)
            ax1.set_title(f"Query Location: ({coord_h}, {coord_w}) -> Index: {query_idx}", fontsize=14)
            ax1.axis('off')
            ax2.scatter(scaled_w, scaled_h, s=150, facecolors='none', edgecolors='r', linewidths=2.5)
            ax2.set_title(f"Query Location: ({coord_h}, {coord_w}) -> Index: {query_idx}", fontsize=14)
            ax2.axis('off')
            
            # --- subplot 2: 어텐션 분포 히스토그램 ---
            attn_vector_1 = logits1[head, query_idx, :]
            attn_np_1 = attn_vector_1.detach().cpu().numpy()

            attn_vector_2 = logits2[head, query_idx, :]
            attn_np_2 = attn_vector_2.detach().cpu().numpy()
            
            min_val = min(attn_np_1.min(), attn_np_2.min())
            max_val = max(attn_np_1.max(), attn_np_2.max())
            bins = np.linspace(min_val, max_val, 50)
            
            ax3.hist(attn_np_1, bins=bins, color='steelblue', alpha=0.7, label='Original Attn (logits1)')
            ax3.hist(attn_np_2, bins=bins, color='red', alpha=0.6, label='Changed Attn (logits2)')
            ax3.set_title(f"Head {head}: Attention Distribution Change", fontsize=14)
            ax3.set_xlabel("Logit Value", fontsize=12)
            ax3.set_ylabel("Frequency", fontsize=12)
            ax3.legend()
            ax3.grid(True, linestyle='--', alpha=0.6)
            
            # --- 전체 그림 저장 ---
            base_path = os.path.splitext(save_path)[0]
            full_save_path = f"{base_path}_coord_{coord_h}-{coord_w}_head{head}.png"
            os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
            
            plt.tight_layout()
            plt.savefig(full_save_path, dpi=150)
            plt.close(fig)
            print(f"시각화 이미지 저장 완료: {full_save_path}")


 






# ==== 사용 예시 ====
q1_pkl_path = "./saved_qk/layer11_t45_style_back.jpg_content_flower.jpg_qk.pkl"
k1_pkl_path = "./saved_qk/layer11_t45_style_back.jpg_content_flower.jpg_qk.pkl"
# q2_pkl_path = "/home/rkdehdrud/anaconda3/envs/StyleInj/precomputed_feats_k/style1_sty.pkl"
# k2_pkl_path = "/home/rkdehdrud/anaconda3/envs/StyleInj/precomputed_feats_k/style1_sty.pkl"
sim_path = "./saved_sim/layer11_t45_style_back.jpg_content_flower.jpg_sim.pkl"

coordinates = load_mask("./dataset/mask/content_flower_mask.npy", 64, 64) ## 64, 64는 각자의 latent resolution에 맞게 변경!!

# calculate_attention_map(
#     q1_pkl_path,
#     k1_pkl_path,
#     q2_pkl_path,
#     k2_pkl_path,
#     save_path="./mask_attn_results/attn_logit.png",
#     latent_resolution=(64,64),# SD default latent spatial size
#     coordinates=coordinates,
#     Tau=1.5,
# )

calculate_attention_map(
    q1_pkl_path,
    k1_pkl_path,
    sim_path=sim_path,
    save_path="./attn_histogram_results/attn_logit.png",
    latent_resolution=(64,64),# SD default latent spatial size
    coordinates=coordinates,
    Tau=1.5,
    original_img_path="./dataset/content/content_flower.jpg",
    stylized_img_path="./output_dk/content_flower_stylized_style_back.png",
)
