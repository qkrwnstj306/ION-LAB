import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math
import os
import torch.nn.functional as F

def visualize_attention_overlay(
    q_pkl_path,
    k_pkl_path,
    img_path,
    q_key="output_block_11_self_attn_q",
    k_key="output_block_11_self_attn_k",
    query_index=None,
    mask_path=None,  
    save_path="attention_overlay.png",
    latent_resolution=(64, 64)
):
    

    
    # 1) Load PKL
    with open(q_pkl_path, "rb") as f:
        q_feat_maps = pickle.load(f)
    with open(k_pkl_path, "rb") as f:
        k_feat_maps = pickle.load(f)
    q_last_step = q_feat_maps[-1]
    k_last_step = k_feat_maps[-1]
    
    q = q_last_step[q_key]  # (Head, B*N, D)
    k = k_last_step[k_key]  # (Head, B*N, D)

    

    # 2) Attention logit 계산
    d = q.shape[-1]
    logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)  # (Head, B*N, B*N)
    logits = logits.mean(0)  # head 평균 (B*N, B*N)

    # 3) Query index 선택
    N = latent_resolution[0] * latent_resolution[1]
    
   
    # ### mask 적용 부분
    # # if mask_path is not None:
    # #     mask = torch.tensor(np.load(mask_path), dtype=torch.float32).cuda()
    # #     mask = mask.unsqueeze(0).unsqueeze(0)
    # #     H, N, D = q.shape
    # #     mask = F.interpolate(mask, size=(int(math.sqrt(q.shape[1])),
    # #                                       int(math.sqrt(q.shape[1]))),
    # #                           mode='bilinear', align_corners=False)
    # #     mask = mask.view(1, N, 1)              # (1, N, 1)
    # #     mask = mask.expand(H, -1, -1)          # (H, N, 1)
    # #     # 그 다음 mask을 logits에 곱해줌
    # #     logits = logits * mask  # (H, N, N)
        
        
        
        
    # if query_index is None:
    #     # # min
    #     # query_mean_scores = logits.mean(dim=1)  # 각 query의 평균 attention score
    #     # query_index = torch.argmin(query_mean_scores).item()
    #     # print(f"[Auto-selected min query_index] {query_index}")
    #     ## max
    #     query_mean_scores = logits.mean(dim=1)  # 각 query의 평균 attention score
    #     query_index = torch.argmax(query_mean_scores).item()  # <-- 여기만 argmax로 변경
    #     print(f"[Auto-selected max query_index] {query_index}")

    # attn_map = logits[query_index].reshape(latent_resolution)  # (H, W)

    # ---- 기존 query index 선택 부분 대신 ----
    # 전체 query에 대한 평균 attention score (열 평균)
    attn_map = logits.mean(dim=1).reshape(latent_resolution)  # (H, W)

        
    

    # 4) Normalize heatmap
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    attn_map = attn_map.cpu().numpy()

    # 5) Content image load & resize
    img = Image.open(img_path).convert("RGB")
    img = img.resize((latent_resolution[1]*8, latent_resolution[0]*8))  # upscale for visualization

    # 6) Overlay
    plt.figure(figsize=(8,8))
    plt.imshow(img)
    plt.imshow(attn_map, cmap="jet", alpha=0.5, extent=(0, img.width, img.height, 0))
    plt.title("Attention Overlay (query index = {})".format(query_index))
    plt.axis("off")

    # query_index 반영된 경로 생성
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    base, ext = os.path.splitext(save_path)
    save_path_with_idx = f"{base}_{img_name}_q{query_index}{ext}"

    os.makedirs(os.path.dirname(save_path_with_idx), exist_ok=True)
    plt.savefig(save_path_with_idx, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path_with_idx}")


# ==== 사용 예시 ====
q_pkl_path = "/home/rkdehdrud/anaconda3/envs/StyleInj/precomputed_feats_k/content_5_cnt.pkl"
k_pkl_path = "/home/rkdehdrud/anaconda3/envs/StyleInj/precomputed_feats_k/style_back_sty.pkl"
img_path = "/home/rkdehdrud/anaconda3/envs/StyleInj/dataset/content/content_5.jpg"
visualize_attention_overlay(
    q_pkl_path,
    k_pkl_path,
    img_path,
    save_path="./attn_lastlayer_laststep_logit.png",
    latent_resolution=(64,64),# SD default latent spatial size
    # query_index=None
    # mask_path=None, 
)
