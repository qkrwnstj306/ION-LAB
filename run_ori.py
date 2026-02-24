import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import copy
import gc
import sys

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

import torch.nn.functional as F
import time
import pickle

## 마스크 적용
from ldm.modules.attention import CrossAttention
from high_frequency_final import patch_decoder_resblocks_h_and_cnt_hf, make_content_injection_schedule

from dataset import TripleImageDataset
from torch.utils.data import DataLoader

import psutil
import torch
import time


process = psutil.Process()

# 지원할 실행 해상도 조합을 명시적으로 제한.
# (추후 실험 범위를 늘릴 때 이 집합만 확장하면 됨.)
SUPPORTED_INFERENCE_HW = {
    (512, 512),
    (512, 1024),
    (1024, 512),
    (1024, 1024),
}


def resolve_runtime_resolution(opt, model_config):
    """
    실행 해상도(H, W)를 model_config 기본값 + CLI override 규칙으로 결정합니다.
    - CLI(--H/--W)가 있으면 최우선
    - 없으면 model_config.inference.default_h/default_w 사용
    - 둘 다 없으면 512x512 사용
    """
    cfg_h, cfg_w = None, None

    # OmegaConf 기반 config에서 커스텀 inference 섹션을 읽음.
    if model_config is not None and "inference" in model_config:
        inference_cfg = model_config.inference
        cfg_h = inference_cfg.get("default_h", None)
        cfg_w = inference_cfg.get("default_w", None)

        # [호환용] default_hw: [H, W] 형태도 허용.
        if (cfg_h is None or cfg_w is None) and "default_hw" in inference_cfg:
            default_hw = inference_cfg.get("default_hw")
            if default_hw is not None and len(default_hw) == 2:
                cfg_h = default_hw[0]
                cfg_w = default_hw[1]

    h = opt.H if opt.H is not None else (cfg_h if cfg_h is not None else 512)
    w = opt.W if opt.W is not None else (cfg_w if cfg_w is not None else 512)

    h = int(h)
    w = int(w)
    return h, w


def validate_runtime_resolution(h, w, f):
    """
    현재 코드가 의도한 해상도 조합만 허용하고,
    latent downsampling 계수(f)와도 정합이 맞는지 확인합니다.
    """
    if (h, w) not in SUPPORTED_INFERENCE_HW:
        raise ValueError(
            f"지원하지 않는 해상도 조합입니다: ({h}, {w}). "
            f"지원 해상도: {sorted(SUPPORTED_INFERENCE_HW)}"
        )

    if h % f != 0 or w % f != 0:
        raise ValueError(
            f"H/W는 f={f}의 배수여야 합니다. 현재 H={h}, W={w}, f={f}"
        )


def get_resolution_scoped_precomputed_dir(base_dir, h, w):
    """
    precomputed feature 경로 규칙:
    - 512x512: 기존 실험 자산과 호환되도록 base_dir 그대로 사용
    - 그 외 지원 해상도: {base_dir}/{H}x{W} 하위 폴더 사용
    """
    if (h, w) == (512, 512):
        return base_dir
    return os.path.join(base_dir, f"{h}x{w}")


def center_crop_tensor_hw(x, target_h, target_w):
    """
    4D tensor(B, C, H, W)를 정중앙 기준으로 target_h x target_w로 crop합니다.
    run 스크립트에서 후처리로 해상도 정렬할 때 사용합니다.
    """
    if x.ndim != 4:
        raise ValueError(f"center_crop_tensor_hw는 4D tensor만 지원합니다. 현재 shape={tuple(x.shape)}")

    _, _, h, w = x.shape
    if target_h > h or target_w > w:
        raise ValueError(
            f"target crop 크기가 입력보다 큽니다. input=({h}, {w}), target=({target_h}, {target_w})"
        )

    top = (h - target_h) // 2
    left = (w - target_w) // 2
    return x[:, :, top:top + target_h, left:left + target_w]


def prepare_mask_tensor_for_runtime(mask_npy_path, target_h, target_w, device):
    """
    content 마스크(.npy)를 실행 해상도 기준으로 정렬합니다.

    반환 shape: (1, 1, target_h, target_w)
    """
    mask_np = np.load(mask_npy_path).astype(np.float32)
    if mask_np.ndim != 2:
        raise ValueError(f"마스크는 2D npy여야 합니다: {mask_npy_path}, shape={mask_np.shape}")

    mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)

    # 1) 먼저 정중앙 기준으로 정사각형 crop (dataset.py의 image crop과 동일한 방향)
    h0, w0 = mask.shape[-2], mask.shape[-1]
    side = min(h0, w0)
    top = (h0 - side) // 2
    left = (w0 - side) // 2
    mask = mask[:, :, top:top + side, left:left + side]

    
    # 2)  run 단계에서 target(H,W)로 중앙 crop
    square_side = max(target_h, target_w)
    mask = F.interpolate(mask, size=(square_side, square_side), mode="bilinear", align_corners=False)
    mask = center_crop_tensor_hw(mask, target_h, target_w)
    return mask


def configure_cross_attention_runtime_resolution(model_or_unet, image_h, image_w, f):
    """
    attention 모듈이 self-attention token 개수를 직사각형 latent 해상도로 해석할 수 있도록
    base latent 해상도(H//f, W//f)를 주입합니다.
    """
    latent_hw = (image_h // f, image_w // f)
    for m in model_or_unet.modules():
        if isinstance(m, CrossAttention):
            # attention.py에서 sqrt(N) 대신 이 값을 기준으로 계층별 해상도를 역추론합니다.
            m.base_latent_hw = latent_hw


def validate_precomputed_z_enc_shape(name, z_enc, target_h, target_w, f):
    """
    precomputed feature가 현재 실행 해상도용으로 생성된 것인지 빠르게 검증합니다.
    가장 먼저 z_enc spatial 크기를 확인하면 해상도 mismatch를 조기에 잡을 수 있습니다.
    """
    if z_enc is None:
        raise FileNotFoundError(f"precomputed z_enc가 없습니다: {name}")

    expected_h = target_h // f
    expected_w = target_w // f
    actual_h, actual_w = z_enc.shape[-2], z_enc.shape[-1]
    if (actual_h, actual_w) != (expected_h, expected_w):
        raise ValueError(
            f"precomputed z_enc 해상도 불일치: {name} "
            f"(expected latent=({expected_h}, {expected_w}), actual=({actual_h}, {actual_w}))"
        )


def resolve_meta_path(data_root, meta_file):
    """
    run_ori.py 전용 메타 파일 경로를 결정합니다.
    - 절대경로면 그대로 사용
    - 상대경로면 data_root 기준으로 해석
    """
    if os.path.isabs(meta_file):
        return meta_file
    return os.path.join(data_root, meta_file)


def _resolve_content_token_path(data_root, token):
    """
    메타 파일의 content 토큰을 실제 경로로 변환합니다.
    지원 예시:
    - `content_001.jpg` -> {data_root}/cnt/content_001.jpg
    - `cnt/content_001.jpg` -> {data_root}/cnt/content_001.jpg
    - 절대경로 -> 그대로 사용
    """
    if os.path.isabs(token):
        return token
    if token.startswith("cnt/"):
        return os.path.join(data_root, token)
    return os.path.join(data_root, "cnt", token)


def _resolve_style_token_path(data_root, token, style_index, total_styles):
    """
    메타 파일의 style 토큰을 실제 경로로 변환합니다.
    지원 예시:
    - 절대경로
    - `sty/...` (data_root 기준 상대경로)
    - `char/...`, `back/...` (자동으로 {data_root}/sty 하위로 해석)
    - legacy 2-style plain filename:
      - 첫 번째 style -> sty/char
      - 두 번째 style -> sty/back
    - 그 외 plain filename -> sty/ 하위 flat 파일로 해석
    """
    if os.path.isabs(token):
        return token

    if token.startswith("sty/"):
        return os.path.join(data_root, token)

    if "/" in token:
        return os.path.join(data_root, "sty", token)

    # legacy 2-style 포맷(_dataset.txt) 호환:
    # style filename만 적혀 있을 때 첫 번째는 char, 두 번째는 back으로 해석합니다.
    if total_styles == 2:
        if style_index == 0:
            return os.path.join(data_root, "sty", "char", token)
        if style_index == 1:
            return os.path.join(data_root, "sty", "back", token)

    return os.path.join(data_root, "sty", token)


def parse_multistyle_meta_samples(data_root, meta_file):
    """
    run_ori.py 전용 메타 파서.
    한 줄 형식(공백 구분):
      <content_token> <style_token_1> <style_token_2> ... <style_token_N>

    반환:
      [
        {
          "cnt_path": ...,
          "style_paths": [...],
        },
        ...
      ]
    """
    meta_path = resolve_meta_path(data_root, meta_file)
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"메타 파일을 찾을 수 없습니다: {meta_path}")

    samples = []
    with open(meta_path, "r") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            # 빈 줄/주석 줄은 무시. (메타 파일 가독성용)
            if (not line) or line.startswith("#"):
                continue

            tokens = line.split()
            if len(tokens) < 2:
                raise ValueError(
                    f"메타 파일 형식 오류(line {line_no}): content + style 1개 이상이 필요합니다. "
                    f"line='{line}'"
                )

            cnt_token = tokens[0]
            style_tokens = tokens[1:]
            cnt_path = _resolve_content_token_path(data_root, cnt_token)
            style_paths = [
                _resolve_style_token_path(data_root, token, idx, len(style_tokens))
                for idx, token in enumerate(style_tokens)
            ]

            if not os.path.isfile(cnt_path):
                raise FileNotFoundError(f"Content image not found (line {line_no}): {cnt_path}")
            for style_path in style_paths:
                if not os.path.isfile(style_path):
                    raise FileNotFoundError(f"Style image not found (line {line_no}): {style_path}")

            samples.append({
                "cnt_path": cnt_path,
                "style_paths": style_paths,
            })

    # 메타 파일에 적은 순서를 그대로 보존합니다.
    # (사용자가 샘플 처리 순서를 제어할 수 있도록 정렬하지 않습니다.)
    return samples


def prepare_style_mask_tensor_for_runtime(mask_npy_path, target_h, target_w, device):
    """
    스타일 마스크 파일(_mask{i}.npy)을 실행 해상도(H, W) 기준으로 정렬합니다.
    (중앙 정사각 crop -> 정사각 resize -> 중앙 crop)을 적용합니다.

    반환 shape: (1, 1, target_h, target_w), dtype=float32, 값 범위는 [0, 1]
    """
    mask_np = np.load(mask_npy_path).astype(np.float32)
    if mask_np.ndim != 2:
        raise ValueError(f"style mask는 2D npy여야 합니다: {mask_npy_path}, shape={mask_np.shape}")

    # 외부에서 0/255 uint8 형태로 저장해도 수용할 수 있도록 정규화합니다.
    if mask_np.max() > 1.0:
        mask_np = mask_np / 255.0
    mask_np = np.clip(mask_np, 0.0, 1.0)

    mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)

    h0, w0 = mask.shape[-2], mask.shape[-1]
    side = min(h0, w0)
    top = (h0 - side) // 2
    left = (w0 - side) // 2
    mask = mask[:, :, top:top + side, left:left + side]

    square_side = max(target_h, target_w)
    # 바이너리 마스크라도 리사이즈 경계부에서 soft weight를 허용해 더 부드럽게 혼합될 수 있게 합니다.
    mask = F.interpolate(mask, size=(square_side, square_side), mode="bilinear", align_corners=False)
    mask = center_crop_tensor_hw(mask, target_h, target_w)
    return mask.clamp_(0.0, 1.0)


def _multimask_path_list_from_content(cnt_path, num_styles):
    """
    content 경로로부터 N-style 마스크 파일 경로 목록을 생성합니다.
    예: content_001.png -> content_001_mask0.npy, content_001_mask1.npy, ...
    """
    if num_styles < 1:
        raise ValueError(f"num_styles는 1 이상이어야 합니다. 현재 num_styles={num_styles}")

    base = os.path.splitext(cnt_path)[0]
    return [f"{base}_mask{style_idx}.npy" for style_idx in range(num_styles)]


def load_style_mask_stack_for_runtime(cnt_path, target_h, target_w, num_styles, device):
    """
    `_mask0.npy`, `_mask1.npy`, ... `_mask{N-1}.npy` 파일을 읽어 실행 해상도 기준 style mask stack으로 만듭니다.

    반환 shape: (num_styles, 1, target_h, target_w), dtype=float32

    구현 의도:
    - 마스크 파일은 여러 장으로 분리되어 들어오므로, 여기서는 "raw style weight"로만 읽습니다.
    - 정규화/coverage 처리(겹침/비어있는 영역 처리)는 호출부(AdaIN/attention)에서 목적에 맞게 수행합니다.
    """
    mask_paths = _multimask_path_list_from_content(cnt_path, num_styles)
    missing_paths = [p for p in mask_paths if not os.path.isfile(p)]
    if missing_paths:
        raise FileNotFoundError(
            "N-style 멀티 마스크 파일이 누락되었습니다. "
            f"기대 파일 예시: {mask_paths[0]} ... {mask_paths[-1]}\n"
            f"누락 파일: {missing_paths}"
        )

    mask_tensors = []
    for mask_path in mask_paths:
        mask_tensors.append(
            prepare_style_mask_tensor_for_runtime(
                mask_path,
                target_h=target_h,
                target_w=target_w,
                device=device,
            )
        )
    return torch.cat(mask_tensors, dim=0)  # (N,1,H,W)


def build_style_weight_maps_for_latent_from_masks(style_mask_stack_img, latent_h, latent_w):
    """
    실행 해상도 기준 style mask stack(N,1,H,W)을 latent 해상도용 weight map으로 변환합니다.

    반환:
    - norm_weights: (N,1,latent_h,latent_w) 스타일 간 정규화된 가중치 (합이 1 또는 0)
    - coverage: (1,1,latent_h,latent_w) 어떤 스타일 마스크라도 존재하는 영역(0~1)

    정책:
    - 마스크가 겹치면 style 축으로 정규화하여 혼합
    - 마스크가 비어있는 영역(sum=0)은 coverage=0으로 남겨 호출부에서 content fallback 가능
    """
    raw_weights = F.interpolate(
        style_mask_stack_img,
        size=(latent_h, latent_w),
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)

    weight_sum = raw_weights.sum(dim=0, keepdim=True)  # (1,1,h,w)
    coverage = weight_sum.clamp(0.0, 1.0)
    norm_weights = raw_weights / weight_sum.clamp_min(1e-6)
    return norm_weights, coverage

def get_cpu_mem():
    # MB 단위 반환
    return process.memory_info().rss / 1024 ** 2

def save_img_from_sample(model, samples_ddim, fname):
    with torch.no_grad():
        x_samples_ddim = model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
        x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
        x_sample = 255. * rearrange(x_image_torch[0].cpu().numpy(), 'c h w -> h w c')
        img = Image.fromarray(x_sample.astype(np.uint8))
        img.save(fname)

def load_minimal_feat(feat_path, extract_z_enc=True):
    """필요한 최소한의 데이터만 추출하고 원본 즉시 삭제"""
    with open(feat_path, 'rb') as h:
        feat_full = pickle.load(h)
    
    # z_enc 추출
    z_enc = None
    if extract_z_enc and feat_full[0] is not None and 'z_enc' in feat_full[0]:
        z_enc = feat_full[0]['z_enc'].clone().detach().cpu()
    
    # feat_merge에 필요한 최소 데이터만 추출
    feat_minimal = []
    for item in feat_full:
        if item is None:
            feat_minimal.append(None)
        else:
            minimal_dict = {}
            for key in item.keys():
                # q, k, v로 끝나는 키들과 기타 필요한 키들만 복사
                if any(key.endswith(suffix) for suffix in ['q', 'k', 'v', '_cnt_h']):
                    if torch.is_tensor(item[key]):
                        minimal_dict[key] = item[key].clone().detach().cpu()
                    else:
                        minimal_dict[key] = item[key]
            feat_minimal.append(minimal_dict)
    
    # 원본 즉시 삭제
    del feat_full
    gc.collect()
    
    return z_enc, feat_minimal

def feat_merge_2sty_clean(opt, cnt_feats, sty_feats_1, sty_feats_2, start_step=0):
    """feat_maps_local을 받지 않고 새로 생성"""
    merged_maps = []
    
    for i in range(len(cnt_feats)):
        feat_map = {
            'config': {  
                'gamma': opt.gamma,
                'T': opt.T,
                'timestep': i,
                'cnt_k': None,
                'sty_q': None,
            }
        }
        
        cnt_feat = cnt_feats[i]
        sty_feat_1 = sty_feats_1[i]
        sty_feat_2 = sty_feats_2[i]
        
        if cnt_feat is None or sty_feat_1 is None or sty_feat_2 is None:
            merged_maps.append(feat_map)
            continue
            
        ori_keys = cnt_feat.keys()

        for ori_key in ori_keys:
            if ori_key.endswith('q'):
                if torch.is_tensor(cnt_feat[ori_key]):
                    feat_map[ori_key] = cnt_feat[ori_key].clone().detach()
                else:
                    feat_map[ori_key] = cnt_feat[ori_key]
                    
            if ori_key.endswith('k') or ori_key.endswith('v'):
                if torch.is_tensor(sty_feat_1[ori_key]):
                    feat_map[ori_key] = sty_feat_1[ori_key].clone().detach()
                else:
                    feat_map[ori_key] = sty_feat_1[ori_key]

            if ori_key.endswith('k') or ori_key.endswith('v'):
                if torch.is_tensor(cnt_feat[ori_key]):
                    feat_map[ori_key + '_cnt'] = cnt_feat[ori_key].clone().detach()
                else:
                    feat_map[ori_key + '_cnt'] = cnt_feat[ori_key]

            if ori_key.endswith('q'):
                if torch.is_tensor(sty_feat_1[ori_key]):
                    feat_map[ori_key + '_sty1'] = sty_feat_1[ori_key].clone().detach()
                else:
                    feat_map[ori_key + '_sty1'] = sty_feat_1[ori_key]

            if ori_key.endswith('q') or ori_key.endswith('k') or ori_key.endswith('v'):
                if torch.is_tensor(sty_feat_2[ori_key]):
                    feat_map[ori_key + '_sty2'] = sty_feat_2[ori_key].clone().detach()
                else:
                    feat_map[ori_key + '_sty2'] = sty_feat_2[ori_key]
        
        merged_maps.append(feat_map)
    
    return merged_maps


def feat_merge_nsty_clean(opt, cnt_feats, style_feats_list, pi_style_total=0.9, start_step=0):
    """
    N-style attention 주입용 feature map을 생성합니다.

    설계 원칙:
    - content q -> 기본 key(`..._self_attn_q`)
    - style1 k/v -> 기본 key(`..._self_attn_k/v`)  # 기존 경로와 호환
    - content k/v -> `..._cnt`
    - style2..styleN k/v -> `..._sty2`, `..._sty3`, ...
    - config에 `num_styles`, `pi_style_total` 저장 (attention N-style 경로에서 사용)
    """
    if len(style_feats_list) == 0:
        raise ValueError("style_feats_list가 비어 있습니다. N-style 주입에는 최소 1개 style이 필요합니다.")

    num_styles = len(style_feats_list)
    merged_maps = []

    for i in range(len(cnt_feats)):
        feat_map = {
            'config': {
                'gamma': opt.gamma,
                'T': opt.T,
                'timestep': i,
                'cnt_k': None,
                'sty_q': None,
                'num_styles': num_styles,
                'pi_style_total': pi_style_total,
            }
        }

        cnt_feat = cnt_feats[i]
        style_feats_at_i = [style_feats[i] for style_feats in style_feats_list]

        if cnt_feat is None or any(sf is None for sf in style_feats_at_i):
            merged_maps.append(feat_map)
            continue

        style1_feat = style_feats_at_i[0]
        ori_keys = cnt_feat.keys()

        for ori_key in ori_keys:
            # content query를 기본 q key로 사용 (gamma로 live q와 혼합됨)
            if ori_key.endswith('q'):
                feat_map[ori_key] = cnt_feat[ori_key].clone().detach() if torch.is_tensor(cnt_feat[ori_key]) else cnt_feat[ori_key]

            # style1 key/value는 기존 기본 key를 재사용해 backward compatibility 유지
            if ori_key.endswith('k') or ori_key.endswith('v'):
                feat_map[ori_key] = style1_feat[ori_key].clone().detach() if torch.is_tensor(style1_feat[ori_key]) else style1_feat[ori_key]

            # content key/value는 별도 key로 저장 (content branch용)
            if ori_key.endswith('k') or ori_key.endswith('v'):
                feat_map[ori_key + '_cnt'] = cnt_feat[ori_key].clone().detach() if torch.is_tensor(cnt_feat[ori_key]) else cnt_feat[ori_key]

            # style2..styleN key/value는 동적 suffix key로 저장
            if ori_key.endswith('k') or ori_key.endswith('v'):
                for style_idx_one_based in range(2, num_styles + 1):
                    feat_src = style_feats_at_i[style_idx_one_based - 1][ori_key]
                    dyn_key = f"{ori_key}_sty{style_idx_one_based}"
                    feat_map[dyn_key] = feat_src.clone().detach() if torch.is_tensor(feat_src) else feat_src

            # (디버그/호환용) style q도 저장은 해두되, 현재 attention N-style 경로에서는 직접 사용하지 않습니다.
            if ori_key.endswith('q'):
                for style_idx_one_based in range(1, num_styles + 1):
                    feat_src = style_feats_at_i[style_idx_one_based - 1][ori_key]
                    dyn_key = f"{ori_key}_sty{style_idx_one_based}"
                    feat_map[dyn_key] = feat_src.clone().detach() if torch.is_tensor(feat_src) else feat_src

        merged_maps.append(feat_map)

    return merged_maps

def adain(cnt_feat, sty_feat):
    with torch.no_grad():
        cnt_mean = cnt_feat.mean(dim=[0, 2, 3], keepdim=True)
        cnt_std = cnt_feat.std(dim=[0, 2, 3], keepdim=True)
        sty_mean = sty_feat.mean(dim=[0, 2, 3], keepdim=True)
        sty_std = sty_feat.std(dim=[0, 2, 3], keepdim=True)
        return ((cnt_feat - cnt_mean) / cnt_std) * sty_std + sty_mean

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda().eval()
    return model

def clear_all_caches(model):
    """모든 모듈의 캐시와 임시 속성 정리"""
    for m in model.modules():
        # CrossAttention 캐시 정리
        if isinstance(m, CrossAttention):
            if hasattr(m, "mask_cache"):
                m.mask_cache = {}
            if hasattr(m, 'cnt_name'):
                delattr(m, 'cnt_name')
                
        # ResBlock 캐시 정리
        if m.__class__.__name__.endswith("ResBlock"):
            attrs_to_remove = []
            for attr_name in dir(m):
                if not attr_name.startswith('_') and not callable(getattr(m, attr_name)):
                    if any(keyword in attr_name for keyword in ['feat', 'cache', 'timestep', 'inject']):
                        attrs_to_remove.append(attr_name)
            
            for attr_name in attrs_to_remove:
                try:
                    delattr(m, attr_name)
                except:
                    pass

def move_feat_maps_to_device_inplace(feat_maps, device):
    """GPU로 이동하고 CPU 버전을 즉시 해제"""
    for i, f in enumerate(feat_maps):
        if isinstance(f, dict):
            keys_to_update = []
            for k, v in f.items():
                if torch.is_tensor(v) and v.device != device:
                    keys_to_update.append(k)
            
            for k in keys_to_update:
                v = f[k]
                f[k] = v.to(device, non_blocking=False)
                del v
                
        elif torch.is_tensor(f) and f.device != device:
            feat_maps[i] = f.to(device, non_blocking=False)
    
    gc.collect()
    return feat_maps

def restore_original_forwards(unet):
    """원래 forward 메서드로 복원"""
    for block_id in range(6, 12):
        if block_id >= len(unet.output_blocks):
            break
        for module in reversed(unet.output_blocks[block_id]):
            if module.__class__.__name__.endswith("ResBlock"):
                if hasattr(module, '_original_forward_backup'):
                    module._forward = module._original_forward_backup
                # 모든 동적 속성 제거
                attrs_to_remove = ['block_id', 'ri_timestep', 'cnt_feat', 'schedule']
                for attr in attrs_to_remove:
                    if hasattr(module, attr):
                        delattr(module, attr)
                break

def backup_original_forwards(unet):
    """패치 전 원래 forward 메서드들을 백업"""
    for block_id in range(6, 12):
        if block_id >= len(unet.output_blocks):
            break
        for module in reversed(unet.output_blocks[block_id]):
            if module.__class__.__name__.endswith("ResBlock"):
                if not hasattr(module, '_original_forward_backup'):
                    module._original_forward_backup = module._forward
                break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ddim_inv_steps', type=int, default=50)
    parser.add_argument('--save_feat_steps', type=int, default=50)
    parser.add_argument('--start_step', type=int, default=49)
    parser.add_argument('--ddim_eta', type=float, default=0.0)
    # CLI에서 명시하지 않으면 model_config의 inference 기본값을 사용합니다.
    parser.add_argument('--H', type=int, default=None)
    parser.add_argument('--W', type=int, default=None)
    parser.add_argument('--C', type=int, default=4)
    parser.add_argument('--f', type=int, default=8)
    parser.add_argument('--T', type=float, default=1.5)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--pi_style_total', type=float, default=0.9, help='전체 style 질량 비율 (content 질량은 1-pi_style_total)')
    parser.add_argument("--attn_layer", type=str, default='6,7,8,9,10,11')
    parser.add_argument('--model_config', type=str, default='models/ldm/stable-diffusion-v1/v1-inference.yaml')
    parser.add_argument('--precomputed', type=str, default='./precomputed_feats_k')
    parser.add_argument('--ckpt', type=str, default='models/ldm/stable-diffusion-v1/model.ckpt')
    parser.add_argument('--precision', type=str, default='autocast')
    parser.add_argument('--output_path', type=str, default='output_dk')
    parser.add_argument("--without_init_adain", action='store_true')
    parser.add_argument("--without_attn_injection", action='store_true')
    parser.add_argument("--ratio", default=0.5, type=float)
    parser.add_argument("--seed", default=22, type=int)
    parser.add_argument('--data_root', type=str, default='./data_vis')
    parser.add_argument('--meta_file', type=str, default='_dataset.txt', help='run_ori.py 전용 메타 파일 (content + N style 토큰)')
    opt = parser.parse_args()

    seed_everything(opt.seed)
    os.makedirs(opt.output_path, exist_ok=True)
    if len(opt.precomputed) > 0:
        os.makedirs(opt.precomputed, exist_ok=True)

    model_config = OmegaConf.load(f"{opt.model_config}")
    # model_config 기본값 + CLI override 규칙으로 최종 실행 해상도를 결정합니다.
    opt.H, opt.W = resolve_runtime_resolution(opt, model_config)
    validate_runtime_resolution(opt.H, opt.W, opt.f)
    print(f"Runtime resolution: H={opt.H}, W={opt.W}")

    # 해상도별 precomputed feature를 분리해 shape 충돌을 방지합니다.
    precomputed_dir = get_resolution_scoped_precomputed_dir(opt.precomputed, opt.H, opt.W)
    os.makedirs(precomputed_dir, exist_ok=True)

    model = load_model_from_config(model_config, f"{opt.ckpt}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=opt.save_feat_steps, ddim_eta=opt.ddim_eta, verbose=False)
    print("DDIM timesteps:", sampler.ddim_timesteps) 
    
    time_range = np.flip(sampler.ddim_timesteps)
    idx_time_dict = {t:i for i,t in enumerate(time_range)}
    time_idx_dict = {i:t for i,t in enumerate(time_range)}

    uc = model.get_learned_conditioning([""])
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
    
    
    # content + N styles 입력을 직접 해석합니다.
    samples_meta = parse_multistyle_meta_samples(opt.data_root, opt.meta_file)
    print(f"Loaded {len(samples_meta)} samples from meta: {resolve_meta_path(opt.data_root, opt.meta_file)}")

    unet_model = model.model.diffusion_model
    configure_cross_attention_runtime_resolution(unet_model, opt.H, opt.W, opt.f)
    
    # 원래 forward 백업
    backup_original_forwards(unet_model)
    if hasattr(model, "model_ema"):
        configure_cross_attention_runtime_resolution(model.model_ema.diffusion_model, opt.H, opt.W, opt.f)
        backup_original_forwards(model.model_ema.diffusion_model)
    
    begin = time.time()
    
    def residual_injection_callback(step_idx):
        t = sampler.ddim_timesteps[step_idx]
        for block_id in range(6, 12):
            if block_id >= len(unet_model.output_blocks):
                break
            for module in reversed(unet_model.output_blocks[block_id]):
                if module.__class__.__name__.endswith("ResBlock"):
                    module.ri_timestep = int(t)
                    break

        if hasattr(model, "model_ema"):
            ema_unet = model.model_ema.diffusion_model
            for block_id in range(6, 12):
                if block_id >= len(ema_unet.output_blocks):
                    break
                for module in reversed(ema_unet.output_blocks[block_id]):
                    if module.__class__.__name__.endswith("ResBlock"):
                        module.ri_timestep = int(t)
                        break

    for batch_idx, sample_meta in enumerate(samples_meta):
        print(f"Processing image {batch_idx+1}/{len(samples_meta)}")

        cnt_path = sample_meta["cnt_path"]
        style_paths = sample_meta["style_paths"]
        if len(style_paths) < 1:
            raise ValueError(f"style path가 1개 이상이어야 합니다: {sample_meta}")
        # 분기 기준은 해상도가 아니라 "스타일 개수"입니다.
        # - style 2개: 기존 2-style 경로(legacy) 유지
        # - 그 외(style 1개 또는 3개 이상): N-style 경로 사용
        use_legacy_two_style_path = (len(style_paths) == 2)
        
        with torch.no_grad():
            # 마스크 적용
            for m in unet_model.modules():
                if isinstance(m, CrossAttention):
                    m.cnt_name = cnt_path
            
            cnt_base = os.path.splitext(os.path.basename(cnt_path))[0]
            style_bases = [os.path.splitext(os.path.basename(p))[0] for p in style_paths]

            print(f"Iteration {batch_idx}, CPU RAM: {get_cpu_mem():.2f} MB")
            
            # style feature들을 메타 파일의 순서대로 로드합니다.
            style_z_enc_list = []
            style_feat_list = []
            for style_path in style_paths:
                style_base = os.path.splitext(os.path.basename(style_path))[0]
                style_feat_name = os.path.join(precomputed_dir, f"{style_base}_sty.pkl")
                if not os.path.isfile(style_feat_name):
                    raise FileNotFoundError(f"Style precomputed feature not found: {style_feat_name}")

                style_z_enc, style_feat = load_minimal_feat(style_feat_name)
                validate_precomputed_z_enc_shape(style_feat_name, style_z_enc, opt.H, opt.W, opt.f)
                style_z_enc_list.append(style_z_enc)
                style_feat_list.append(style_feat)

            # content feature 로드
            cnt_feat_name = os.path.join(precomputed_dir, f"{cnt_base}_cnt.pkl")
            cnt_z_enc, cnt_feat = None, None
            if os.path.isfile(cnt_feat_name):
                cnt_z_enc, cnt_feat = load_minimal_feat(cnt_feat_name)
                validate_precomputed_z_enc_shape(cnt_feat_name, cnt_z_enc, opt.H, opt.W, opt.f)
            else:
                raise FileNotFoundError(f"Content precomputed feature not found: {cnt_feat_name}")
            
            print(f"After loading, CPU RAM: {get_cpu_mem():.2f} MB")
            
            # High-frequency enhancement
            schedule = make_content_injection_schedule(sampler.ddim_timesteps)
            
            # cnt_feat 깊은 복사본으로 patch
            cnt_feat_copy = copy.deepcopy(cnt_feat)
            patch_decoder_resblocks_h_and_cnt_hf(unet_model, schedule, cnt_feat_copy, ratio=opt.ratio)
            if hasattr(model, "model_ema"):
                patch_decoder_resblocks_h_and_cnt_hf(model.model_ema.diffusion_model, 
                                                    schedule, cnt_feat_copy, ratio=opt.ratio)
            del cnt_feat_copy
            gc.collect()
            
            # AdaIN blending
            if opt.without_init_adain:
                adain_z_enc = cnt_z_enc.clone().detach()
            else:
                if use_legacy_two_style_path:
                    # ===== 기존 2-style 경로 유지 =====
                    # 기존 구현과 동일하게 단일 바이너리 마스크(_mask.npy)를 사용합니다.
                    # mask=1 영역은 style1, mask=0 영역은 style2로 초기 latent를 구성합니다.
                    legacy_mask_path = os.path.splitext(cnt_path)[0] + "_mask.npy"
                    if not os.path.isfile(legacy_mask_path):
                        raise FileNotFoundError(
                            "2-style legacy 경로에서는 단일 마스크 파일이 필요합니다: "
                            f"{legacy_mask_path}"
                        )

                    cnt_mask_img = prepare_mask_tensor_for_runtime(
                        legacy_mask_path,
                        target_h=opt.H,
                        target_w=opt.W,
                        device=device,
                    )
                    cnt_mask_latent = F.interpolate(
                        cnt_mask_img,
                        size=(cnt_z_enc.shape[2], cnt_z_enc.shape[3]),
                        mode="bilinear",
                        align_corners=False,
                    ).to(device=device, dtype=cnt_z_enc.dtype)

                    cnt_z_enc_dev = cnt_z_enc.to(device)
                    sty1_adain = adain(cnt_z_enc_dev, style_z_enc_list[0].to(device))
                    sty2_adain = adain(cnt_z_enc_dev, style_z_enc_list[1].to(device))
                    adain_z_enc = (
                        cnt_mask_latent * sty1_adain +
                        (1.0 - cnt_mask_latent) * sty2_adain
                    ).clone().detach()

                    del cnt_mask_img, cnt_mask_latent, cnt_z_enc_dev, sty1_adain, sty2_adain
                    torch.cuda.empty_cache()
                else:
                    # ===== N-style 경로 =====
                    # 멀티 마스크 파일 규약:
                    #   content_001.png -> content_001_mask0.npy, content_001_mask1.npy, ...
                    # 메타 파일의 style 순서와 mask index를 동일하게 맞춥니다.
                    style_mask_stack_img = load_style_mask_stack_for_runtime(
                        cnt_path=cnt_path,
                        target_h=opt.H,
                        target_w=opt.W,
                        num_styles=len(style_z_enc_list),
                        device=device,
                    )  # (N,1,H,W)

                    style_weights_latent, style_coverage_latent = build_style_weight_maps_for_latent_from_masks(
                        style_mask_stack_img,
                        cnt_z_enc.shape[2],
                        cnt_z_enc.shape[3],
                    )
                    style_weights_latent = style_weights_latent.to(device=device, dtype=cnt_z_enc.dtype)      # (N,1,h,w)
                    style_coverage_latent = style_coverage_latent.to(device=device, dtype=cnt_z_enc.dtype)    # (1,1,h,w)

                    # content latent를 기준으로 각 style latent에 AdaIN을 적용한 뒤,
                    # 멀티 마스크를 이용해 스타일들을 혼합합니다.
                    # (마스크 합이 항상 1이면 coverage는 1이 되어 순수 스타일 혼합만 남습니다.)
                    adain_components = []
                    cnt_z_enc_dev = cnt_z_enc.to(device)
                    for style_z_enc in style_z_enc_list:
                        adain_components.append(adain(cnt_z_enc_dev, style_z_enc.to(device)))

                    adain_stack = torch.stack(adain_components, dim=0)  # (N, B, C, h, w)
                    style_weights_latent = style_weights_latent.unsqueeze(1)  # (N,1,1,h,w)
                    style_coverage_latent = style_coverage_latent.unsqueeze(1)  # (1,1,1,h,w)

                    styled_latent = (style_weights_latent * adain_stack).sum(dim=0)
                    adain_z_enc = (
                        styled_latent * style_coverage_latent.squeeze(0) +
                        cnt_z_enc_dev * (1.0 - style_coverage_latent.squeeze(0))
                    ).clone().detach()

                    del style_mask_stack_img, style_weights_latent, style_coverage_latent
                    del adain_components, adain_stack, styled_latent, cnt_z_enc_dev
                    torch.cuda.empty_cache()
            
            # z_enc 삭제
            del style_z_enc_list, cnt_z_enc
            gc.collect()
            
            # Feature merge
            if use_legacy_two_style_path:
                # 2-style는 기존 feature merge 스키마를 유지하여 attention의 legacy 경로로 들어가게 합니다.
                merged_feat_maps = feat_merge_2sty_clean(
                    opt,
                    cnt_feat,
                    style_feat_list[0],
                    style_feat_list[1],
                    start_step=opt.start_step,
                )
            else:
                merged_feat_maps = feat_merge_nsty_clean(
                    opt,
                    cnt_feat,
                    style_feat_list,
                    pi_style_total=opt.pi_style_total,
                    start_step=opt.start_step,
                )
            
            # 원본 feat 삭제
            del style_feat_list, cnt_feat
            gc.collect()
            
            # GPU로 이동
            merged_feat_maps = move_feat_maps_to_device_inplace(merged_feat_maps, device)
            
            print(f"Before inference, CPU RAM: {get_cpu_mem():.2f} MB")
            
            # Inference
            samples_ddim, _ = sampler.sample(
                S=opt.save_feat_steps,
                batch_size=1,
                shape=shape,
                verbose=False,
                unconditional_conditioning=uc,
                eta=opt.ddim_eta,
                x_T=adain_z_enc,
                injected_features=merged_feat_maps,
                start_step=opt.start_step,
                callback=residual_injection_callback,
            )
            
            # 저장
            if use_legacy_two_style_path:
                # 기존 결과 파일명 규칙 유지: content_style1_style2.png
                result_name = f"{cnt_base}_{style_bases[0]}_{style_bases[1]}.png"
            else:
                # N-style은 style 개수가 가변이므로 메타 파일 순서대로 basename을 이어서 구성합니다.
                result_name = f"{cnt_base}__{'__'.join(style_bases)}.png"
            save_img_from_sample(model, samples_ddim, os.path.join(opt.output_path, result_name))
        
        # Inference 후 패치 복원
        restore_original_forwards(unet_model)
        if hasattr(model, "model_ema"):
            restore_original_forwards(model.model_ema.diffusion_model)
        
        # 메모리 정리
        del samples_ddim, adain_z_enc, merged_feat_maps
        
        # 모든 캐시 정리
        clear_all_caches(unet_model)
        if hasattr(model, "model_ema"):
            clear_all_caches(model.model_ema.diffusion_model)
        
        # 가비지 컬렉션
        for _ in range(3):
            gc.collect()
        
        torch.cuda.empty_cache()
        
        # Linux 메모리 강제 반환
        if sys.platform == 'linux':
            try:
                import ctypes
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
            except:
                pass

    print(f"Total time: {time.time() - begin:.2f} seconds")

if __name__ == "__main__":
    main()
