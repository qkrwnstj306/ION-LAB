import argparse
import os
import torch
from torch import autocast
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from contextlib import nullcontext
import copy
import torchvision.transforms as transforms
import pickle
import time

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from dataset import TripleImageDataset, SingleImageDataset

# # residual injection
#from high_frequency_final import high_pass_filter

feat_maps = []

# 이미지 파일 탐색 시 사용하는 확장자 목록입니다.
VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# run_ori.py와 동일한 해상도 정책을 사용해야 precomputed feature shape가 정확히 맞음.
SUPPORTED_INFERENCE_HW = {
    (512, 512),
    (512, 768),
    (512, 1024),
    (768, 512),
    (768, 768),
    (768, 1024),
    (1024, 512),
    (1024, 768),
    (1024, 1024),
}


def resolve_runtime_resolution(opt, model_config):
    """
    실행 해상도(H, W)를 model_config 기본값 + CLI override 규칙으로 결정.
    generate_pkl_only.py와 run_ori.py가 반드시 같은 규칙을 써야 함.
    """
    cfg_h, cfg_w = None, None
    if model_config is not None and "inference" in model_config:
        inference_cfg = model_config.inference
        cfg_h = inference_cfg.get("default_h", None)
        cfg_w = inference_cfg.get("default_w", None)
        if (cfg_h is None or cfg_w is None) and "default_hw" in inference_cfg:
            default_hw = inference_cfg.get("default_hw")
            if default_hw is not None and len(default_hw) == 2:
                cfg_h = default_hw[0]
                cfg_w = default_hw[1]

    h = opt.H if opt.H is not None else (cfg_h if cfg_h is not None else 512)
    w = opt.W if opt.W is not None else (cfg_w if cfg_w is not None else 512)
    return int(h), int(w)


def validate_runtime_resolution(h, w, f):
    """
    지원 해상도 조합과 latent downsampling 계수(f) 정합성 검사.
    """
    if (h, w) not in SUPPORTED_INFERENCE_HW:
        raise ValueError(
            f"지원하지 않는 해상도 조합입니다: ({h}, {w}). "
            f"지원 해상도: {sorted(SUPPORTED_INFERENCE_HW)}"
        )
    if h % f != 0 or w % f != 0:
        raise ValueError(f"H/W는 f={f}의 배수여야 합니다. 현재 H={h}, W={w}, f={f}")


def get_resolution_scoped_precomputed_dir(base_dir, h, w):
    """
    feature pkl 저장 경로 규칙:
    - 512x512: 기존 실험 자산과 호환되도록 base_dir 그대로 사용
    - 그 외 지원 해상도: {base_dir}/{H}x{W} 하위 폴더 사용
    """
    if (h, w) == (512, 512):
        return base_dir
    return os.path.join(base_dir, f"{h}x{w}")


def center_crop_tensor_hw(x, target_h, target_w):
    """
    4D tensor(B, C, H, W)를 target_h x target_w로 중앙 crop합니다.
    dataset.py는 그대로 두고, 여기서 실행 해상도에 맞춰 최종 정렬합니다.
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


def load_img(path, target_h=512, target_w=512):
    image = Image.open(path).convert("RGB")
    x, y = image.size
    print(f"Loaded input image of size ({x}, {y}) from {path}")
    # 먼저 정사각형(max(H,W))으로 맞춘 뒤 중앙 crop으로 최종 직사각형을 만듦.
    square_side = max(target_h, target_w)
    image = transforms.CenterCrop(min(x, y))(image)
    image = image.resize((square_side, square_side), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = 2. * image - 1.
    image = center_crop_tensor_hw(image, target_h, target_w)
    return image


def list_image_paths(folder, recursive=False):
    """
    지정한 폴더에서 이미지 파일 경로를 수집합니다.
    - recursive=True: 하위 폴더까지 모두 탐색 (style 폴더용)
    - recursive=False: 현재 폴더만 탐색 (기본)
    """
    if not folder or not os.path.isdir(folder):
        return []

    image_paths = []
    if recursive:
        for root, _, files in os.walk(folder):
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                if ext in VALID_IMAGE_EXTENSIONS:
                    image_paths.append(os.path.join(root, name))
    else:
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in VALID_IMAGE_EXTENSIONS:
                image_paths.append(path)

    image_paths.sort()
    return image_paths

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnt', default = None, help='Content image folder path')
    parser.add_argument('--sty', default = None, help='Style image folder path')
    parser.add_argument('--ddim_inv_steps', type=int, default=50)
    parser.add_argument('--save_feat_steps', type=int, default=50, help='DDIM eta')
    parser.add_argument('--start_step', type=int, default=49)
    parser.add_argument('--ddim_eta', type=float, default=0.0)
    # CLI에서 생략되면 model_config의 inference 기본값을 사용.
    parser.add_argument('--H', type=int, default=None)
    parser.add_argument('--W', type=int, default=None)
    parser.add_argument('--C', type=int, default=4)
    parser.add_argument('--f', type=int, default=8)
    parser.add_argument("--attn_layer", type=str, default='6,7,8,9,10,11', help='injection attention feature layers')
    parser.add_argument('--model_config', type=str, default='models/ldm/stable-diffusion-v1/v1-inference.yaml')
    parser.add_argument('--precomputed', type=str, default='./precomputed_feats_k')
    #parser.add_argument('--precomputed', type=str, default='./precomputed_feats_k')
    parser.add_argument('--ckpt', type=str, default='models/ldm/stable-diffusion-v1/model.ckpt')
    parser.add_argument('--precision', type=str, default='autocast', help='choices: ["full", "autocast"]')
    parser.add_argument("--seed", default=22, type=int)
    parser.add_argument('--data_root', type=str, default='./data_vis')
    opt = parser.parse_args()

    seed_everything(opt.seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_config = OmegaConf.load(opt.model_config)
    # run_ori.py와 동일한 규칙으로 해상도를 결정해야 pkl shape mismatch를 피할 수 있음.
    opt.H, opt.W = resolve_runtime_resolution(opt, model_config)
    validate_runtime_resolution(opt.H, opt.W, opt.f)
    print(f"Runtime resolution for precompute: H={opt.H}, W={opt.W}")

    feat_path_root = get_resolution_scoped_precomputed_dir(opt.precomputed, opt.H, opt.W)
    os.makedirs(feat_path_root, exist_ok=True)

    model = load_model_from_config(model_config, opt.ckpt)
    model = model.to(device)
    unet_model = model.model.diffusion_model
    sampler = DDIMSampler(model)
    
    for name, module in unet_model.named_modules():
        if module.__class__.__name__ == "CrossAttention":
            module.gen_pkl = True
            # 현재 스크립트는 gen_pkl 모드에서 주로 동작하지만, 동일 모델 객체를 재사용할 수 있으므로
            # 직사각형 기준 latent 해상도도 함께 주입해 둡니다.
            module.base_latent_hw = (opt.H // opt.f, opt.W // opt.f)
            print(f"Set gen_pkl=True for {name}")

    self_attn_output_block_indices = list(map(int, opt.attn_layer.split(',')))
    ddim_inversion_steps = opt.ddim_inv_steps
    save_feature_timesteps = ddim_steps = opt.save_feat_steps
    

    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=opt.ddim_eta, verbose=False) 
    time_range = np.flip(sampler.ddim_timesteps)

    idx_time_dict = {}
    time_idx_dict = {}
    for i, t in enumerate(time_range):
        idx_time_dict[t] = i
        time_idx_dict[i] = t
    
    global feat_maps
    # 이미지마다 동일한 버퍼 템플릿을 재사용하기 위해 길이를 현재 DDIM step 수에 맞춰 생성합니다.
    def reset_feat_maps_buffer():
        global feat_maps
        feat_maps = [{'config': {'T': 1.5}} for _ in range(len(time_range))]

    reset_feat_maps_buffer()

    def ddim_sampler_callback(pred_x0, xt, i):
        save_feature_maps_callback(i)
        save_feature_map_z(xt, 'z_enc', i)
    
    def save_feature_map(feature_map, filename, time):
        global feat_maps
        cur_idx = idx_time_dict[time]
        feat_maps[cur_idx][f"{filename}"] = feature_map

    def save_feature_maps(blocks, i, feature_type="input_block"):
        block_idx = 0
        for block_idx, block in enumerate(blocks):
            if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
                if block_idx in self_attn_output_block_indices:
                    # self-attn
                    q = block[1].transformer_blocks[0].attn1.q.detach().cpu()
                    k = block[1].transformer_blocks[0].attn1.k.detach().cpu()
                    v = block[1].transformer_blocks[0].attn1.v.detach().cpu()
                    save_feature_map(q, f"{feature_type}_{block_idx}_self_attn_q", i)
                    save_feature_map(k, f"{feature_type}_{block_idx}_self_attn_k", i)
                    save_feature_map(v, f"{feature_type}_{block_idx}_self_attn_v", i)
            block_idx += 1
            
    def save_feature_maps_callback(i):
        save_feature_maps(unet_model.output_blocks , i, "output_block")

    def save_feature_map_z(xt, name, time):
        global feat_maps
        cur_idx = idx_time_dict[time]
        feat_maps[cur_idx][name] = xt.detach().cpu()

    def residual_injection_callback(pred_x0, xt, t):
        # feature map 저장
        save_feature_maps_callback(t)
        save_feature_map_z(xt, 'z_enc', t)

        t_int = int(t)
        if t_int not in residuals_all:
            residuals_all[t_int] = {}

        for block_id in range(6, 12):
            if block_id >= len(unet_model.output_blocks):
                break

            for module in reversed(unet_model.output_blocks[block_id]):
                if module.__class__.__name__.endswith("ResBlock"):
                    # if hasattr(module, 'out_skip') and module.out_skip is not None:
                    #     skip = module.out_skip.detach().cpu()
                    #     #skip_hf = high_pass_filter(skip, radius=6)
                    #     key_skip = f"output_block_{block_id}_cnt_skip"
                    #     residuals_all[t_int][key_skip] = skip
                    #     print(f"[Callback] t={t_int}, saved {key_skip}")

                    if hasattr(module, 'out_h') and module.out_h is not None:
                        h = module.out_h.detach().cpu()
                        #h_hf = high_pass_filter(h, radius=6)
                        key_h = f"output_block_{block_id}_cnt_h"
                        save_feature_map(h, f"{key_h}", t_int)
                        #feat_maps[t_int][f"{key_h}"] = h
                        # residuals_all[t_int][key_h] = h
                        print(f"[Callback] t={t_int}, saved {key_h}")

                    # if hasattr(module, 'out_merged') and module.out_merged is not None:
                    #     key_full = f"output_block_{block_id}_residual"
                    #     residuals_all[t_int][key_full] = module.out_merged.detach().cpu()
                    #     print(f"[Callback] t={t_int}, saved {key_full}")

                    break  # 마지막 ResBlock만 처리

                    
    start_step = opt.start_step
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    uc = model.get_learned_conditioning([""])
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

    # ==========================
    # content/style precompute 대상 경로 결정
    # ==========================
    # --cnt/--sty를 지정하면 해당 경로를 우선 사용하고,
    # 미지정 시에는 data_root 기준 기본 폴더를 사용합니다.
    cnt_folder = opt.cnt if opt.cnt is not None else os.path.join(opt.data_root, "cnt")
    sty_folder = opt.sty if opt.sty is not None else os.path.join(opt.data_root, "sty")

    # content는 보통 cnt 폴더 바로 아래에 있으므로 기본은 비재귀,
    # style은 char/back/extra 등 하위 폴더 구성을 지원하기 위해 재귀 탐색합니다.
    cnt_image_paths = list_image_paths(cnt_folder, recursive=False)
    sty_image_paths = list_image_paths(sty_folder, recursive=True)

    if not cnt_image_paths:
        print(f"[Info] No content images found in: {cnt_folder}")
    if not sty_image_paths:
        print(f"[Info] No style images found in: {sty_folder}")

    # basename 기반으로 pkl 이름을 만들기 때문에, 서로 다른 파일이 같은 basename이면 덮어쓰기됩니다.
    # run_ori.py도 basename 기준으로 찾으므로 여기서 명시적으로 잡아주는 편이 안전합니다.
    seen_output_to_source = {}

    # ===== STYLE FEATURE ( *_sty.pkl ) =====
    for sty_path in sty_image_paths:
        sty_name = os.path.basename(sty_path)
        sty_feat_name = os.path.join(
            feat_path_root, os.path.splitext(sty_name)[0] + '_sty.pkl'
        )

        prev_src = seen_output_to_source.get(sty_feat_name)
        if prev_src is not None and os.path.abspath(prev_src) != os.path.abspath(sty_path):
            raise ValueError(
                f"Style basename 충돌로 동일 pkl 경로가 겹칩니다:\n"
                f" - {prev_src}\n - {sty_path}\n"
                f" -> {sty_feat_name}"
            )
        seen_output_to_source[sty_feat_name] = sty_path

        if os.path.isfile(sty_feat_name):
            print(f"Precomputed style feature exists: {sty_feat_name}")
            continue

        # 이미지마다 feature 버퍼를 초기화해야 이전 이미지의 잔여 키가 섞이지 않습니다.
        reset_feat_maps_buffer()
        init_sty = load_img(sty_path, target_h=opt.H, target_w=opt.W).to(device)
        init_sty_latent = model.get_first_stage_encoding(model.encode_first_stage(init_sty))
        sty_z_enc, _ = sampler.encode_ddim(
            init_sty_latent.clone(),
            num_steps=ddim_inversion_steps,
            unconditional_conditioning=uc,
            end_step=time_idx_dict[ddim_inversion_steps - 1 - start_step],
            callback_ddim_timesteps=save_feature_timesteps,
            img_callback=ddim_sampler_callback
        )
        with open(sty_feat_name, 'wb') as f:
            pickle.dump(copy.deepcopy(feat_maps), f)
        print(f"Saved style feature: {sty_feat_name}")

    # ===== CONTENT FEATURE ( *_cnt.pkl ) =====
    for cnt_path in cnt_image_paths:
        cnt_name = os.path.basename(cnt_path)
        cnt_feat_name = os.path.join(
            feat_path_root, os.path.splitext(cnt_name)[0] + '_cnt.pkl'
        )

        prev_src = seen_output_to_source.get(cnt_feat_name)
        if prev_src is not None and os.path.abspath(prev_src) != os.path.abspath(cnt_path):
            raise ValueError(
                f"Content basename 충돌로 동일 pkl 경로가 겹칩니다:\n"
                f" - {prev_src}\n - {cnt_path}\n"
                f" -> {cnt_feat_name}"
            )
        seen_output_to_source[cnt_feat_name] = cnt_path

        if os.path.isfile(cnt_feat_name):
            print(f"Precomputed content feature exists: {cnt_feat_name}")
            continue

        reset_feat_maps_buffer()
        init_cnt = load_img(cnt_path, target_h=opt.H, target_w=opt.W).to(device)
        init_cnt_latent = model.get_first_stage_encoding(model.encode_first_stage(init_cnt))
        residuals_all = {}
        cnt_z_enc, _ = sampler.encode_ddim(
            init_cnt_latent.clone(),
            num_steps=ddim_inversion_steps,
            unconditional_conditioning=uc,
            end_step=time_idx_dict[ddim_inversion_steps - 1 - start_step],
            callback_ddim_timesteps=save_feature_timesteps,
            img_callback=residual_injection_callback
        )
        with open(cnt_feat_name, 'wb') as f:
            pickle.dump(copy.deepcopy(feat_maps), f)
        print(f"Saved content feature: {cnt_feat_name}")

if __name__ == "__main__":
    main()
