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
    parser.add_argument('--H', type=int, default=512)
    parser.add_argument('--W', type=int, default=512)
    parser.add_argument('--C', type=int, default=4)
    parser.add_argument('--f', type=int, default=8)
    parser.add_argument('--T', type=float, default=1.5)
    parser.add_argument('--gamma', type=float, default=0.5)
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
    opt = parser.parse_args()

    seed_everything(opt.seed)
    os.makedirs(opt.output_path, exist_ok=True)
    if len(opt.precomputed) > 0:
        os.makedirs(opt.precomputed, exist_ok=True)

    model_config = OmegaConf.load(f"{opt.model_config}")
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
    
    dataset = TripleImageDataset(opt.data_root, "yr2_dataset.txt", image_size=opt.H, device=device)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    unet_model = model.model.diffusion_model
    
    # 원래 forward 백업
    backup_original_forwards(unet_model)
    if hasattr(model, "model_ema"):
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

    for batch_idx, batch in enumerate(dataloader):
        print(f"Processing image {batch_idx+1}/{len(dataset)}")
        
        cnt_img, char_img, back_img, cnt_path, char_path, back_path = batch
        if isinstance(cnt_path, (list, tuple)): cnt_path = cnt_path[0]
        if isinstance(char_path, (list, tuple)): char_path = char_path[0]
        if isinstance(back_path, (list, tuple)): back_path = back_path[0]

        cnt_img = cnt_img.to(device, non_blocking=True)
        char_img = char_img.to(device, non_blocking=True)
        back_img = back_img.to(device, non_blocking=True)
        
        with torch.no_grad():
            # 마스크 적용
            for m in unet_model.modules():
                if isinstance(m, CrossAttention):
                    m.cnt_name = cnt_path
            
            cnt_base = os.path.splitext(os.path.basename(cnt_path))[0]
            char_base = os.path.splitext(os.path.basename(char_path))[0]
            back_base = os.path.splitext(os.path.basename(back_path))[0]

            print(f"Iteration {batch_idx}, CPU RAM: {get_cpu_mem():.2f} MB")
            
            # 최소 데이터만 로드
            char_feat_name = os.path.join(opt.precomputed, f"{char_base}_sty.pkl")
            char_z_enc, char_feat = None, None
            if os.path.isfile(char_feat_name):
                char_z_enc, char_feat = load_minimal_feat(char_feat_name)
    
            back_feat_name = os.path.join(opt.precomputed, f"{back_base}_sty.pkl")
            back_z_enc, back_feat = None, None
            if os.path.isfile(back_feat_name):
                back_z_enc, back_feat = load_minimal_feat(back_feat_name)

            cnt_feat_name = os.path.join(opt.precomputed, f"{cnt_base}_cnt.pkl")
            cnt_z_enc, cnt_feat = None, None
            if os.path.isfile(cnt_feat_name):
                cnt_z_enc, cnt_feat = load_minimal_feat(cnt_feat_name)
            
            print(f"After loading, CPU RAM: {get_cpu_mem():.2f} MB")
            
            # High-frequency enhancement
            schedule = make_content_injection_schedule(sampler.ddim_timesteps, alpha=0.4)
            
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
                mask = torch.tensor(np.load(os.path.join(opt.data_root, "cnt", f"{cnt_base}_mask.npy")), 
                                   dtype=torch.float32).to(device)
                mask = mask.unsqueeze(0).unsqueeze(0)
                mask = F.interpolate(mask, size=(cnt_z_enc.shape[2], cnt_z_enc.shape[3]), 
                                    mode="bilinear", align_corners=False)
                mask = mask.expand(-1, cnt_z_enc.shape[1], -1, -1)
                
                adain_z_enc = (mask * adain(cnt_z_enc.to(device), char_z_enc.to(device)) + 
                              (1-mask) * adain(cnt_z_enc.to(device), back_z_enc.to(device))).clone().detach()
                del mask
                torch.cuda.empty_cache()
            
            # z_enc 삭제
            del char_z_enc, back_z_enc, cnt_z_enc
            gc.collect()
            
            # Feature merge
            merged_feat_maps = feat_merge_2sty_clean(opt, cnt_feat, char_feat, back_feat, 
                                                     start_step=opt.start_step)
            
            # 원본 feat 삭제
            del char_feat, back_feat, cnt_feat
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
            result_name = f"{cnt_base}_{char_base}_{back_base}.png"
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