
from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import pickle
import os

from ldm.modules.diffusionmodules.util import checkpoint

## 마스크 적용
from math import sqrt
import numpy as np


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        # print(f"context_dim is exists: {exists(context_dim)}")
        context_dim = default(context_dim, query_dim)
        
        self.scale = dim_head ** -0.5
      
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

        self.attn = None
        self.q = None
        self.k = None
        self.v = None
        #self.qk_sim = None
        
        ## generated pkl
        self.gen_pkl = False

        ## 마스크 적용
        self.sty_name = None
        self.cnt_name = None
        
        ## q, k, sim 저장용
        self.target_t_list = None
        self.layer_id = None

        self.mask_cache = {}
        # run 스크립트에서 주입하는 "기준 latent 해상도"입니다.
        # 예: 512x1024 입력이면 base_latent_hw=(64, 128)
        # attention 계층마다 token 개수(N)가 달라지므로, 이 기준값을 이용해 계층별 (h,w)를 역추론합니다.
        self.base_latent_hw = None

    def _infer_spatial_hw_from_tokens(self, token_count):
        """
        self-attention token 개수(N)를 실제 spatial (h, w)로 복원합니다.
        - 우선 run 스크립트가 넣어준 base_latent_hw를 기준으로 계층 배율(scale)을 역추론
        - 정보가 없으면 기존 square fallback(sqrt(N)) 사용
        """
        base_hw = getattr(self, "base_latent_hw", None)
        if base_hw is not None:
            base_h, base_w = int(base_hw[0]), int(base_hw[1])
            base_tokens = base_h * base_w

            if token_count <= 0:
                raise ValueError(f"잘못된 token_count입니다: {token_count}")

            if base_tokens % token_count == 0:
                area_ratio = base_tokens // token_count
                scale = math.isqrt(area_ratio)

                # U-Net 해상도 변화는 일반적으로 2배 단위이므로, 면적비는 정수 제곱(1,4,16,...)이어야 합니다.
                if scale * scale == area_ratio and scale > 0:
                    if base_h % scale == 0 and base_w % scale == 0:
                        h = base_h // scale
                        w = base_w // scale
                        if h * w == token_count:
                            return h, w

        # [호환 fallback] 기존 정사각형 실험 결과를 깨지 않기 위한 경로.
        # 512x512면 side=64로 기존과 똑같이 수행되는 것.
        side = int(math.isqrt(token_count))
        if side * side == token_count:
            return side, side

        raise ValueError(
            f"token_count={token_count}에서 spatial (h,w)를 복원하지 못했습니다. "
            f"base_latent_hw={base_hw}"
        )

    def _load_and_preprocess_mask(self, mask_path, target_h, target_w, ch, device):
        """
        마스크 npy를 현재 attention 계층 해상도(target_h, target_w)에 맞게 변환합니다.
        crop 없이 resize만 적용하여 전체 마스크 영역을 유지합니다.
        """
        mask = torch.tensor(np.load(mask_path), dtype=torch.float32, device=device)
        mask[mask < 0.5] = -1.0
        mask[mask > 0.5] = 1.0
        mask = mask * ch
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        # 마지막으로 현재 attention 계층 spatial 해상도로 맞춤.
        mask = F.interpolate(mask, size=(target_h, target_w), mode='bilinear', align_corners=False)
        mask = mask.reshape(1, target_h, target_w, 1)
        return mask

    def _compute_sim_with_tau(self, q, k, num_heads, attn_matrix_scale=1.0):
        """
        마스크 보정 없이 style branch logits만 계산합니다.
        N-style 경로에서는 style branch 수가 가변이므로, 기존 2-style 전용 함수와 분리합니다.
        반환 shape: (heads, Nq, Nk)
        """
        qh = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        kh = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", qh, kh)
        sim *= self.scale

        # 기존 코드와 동일한 Tau(T) 스케일링 규칙 유지
        mean = sim.mean(dim=-1, keepdim=True)
        sim = (sim - mean) * attn_matrix_scale + mean
        return sim

    def _load_and_preprocess_label_map(self, mask_path, target_h, target_w, device):
        """
        _mask.npy를 HxW label map(0..N-1)으로 읽고 현재 attention 계층 해상도로 변환합니다.
        crop 없이 nearest resize만 적용하여 전체 라벨 영역을 유지합니다.
        """
        label = torch.tensor(np.load(mask_path), dtype=torch.float32, device=device)
        if label.ndim != 2:
            raise ValueError(f"label map은 2D여야 합니다: {mask_path}, shape={tuple(label.shape)}")

        label = label.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        label = F.interpolate(label, size=(target_h, target_w), mode='nearest')
        return label.squeeze(0).squeeze(0)

    def _load_and_preprocess_style_mask_file(self, mask_path, target_h, target_w, device):
        """
        `_mask{i}.npy` 스타일 마스크 파일을 현재 attention 계층 해상도로 변환합니다.
        반환 shape: (h, w, 1), 값 범위 [0, 1]

        """
        mask = torch.tensor(np.load(mask_path), dtype=torch.float32, device=device)
        if mask.ndim != 2:
            raise ValueError(f"style mask는 2D npy여야 합니다: {mask_path}, shape={tuple(mask.shape)}")

        if mask.max() > 1.0:
            mask = mask / 255.0
        mask = mask.clamp(0.0, 1.0)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        mask = F.interpolate(mask, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return mask.squeeze(0).permute(1, 2, 0).contiguous().clamp_(0.0, 1.0)  # (h,w,1)

    def _load_style_weight_maps_from_label(self, mask_path, target_h, target_w, num_styles, device):
        """
        label map(0..N-1)를 style weight map(one-hot)으로 변환합니다.
        반환 shape: (num_styles, target_h, target_w, 1)

        backward compatibility:
        - num_styles==2인 경우 기존 binary float mask(0~1)도 threshold로 라벨화하여 수용합니다.
        """
        cache_key = (
            f"label_weights::{mask_path}::{target_h}::{target_w}::{num_styles}::"
            f"{getattr(self, 'base_latent_hw', None)}"
        )
        if cache_key in self.mask_cache:
            return self.mask_cache[cache_key]

        label_map = self._load_and_preprocess_label_map(mask_path, target_h, target_w, device)

        if num_styles == 2 and torch.all((label_map >= 0.0) & (label_map <= 1.0)):
            label_map = (label_map >= 0.5).to(torch.long)
        else:
            label_map = torch.round(label_map).to(torch.long)

        label_map = label_map.clamp_min(0).clamp_max(num_styles - 1)
        one_hot = F.one_hot(label_map, num_classes=num_styles).to(dtype=torch.float32)  # (h,w,n)
        weight_maps = one_hot.permute(2, 0, 1).unsqueeze(-1).contiguous()  # (n,h,w,1)

        self.mask_cache[cache_key] = weight_maps
        return weight_maps

    def _load_style_weight_maps_from_mask_files(self, mask_prefix, target_h, target_w, num_styles, device, allow_legacy_label_fallback=True):
        """
        `_mask0.npy`, `_mask1.npy`, ... `_mask{N-1}.npy` 파일을 읽어 style weight map stack을 생성합니다.
        반환 shape: (num_styles, h, w, 1)

        해석 규칙:
        - 각 파일은 해당 style의 raw 마스크(0~1)입니다.
        - 겹침/비어있는 영역 처리는 apply_pi_mass_nway()에서 coverage + 정규화로 처리합니다.
        - `allow_legacy_label_fallback=False`이면 `_mask.npy` fallback을 막고, 멀티 마스크 누락 시 즉시 에러를 냅니다.
        """
        cache_key = (
            f"multi_mask_weights::{mask_prefix}::{target_h}::{target_w}::{num_styles}::"
            f"{getattr(self, 'base_latent_hw', None)}"
        )
        if cache_key in self.mask_cache:
            return self.mask_cache[cache_key]

        mask_paths = [f"{mask_prefix}{style_idx}.npy" for style_idx in range(num_styles)]
        missing_paths = [p for p in mask_paths if not os.path.isfile(p)]
        if missing_paths:
            # 호환성 경로: 기존 label map(_mask.npy)가 남아 있으면 기존 로더를 사용합니다.
            # 단, N-style 멀티 마스크 규약을 명시적으로 사용하는 경우에는 silent fallback을 막기 위해 비활성화할 수 있습니다.
            legacy_label_path = f"{mask_prefix}.npy"
            if allow_legacy_label_fallback and os.path.isfile(legacy_label_path):
                weight_maps = self._load_style_weight_maps_from_label(
                    legacy_label_path,
                    target_h=target_h,
                    target_w=target_w,
                    num_styles=num_styles,
                    device=device,
                )
                self.mask_cache[cache_key] = weight_maps
                return weight_maps

            raise FileNotFoundError(
                "N-style 멀티 마스크 파일이 누락되었습니다. "
                f"기대 파일 예시: {mask_paths[0]} ... {mask_paths[-1]}\n"
                f"누락 파일: {missing_paths}"
            )

        masks = []
        for mask_path in mask_paths:
            masks.append(
                self._load_and_preprocess_style_mask_file(
                    mask_path=mask_path,
                    target_h=target_h,
                    target_w=target_w,
                    device=device,
                )
            )

        weight_maps = torch.stack(masks, dim=0).contiguous()  # (n,h,w,1)
        self.mask_cache[cache_key] = weight_maps
        return weight_maps
    
    def get_batch_sim(self, q, k, num_heads, **kwargs):
        
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        
        sim = torch.einsum("h i d, h j d -> h i j", q, k) * self.scale
        return sim 
    
    
    def get_batch_sim_with_mask(self, cc_sim, delta_q, delta_k, q, k, num_heads, sty_name, cnt_name, mask_path=None, attn_matrix_scale=1.0, ch=None, injection_config=None, target_t_list=None):
      
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)

        sty_name = sty_name
        cnt_name = cnt_name

        sim = torch.einsum("h i d, h j d -> h i j", q, k)
        #10/31 tau 수식 변경전
        #sim *= attn_matrix_scale
        sim *= self.scale

        #10/31 tau 수식 변경후
        #12/03 figure 구성 건으로 배수 2 -> 4로 수정
        #12/10 4로 실행
        # tau 임의 설정
        # mean = sim.mean(dim=-1, keepdim=True)
        
        # sim = (sim - mean) * attn_matrix_scale + mean
        #####
             
        head_num = sim.shape[0]
        pixel_size = sim.shape[1]
        
        # 기존에는 sqrt(N)로 정사각형만 가정했지만, 직사각형 latent도 지원하도록
        # run 스크립트에서 주입한 base_latent_hw를 기준으로 (h, w)를 복원합니다.
        # 기존 구현: h = w = int(sqrt(pixel_size)) 정사각형 가정.
        h, w = self._infer_spatial_hw_from_tokens(pixel_size)

        ##### z* 버전    
        # 기존 mask 적용 방식
        sim_reshaped = sim.reshape(head_num, h, w, pixel_size)
        ## cc_sim은 content-content query-key 내적에 scale까지 한 값.
        # cc_sim_reshaped = cc_sim.reshape(head_num, h, w, pixel_size)
        
        # delta_q = rearrange(delta_q, "(b h) n d -> h (b n) d", h=num_heads)
        # delta_k = rearrange(delta_k, "(b h) n d -> h (b n) d", h=num_heads)
        # max_sim = torch.einsum("h i d, h j d -> h i j", delta_q, delta_k)
        # max_sim_reshaped = max_sim.reshape(head_num, h, w, pixel_size)

        # min_cc_sim_reshaped, _ = torch.min(
        #     cc_sim_reshaped, dim=3, keepdim=True)
        # max_sim_reshaped, _ = torch.max(max_sim_reshaped, dim=3, keepdim=True)
        # start = 0.5
        # end = -0.5
        # # 정사각형 전제(w==h)였던 기존 코드의 안전한 일반화.
        # # 직사각형에서도 모든 row에 동일하게 마스크 보정이 적용되도록 h를 사용.
        # # 기존 구현: length = w 정사각형이었으므로 h나 w나 상관없었음. 
        # # 이제는 w!=h일 수 있으므로 h를 사용해서 모든 row에 동일하게 마스크 보정이 적용되도록 함.
        # length = h
        
        
        # mask = torch.tensor(np.load(mask_path), dtype=torch.float32).cuda()
        # mask[mask < 0.5] = -1.0
        # mask[mask > 0.5] = 1.0
        # mask = mask * ch #mask 영역별로 다르게 주기
        # mask = mask.unsqueeze(0).unsqueeze(0)
        # ### zero_star에서 사용한 방식
        # mask = F.interpolate(mask, size=(
        #     h, w), mode='bilinear', align_corners=False)
        # mask = mask.reshape(1, h, w, 1).to(sim.device)  # (1, h, w, 1)
        # 마스크 캐싱 - 경로를 키로 사용

        # base_latent_hw를 키에 포함해 해상도 변경 시 잘못된 캐시 재사용을 방지합니다.
        mask_key = f"{mask_path}_{h}_{w}_{ch}_{getattr(self, 'base_latent_hw', None)}"
        
        if mask_key not in self.mask_cache:
            # 처음 로드할 때만 현재 계층 해상도에 맞춰 전처리/캐싱합니다.
            mask = self._load_and_preprocess_mask(
                mask_path=mask_path,
                target_h=h,
                target_w=w,
                ch=ch,
                device=sim.device,
            )
            
            # 캐시에 저장
            self.mask_cache[mask_key] = mask
            del mask
        
        # 캐시된 마스크 사용
        mask = self.mask_cache[mask_key]
        # gradual_vanished_array = mask.reshape(1, h, w, 1).to(sim.device)
        # delta = min_cc_sim_reshaped - max_sim_reshaped
        # gradual_vanished_mask = (delta)[:, :, :, :] * gradual_vanished_array
        #print(f"gradual_vanished_mask shape: {gradual_vanished_mask.shape}")
        # print(f"sim_reshaped.shape:{sim_reshaped.shape}")
        # sim_reshaped[:, :length, :, :] += gradual_vanished_mask
        
        sim = sim_reshaped.reshape(head_num, pixel_size, pixel_size)
        
        ## 26/01/01: pi_star 버전
        self._last_mask = mask  
        return sim
    
    ## 26/01/01: pi_star 버전
    def apply_pi_mass_3way(self, sim1, sim2, cc_sim, mask, pi_star=0.9, eps=1e-6):
        # sim*: (H, N, N), mask: (1, h, w, 1) in [-1,1]
        H, Nq, Nk = sim1.shape
        # 직사각형 latent 지원을 위해 mask의 실제 spatial 크기를 사용합니다.
        side_h = int(mask.shape[1])
        side_w = int(mask.shape[2])
        assert side_h * side_w == Nq and Nk == Nq, (
            f"mask spatial과 attention token 수가 맞지 않습니다: "
            f"mask=({side_h},{side_w}), Nq={Nq}, Nk={Nk}"
        )

        m = mask.to(device=sim1.device, dtype=sim1.dtype).clamp(-1.0, 1.0)  # (1,h,w,1)
        m = -m
        # style1 활성도 / style2 활성도 (경계에서 섞이게)
        w1 = (m + 1.0) * 0.5            # (1,h,w,1)
        w2 = 1.0 - w1

        pi_c = torch.tensor(1.0 - pi_star, device=sim1.device, dtype=sim1.dtype).clamp(eps, 1-eps)
        pi1 = (pi_star * w1).clamp_min(eps)
        pi2 = (pi_star * w2).clamp_min(eps)

        # reshape to (H, h, w, N)
        s1 = sim1.reshape(H, side_h, side_w, Nq)
        s2 = sim2.reshape(H, side_h, side_w, Nq)
        sc = cc_sim.reshape(H, side_h, side_w, Nq)

        logZ1 = torch.logsumexp(s1, dim=3, keepdim=True)
        logZ2 = torch.logsumexp(s2, dim=3, keepdim=True)
        logZc = torch.logsumexp(sc, dim=3, keepdim=True)

        # b1,b2: (H,side,side,1) (pi1/pi2는 (1,side,side,1)이라 broadcast됨)
        b1 = logZc + (torch.log(pi1) - torch.log(pi_c)) - logZ1
        b2 = logZc + (torch.log(pi2) - torch.log(pi_c)) - logZ2

        s1 = s1 + b1
        s2 = s2 + b2

        return s1.reshape(H, Nq, Nq), s2.reshape(H, Nq, Nq)

    def apply_pi_mass_nway(self, style_sims, cc_sim, style_weight_maps, pi_style_total=0.9, eps=1e-6):
        """
        N개의 style branch와 1개의 content branch에 대해 확률질량(pi)을 재분배합니다.

        Args:
            style_sims: list[(heads, Nq, Nk)] 길이 = num_styles
            cc_sim: (heads, Nq, Nk)
            style_weight_maps: (num_styles, h, w, 1) 픽셀별 style 분배 가중치
            pi_style_total: 전체 style 질량 비율 (content는 1 - pi_style_total)
        """
        if len(style_sims) == 0:
            return []

        H, Nq, Nk = cc_sim.shape
        num_styles = len(style_sims)
        h = int(style_weight_maps.shape[1])
        w = int(style_weight_maps.shape[2])
        assert h * w == Nq and Nk == Nq, (
            f"style_weight_maps spatial과 attention token 수가 맞지 않습니다: "
            f"weights=({num_styles},{h},{w}), Nq={Nq}, Nk={Nk}"
        )

        # style_weight_maps는 멀티 마스크 파일에서 읽은 raw weight(0~1)일 수 있습니다.
        # 따라서 여기서 style 축 정규화와 coverage(마스크가 실제 존재하는 정도)를 함께 계산합니다.
        weights_raw = style_weight_maps.to(device=cc_sim.device, dtype=cc_sim.dtype).clamp_min(0.0)
        weight_sum = weights_raw.sum(dim=0, keepdim=True)  # (1,h,w,1)
        coverage = weight_sum.clamp(0.0, 1.0)
        weights = weights_raw / weight_sum.clamp_min(eps)

        pi_style_total_t = torch.tensor(pi_style_total, device=cc_sim.device, dtype=cc_sim.dtype).clamp(eps, 1 - eps)
        # 즉, style 총 질량(pi_style_total)을 coverage로 스케일해 unmasked 영역에서 content 질량이 1에 가깝게 됩니다.
        pi_style_total_eff = (pi_style_total_t * coverage).clamp(0.0, 1.0 - eps)
        pi_c = (1.0 - pi_style_total_eff).clamp(eps, 1 - eps)

        sc = cc_sim.reshape(H, h, w, Nq)
        logZc = torch.logsumexp(sc, dim=3, keepdim=True)

        adjusted = []
        for style_idx, sim in enumerate(style_sims):
            sj = sim.reshape(H, h, w, Nq)
            logZj = torch.logsumexp(sj, dim=3, keepdim=True)
            pi_j = (pi_style_total_eff * weights[style_idx:style_idx + 1]).clamp_min(eps)
            bj = logZc + (torch.log(pi_j) - torch.log(pi_c)) - logZj
            adjusted.append((sj + bj).reshape(H, Nq, Nq))

        return adjusted
    
    def forward(self,
                x,
                context=None,
                mask=None,
                q_injected=None,
                k_injected=None,
                v_injected=None,
                cnt_k_injected=None,
                sty_q_injected=None,
                cnt_v_injected=None,
                ##2개의 스타일 인젝션
                sty2_q_injected=None,
                sty2_k_injected=None,
                sty2_v_injected=None,
                # N-style 확장용: style2 이후의 스타일 k/v를 리스트로 전달합니다.
                # style1은 기존 k_injected/v_injected를 그대로 사용해 backward compatibility를 유지합니다.
                extra_style_k_injected_list=None,
                extra_style_v_injected_list=None,
                injection_config=None,):
        self.attn = None
        batch, seq_len, _ = x.shape
        h = self.heads
        b = x.shape[0]
     
        attn_matrix_scale = 1.0
        q_mix = 0.
        pi_style_total = 0.9
        is_cross = context is not None
        

        
        # import builtins
        if injection_config is not None:
            # cfg = builtins.feat_maps[builtins.global_step_idx]['config']
            # attn_matrix_scale = cfg.get("T", 1.0)#injection_config['T']
            #q_mix = cfg.get("gamma", 0.0)#injection_config['gamma']
            
            ## 원본
            attn_matrix_scale = injection_config['T']
            q_mix = injection_config['gamma']
            pi_style_total = injection_config.get('pi_style_total', 0.9)
            
        

        if q_injected is None:
            q = self.to_q(x)
            q = rearrange(q, 'b n (h d) -> (b h) n d', h=h)
        
        else:
            q_uncond = q_injected
            q_in = torch.cat([q_uncond]*b)
            q_ = self.to_q(x)
            q_ = rearrange(q_, 'b n (h d) -> (b h) n d', h=h)
            
            # q = q_in
            q = q_in * q_mix + q_ * (1. - q_mix) #content query가 q_in이다.
            
        context = default(context, x)

        if k_injected is None:
            k = self.to_k(context)
            k = rearrange(k, 'b m (h d) -> (b h) m d', h=h)
            
        else:
            k_uncond = k_injected
            k = torch.cat([k_uncond]*b ,dim=0)
           

            
        if v_injected is None:
            v = self.to_v(context)
            v = rearrange(v, 'b m (h d) -> (b h) m d', h=h)
         
        else:
            v_uncond = v_injected
            v = torch.cat([v_uncond]*b ,dim=0)
           

        self.q = q
        self.k = k
        self.v = v

        ##################### 마스크 적용 시작 ######################
        if not self.gen_pkl:
            base_name, _ = os.path.splitext(self.cnt_name)
            mask_path = base_name + "_mask.npy"
            # N-style 신규 규약: content_001_mask0.npy, content_001_mask1.npy, ...
            mask_prefix = base_name + "_mask"

            is_mask_exists = os.path.exists(mask_path)  # legacy 단일 마스크(_mask.npy) 존재 여부
            # N-style에서는 `_mask0.npy`, `_mask1.npy`, ... 존재 여부를 별도로 확인해야 합니다.
            num_styles_cfg_for_mask_gate = None
            has_explicit_num_styles_cfg_for_mask_gate = False
            if injection_config is not None and ("num_styles" in injection_config):
                has_explicit_num_styles_cfg_for_mask_gate = True
                num_styles_cfg_for_mask_gate = int(injection_config.get("num_styles", 0))

            is_nstyle_multimask_exists = False
            if has_explicit_num_styles_cfg_for_mask_gate and num_styles_cfg_for_mask_gate is not None and num_styles_cfg_for_mask_gate > 0:
                expected_multi_mask_paths = [
                    f"{mask_prefix}{style_idx}.npy" for style_idx in range(num_styles_cfg_for_mask_gate)
                ]
                is_nstyle_multimask_exists = all(os.path.exists(p) for p in expected_multi_mask_paths)
            
            use_mask = (
                self.cnt_name is not None
                and not is_cross
                # 2-style legacy는 `_mask.npy`, N-style은 `_mask{i}.npy` 묶음으로 마스크 사용 여부를 판정합니다.
                and (is_mask_exists or is_nstyle_multimask_exists)
            )
            


            if use_mask:
                # self.sty_name = os.path.basename(self.sty_name)
                if q_injected is not None and k_injected is not None:
                    # content branch (q,k,v) 구성은 N-style에서도 동일하게 유지합니다.
                    q_cnt = q_in
                    k_cnt = torch.cat([cnt_k_injected] * b, dim=0)
                    v_cnt = torch.cat([cnt_v_injected] * b, dim=0)

                    # style1은 기존 k/v 입력을 그대로 사용하고,
                    # style2 이후는 리스트(extra_style_*) 또는 legacy sty2_* 입력에서 수집합니다.
                    style_k_branches = [k]
                    style_v_branches = [v]

                    if extra_style_k_injected_list is not None and extra_style_v_injected_list is not None:
                        for k_extra, v_extra in zip(extra_style_k_injected_list, extra_style_v_injected_list):
                            if k_extra is None or v_extra is None:
                                continue
                            style_k_branches.append(torch.cat([k_extra] * b, dim=0))
                            style_v_branches.append(torch.cat([v_extra] * b, dim=0))
                    elif sty2_k_injected is not None and sty2_v_injected is not None:
                        # backward compatibility: 기존 2-style 전용 입력 경로도 자동 수용
                        style_k_branches.append(torch.cat([sty2_k_injected] * b, dim=0))
                        style_v_branches.append(torch.cat([sty2_v_injected] * b, dim=0))

                    num_styles_cfg = None
                    has_explicit_num_styles_cfg = False
                    if injection_config is not None:
                        # 새 N-style 경로(run_ori.py의 feat_merge_nsty_clean)는 config에 num_styles를 항상 기록합니다.
                        # 값이 1이어도 신규 경로를 타야 하므로 "키 존재 여부"를 함께 보관합니다.
                        has_explicit_num_styles_cfg = ("num_styles" in injection_config)
                        num_styles_cfg = int(injection_config.get("num_styles", 0))
                    num_styles = num_styles_cfg if (num_styles_cfg is not None and num_styles_cfg > 0) else len(style_k_branches)
                    if num_styles > len(style_k_branches):
                        raise ValueError(
                            f"주입된 style branch 수가 config.num_styles보다 적습니다: "
                            f"num_styles={num_styles}, available={len(style_k_branches)}"
                        )
                    # 신규 N-style 메타/merge 경로는 N=1,2,... 모두 여기로 처리합니다.
                    # legacy 2-style(feat_merge_2sty_clean)는 num_styles 키가 없으므로 아래 조건에서 제외됩니다.
                    use_nstyle_path = has_explicit_num_styles_cfg or (len(style_k_branches) > 2)

                    if use_nstyle_path:
                        cc_sim = self.get_batch_sim(
                            q=q_cnt,
                            k=k_cnt,
                            num_heads=h,
                        )

                        # 각 스타일 브랜치 logits를 동일한 Tau(T) 스케일링으로 계산
                        style_sims = []
                        for k_branch in style_k_branches[:num_styles]:
                            style_sims.append(
                                self._compute_sim_with_tau(
                                    q=q,
                                    k=k_branch,
                                    num_heads=h,
                                    attn_matrix_scale=attn_matrix_scale,
                                )
                            )

                        # 기본 규약은 `_mask0.npy`, `_mask1.npy`, ... 멀티 마스크 파일이며,
                        # 호환성 차원에서 `_mask.npy` label map도 있으면 fallback으로 수용합니다.
                        layer_h, layer_w = self._infer_spatial_hw_from_tokens(q.shape[1])
                        style_weight_maps = self._load_style_weight_maps_from_mask_files(
                            mask_prefix=mask_prefix,
                            target_h=layer_h,
                            target_w=layer_w,
                            num_styles=num_styles,
                            device=q.device,
                            # N-style 경로에서는 멀티 마스크 파일 규약이 의도이므로,
                            # `_mask.npy`로의 silent fallback을 막아 잘못된 전체 스타일 적용을 방지합니다.
                            allow_legacy_label_fallback=False,
                        )

                        style_sims = self.apply_pi_mass_nway(
                            style_sims=style_sims,
                            cc_sim=cc_sim,
                            style_weight_maps=style_weight_maps,
                            pi_style_total=pi_style_total,
                        )
                      
                        cat_sim = torch.cat(style_sims + [cc_sim], 2)
                        ### poly fit을 통한 tau 정하기
                        H, Nq, Nk_cat = cat_sim.shape
                        def log_pmax(logits, dim=-1):
                            # logits: (..., N)
                            max_logit, _ = logits.max(dim=dim, keepdim=True)
                            lse = torch.logsumexp(logits, dim=dim, keepdim=True)
                            return (max_logit - lse).squeeze(dim)
                        logp_cc  = log_pmax(cc_sim)    # (H, Nq)
                        logp_cat = log_pmax(cat_sim)   # (H, Nq)
    
                        # (H, Nq)
                        delta = logp_cc - logp_cat   # head-wise per query

                        # head별 하나의 delta로 만들기 (query 평균)
                        delta_head = delta.mean(dim=1)   # (H,)

                        # ----------------------------------------
                        # polynomial tau (2nd order)
                        # tau = a * Δ^2 + b * Δ + c
                        # ----------------------------------------

                        a1 = 0.08395199
                        b2 = 0.43704639
                        c3 = 1.00998177

                        tau = a1 * delta_head**2 + b2 * delta_head + c3   # (H,)

                        # 안정성 클램프 (optional but strongly recommended)
                        tau = torch.clamp(tau, min=1.0, max=5.0) # tau가 1보다 작아지는 것을 방지
                        # ----------------------------------------
                        # head-wise temperature scaling
                        # ----------------------------------------

                        # (H, 1, 1)로 reshape해서 broadcasting
                        tau = tau.view(H, 1, 1)

                        cat_sim = tau * (cat_sim - cat_sim.mean(dim=-1, keepdim=True)) + cat_sim.mean(dim=-1, keepdim=True) # group-wise mean 유지
                        ### poly fit을 통한 tau 정하기      
                        cat_v = torch.cat(style_v_branches[:num_styles] + [v_cnt], 1)

                        cat_sim = cat_sim.softmax(-1)
                        cat_out = einsum('b i j, b j d -> b i d', cat_sim, cat_v)
                        out = rearrange(cat_out, 'h (b n) d -> b n (h d)', h=h, b=b)
                    else:
                        # ===== legacy 2-style 경로 (기존 결과 재현용) =====
                        k_sty_2 = torch.cat([sty2_k_injected]*b, dim=0)
                        v_sty_2 = torch.cat([sty2_v_injected]*b, dim=0)
                        
                        cc_sim = self.get_batch_sim(
                            q=q_cnt,
                            k=k_cnt,
                            num_heads=h,
                        )

                        sim_1 = self.get_batch_sim_with_mask(
                            cc_sim=cc_sim,
                            q=q, ##  self.q 면 q_cs를 의미, q_cnt면 감마가 적용되지않은 cnt그대로
                            delta_q=q, # Qcs
                            delta_k=k, # self.k = k_sty와 같음. inject당시 sty에서 key와 value를 가져오기 때문.
                            k=k,
                            num_heads=h,
                            sty_name=self.sty_name,
                            cnt_name=self.cnt_name,
                            mask_path=mask_path,
                            attn_matrix_scale=attn_matrix_scale,
                            ch = -1.0,
                            injection_config=injection_config,
                            target_t_list=self.target_t_list,
                        )
                        ## 26/01/01: pi_star 버전
                        mask_char = self._last_mask
                        
                        # Back
                        sim_2 = self.get_batch_sim_with_mask(
                            cc_sim=cc_sim,
                            q=q, ##  self.q 면 q_cs를 의미, q_cnt면 감마가 적용되지않은 cnt그대로
                            delta_q=q, # Qcs,
                            delta_k=k_sty_2, # stlye_2 key
                            k=k_sty_2,
                            num_heads=h,
                            sty_name=self.sty_name,
                            cnt_name=self.cnt_name,
                            mask_path=mask_path,
                            attn_matrix_scale=attn_matrix_scale,
                            ch = 1.0,
                        )# Qcs Ks2 + a2
                    
                        ### 26/01/01: pi_star 버전
                        pi_star = 0.9
                        # print(f"2-style은 기존것 그대로 구현중.")
                        sim_1, sim_2 = self.apply_pi_mass_3way(sim_1, sim_2, cc_sim, mask_char, pi_star)
                        
                        cat_sim = torch.cat((sim_1, sim_2, cc_sim), 2)
                        ### poly fit을 통한 tau 정하기
                        H, Nq, Nk_cat = cat_sim.shape
                        def log_pmax(logits, dim=-1):
                            # logits: (..., N)
                            max_logit, _ = logits.max(dim=dim, keepdim=True)
                            lse = torch.logsumexp(logits, dim=dim, keepdim=True)
                            return (max_logit - lse).squeeze(dim)
                        logp_cc  = log_pmax(cc_sim)    # (H, Nq)
                        logp_cat = log_pmax(cat_sim)   # (H, Nq)
    
                        # (H, Nq)
                        delta = logp_cc - logp_cat   # head-wise per query

                        # head별 하나의 delta로 만들기 (query 평균)
                        delta_head = delta.mean(dim=1)   # (H,)

                        # ----------------------------------------
                        # polynomial tau (2nd order)
                        # tau = a * Δ^2 + b * Δ + c
                        # ----------------------------------------

                        a1 = 0.08395199
                        b2 = 0.43704639
                        c3 = 1.00998177

                        tau = a1 * delta_head**2 + b2 * delta_head + c3   # (H,)

                        # 안정성 클램프 (optional but strongly recommended)
                        tau = torch.clamp(tau, min=1.0, max=5.0) # tau가 1보다 작아지는 것을 방지
                        # ----------------------------------------
                        # head-wise temperature scaling
                        # ----------------------------------------

                        # (H, 1, 1)로 reshape해서 broadcasting
                        tau = tau.view(H, 1, 1)

                        cat_sim = tau * (cat_sim - cat_sim.mean(dim=-1, keepdim=True)) + cat_sim.mean(dim=-1, keepdim=True) # group-wise mean 유지
                        ### poly fit을 통한 tau 정하기
                        # batch 차원 맞추기
                        cat_v = torch.cat((v, v_sty_2, v_cnt), 1)# stlye_1 value, stlye_2 value, content value

                        cat_sim = cat_sim.softmax(-1)

                        cat_out = einsum('b i j, b j d -> b i d', cat_sim, cat_v)
                        
                        # cat시에는
                        out = rearrange(cat_out, 'h (b n) d -> b n (h d)', h=h, b=b)

                # style injection이 일어나지 않는 경우 -- 원본과 동일하게 진행
                else:
                    sim = einsum('b i d, b j d -> b i j', q, k)
                    sim *= attn_matrix_scale
                    sim *= self.scale
                    attn = sim.softmax(dim=-1)

                    out = einsum('b i j, b j d -> b i d', attn, v)
                
                    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
                    
            # {마스크용 npy파일이 없으면 마스킹 적용 x} or {self.attn2 즉, cross-attention}  -- 원본과 동일하게 진행  
            else:
                sim = einsum('b i d, b j d -> b i j', q, k)
                if q_injected is not None or k_injected is not None:
                # print(attn_matrix_scale, 'attn_matrix_scale')
                    sim *= attn_matrix_scale    
                sim *= self.scale
                attn = sim.softmax(dim=-1)

                out = einsum('b i j, b j d -> b i d', attn, v)
                out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
                
                # print(f"마스크 적용 안함, sim.shape: {sim.shape} \n")
        ################# 마스크 적용 끝 ###################
        
        else:
        ################# 원본 ###################
            sim = einsum('b i d, b j d -> b i j', q, k)
            
            
            sim *= self.scale

            if exists(mask):
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)

            out = einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        ################# 원본 끝 ###################
        
        
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        
    def forward(self,
                x,
                context=None,
                self_attn_q_injected=None,
                self_attn_k_injected=None,
                self_attn_v_injected=None,
                ## 마스크용
                self_attn_cnt_k_injected=None,
                self_attn_sty_q_injected=None,
                self_attn_cnt_v_injected=None,
                ## 2개의 스타일 인젝션
                self_attn_sty2_q_injected=None,
                self_attn_sty2_k_injected=None,
                self_attn_sty2_v_injected=None,
                # N-style 확장용 추가 스타일(k/v) 리스트 (style2 이후)
                self_attn_extra_style_k_injected_list=None,
                self_attn_extra_style_v_injected_list=None,
                
                injection_config=None,
                ):
        return checkpoint(self._forward, (x,
                                          context,
                                          self_attn_q_injected,
                                          self_attn_k_injected,
                                          self_attn_v_injected,
                                          ##마스크용
                                          self_attn_cnt_k_injected,
                                          self_attn_sty_q_injected,
                                          self_attn_cnt_v_injected,
                                          ## 2개의 스타일 인젝션
                                          self_attn_sty2_q_injected,
                                          self_attn_sty2_k_injected,
                                          self_attn_sty2_v_injected,
                                          self_attn_extra_style_k_injected_list,
                                          self_attn_extra_style_v_injected_list,
                                          
                                          injection_config,), self.parameters(), self.checkpoint)

    def _forward(self,
                 x,
                 context=None,
                 self_attn_q_injected=None,
                 self_attn_k_injected=None,
                 self_attn_v_injected=None,
                 ##마스크용
                 self_attn_cnt_k_injected=None,
                 self_attn_sty_q_injected=None,
                 self_attn_cnt_v_injected=None,
                 ##2개의 스타일 인젝션
                 self_attn_sty2_q_injected=None,
                 self_attn_sty2_k_injected=None,
                 self_attn_sty2_v_injected=None,
                 self_attn_extra_style_k_injected_list=None,
                 self_attn_extra_style_v_injected_list=None,
                 
                 injection_config=None):
        x_ = self.attn1(self.norm1(x),
                       q_injected=self_attn_q_injected,
                       k_injected=self_attn_k_injected,
                       v_injected=self_attn_v_injected,
                       #마스크
                       cnt_k_injected=self_attn_cnt_k_injected,
                       sty_q_injected=self_attn_sty_q_injected,
                       cnt_v_injected=self_attn_cnt_v_injected,
                       ##2개의 스타일 인젝션
                       sty2_q_injected=self_attn_sty2_q_injected,
                       sty2_k_injected=self_attn_sty2_k_injected,
                       sty2_v_injected=self_attn_sty2_v_injected,
                       extra_style_k_injected_list=self_attn_extra_style_k_injected_list,
                       extra_style_v_injected_list=self_attn_extra_style_v_injected_list,
                       
                       injection_config=injection_config,)
        x = x_ + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self,
                x,
                context=None,
                self_attn_q_injected=None,
                self_attn_k_injected=None,
                self_attn_v_injected=None,
                ## 마스크용
                self_attn_cnt_k_injected=None, 
                self_attn_sty_q_injected=None,
                self_attn_cnt_v_injected=None,
                ##2개의 스타일 인젝션
                self_attn_sty2_q_injected=None,
                self_attn_sty2_k_injected=None,
                self_attn_sty2_v_injected=None,
                self_attn_extra_style_k_injected_list=None,
                self_attn_extra_style_v_injected_list=None,
                
                injection_config=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        for block in self.transformer_blocks:
            x = block(x,
                      context=context,
                      self_attn_q_injected=self_attn_q_injected,
                      self_attn_k_injected=self_attn_k_injected,
                      self_attn_v_injected=self_attn_v_injected,
                      ##마스크용
                      self_attn_cnt_k_injected=self_attn_cnt_k_injected,
                      self_attn_sty_q_injected=self_attn_sty_q_injected,
                      self_attn_cnt_v_injected=self_attn_cnt_v_injected,
                      ##2개의 스타일 인젝션
                      self_attn_sty2_q_injected=self_attn_sty2_q_injected,
                      self_attn_sty2_k_injected=self_attn_sty2_k_injected,
                      self_attn_sty2_v_injected=self_attn_sty2_v_injected,
                      self_attn_extra_style_k_injected_list=self_attn_extra_style_k_injected_list,
                      self_attn_extra_style_v_injected_list=self_attn_extra_style_v_injected_list,
                
                      injection_config=injection_config)

            
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in
