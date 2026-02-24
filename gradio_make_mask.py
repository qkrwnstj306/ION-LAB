# app_fixed_canvas.py  (Gradio 4.44.1)
import gradio as gr
import numpy as np
from PIL import Image
import tempfile

CANVAS = (512, 512)  # 내부 작업 해상도 고정

def fit_pad_to_canvas(img: Image.Image, canvas=CANVAS, fill=(0,0,0)):
    """이미지를 캔버스 크기에 '전체가 보이도록' 리사이즈 후 패딩."""
    W, H = canvas
    w, h = img.size
    scale = min(W / w, H / h)
    nw, nh = int(round(w*scale)), int(round(h*scale))
    img_resized = img.resize((nw, nh), Image.BICUBIC)
    bg = Image.new("RGB", (W, H), fill)
    x = (W - nw)//2
    y = (H - nh)//2
    bg.paste(img_resized, (x, y))
    return bg

def prep_editor_background(uploaded):
    """업로드 이미지를 캔버스 규격으로 정규화 → ImageEditor value(dict) 반환"""
    if uploaded is None:
        # 세 키 모두 제공(비어 있어도)
        return {"background": None, "layers": [], "composite": None}

    if isinstance(uploaded, np.ndarray):
        uploaded = Image.fromarray(uploaded[..., :3])
    elif not isinstance(uploaded, Image.Image):
        raise gr.Error("이미지 타입을 인식하지 못했습니다.")

    bg = fit_pad_to_canvas(uploaded)

    # 🔧 Gradio 4.44.x: postprocess에서 'layers'를 바로 참조하므로 반드시 포함
    # composite도 같이 채워주는 게 안전
    return {"background": bg, "layers": [], "composite": bg}

def _alpha_mask_from_layer(layer_img: Image.Image):
    arr = np.array(layer_img)
    if arr.ndim == 2:
        m = (arr > 0).astype(np.uint8) * 255
    elif arr.ndim == 3 and arr.shape[2] == 4:
        m = (arr[:, :, 3] > 0).astype(np.uint8) * 255
    else:
        m = (arr.mean(axis=2) > 0).astype(np.uint8) * 255
    return m

def build_mask_and_apply(editor_value, invert=False):
    if not isinstance(editor_value, dict):
        raise gr.Error("좌측 에디터에 이미지를 넣고 브러시로 칠해 주세요.")

    bg = editor_value.get("background")
    layers = editor_value.get("layers") or []
    comp = editor_value.get("composite")

    # PIL로 통일
    def to_pil(x):
        if x is None: return None
        if isinstance(x, Image.Image): return x
        if isinstance(x, np.ndarray):
            x = np.clip(x, 0, 255).astype(np.uint8)
            if x.ndim == 2: return Image.fromarray(x, "L")
            if x.ndim == 3 and x.shape[2] == 4: return Image.fromarray(x, "RGBA")
            return Image.fromarray(x[..., :3], "RGB")
        return None

    bg = to_pil(bg)
    comp = to_pil(comp)
    if bg is None and comp is None:
        raise gr.Error("배경이 없어요. 먼저 이미지를 업로드하세요.")

    # ✅ 마지막 레이어의 알파를 마스크로
    m_bin = None
    for layer in reversed(layers):
        L = to_pil(layer)
        if L is None: 
            continue
        m = _alpha_mask_from_layer(L)
        if (m > 0).any():
            m_bin = m
            break

    # 레이어가 비었으면 실패 처리
    if m_bin is None or (m_bin == 0).all():
        raise gr.Error("마스크가 비어있습니다. 브러시로 새 레이어에 칠해 주세요.")

    if invert:
        m_bin = 255 - m_bin

    mask_img = Image.fromarray(m_bin, "L")
    base = (bg or comp).convert("RGB")       # 항상 768×768
    rgb = np.array(base)
    masked = (rgb * (m_bin[..., None] / 255.0)).astype(np.uint8)
    masked_img = Image.fromarray(masked)

    f_mask = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    mask_img.save(f_mask.name)
    f_masked = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    masked_img.save(f_masked.name)

    return mask_img, masked_img, f_mask.name, f_masked.name

with gr.Blocks() as demo:
    gr.Markdown("## 고정 해상도 마스킹 도구 (768×768)\n1) 이미지 업로드 → 2) 자동 정규화 → 3) 새 레이어에 칠하고 ‘마스크 생성’")

    with gr.Row():
        with gr.Column(scale=1, min_width=820):
            uploader = gr.Image(label="원본 업로드", type="pil")
            editor = gr.ImageEditor(
                label="편집(브러시/레이어) — 내부 해상도 고정 768×768",
                show_download_button=False,
                height=820,  # UI가 줄어들며 잘리는 것을 방지(스크롤)
            )
        with gr.Column(scale=1, min_width=420):
            mask_preview = gr.Image(label="흑백 마스크 미리보기")
            masked_preview = gr.Image(label="마스크 적용 결과")

    invert = gr.Checkbox(label="마스크 반전", value=False)
    btn = gr.Button("마스크 생성")

    with gr.Row():
        mask_file = gr.File(label="다운로드: mask.png")
        masked_file = gr.File(label="다운로드: masked.png")

    # 업로드 → 768x768로 정규화하여 에디터 background에 세팅
    uploader.change(prep_editor_background, inputs=uploader, outputs=editor)

    btn.click(build_mask_and_apply, [editor, invert],
              [mask_preview, masked_preview, mask_file, masked_file])

if __name__ == "__main__":
    demo.launch(show_error=True)
