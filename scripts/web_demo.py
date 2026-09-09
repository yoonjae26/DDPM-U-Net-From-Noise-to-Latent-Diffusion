"""Phase 12: Interactive Web Application (Streamlit).

Run with:
    streamlit run scripts/web_demo.py

Three tabs, each a self-contained lesson built on the trained checkpoints:
  1. VAE Explorer     -- Image -> Encoder -> mu/sigma -> reparameterize -> Decoder
  2. Forward Diffusion -- Latent -> add noise at timestep t (linear vs cosine)
  3. Reverse Diffusion -- Noise -> denoise -> Image (DDPM, with an experimental DDIM mode)
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str((ROOT / "src").resolve()))

from latent_diffusion_playground.diffusion import (  # noqa: E402
    ddim_sample_step,
    make_schedule,
    p_sample_loop,
    q_sample,
)
from latent_diffusion_playground.phases.latent_space_viewer import load_viewer_artifacts  # noqa: E402
from latent_diffusion_playground.phases.reverse_diffusion_viewer import (  # noqa: E402
    decoded_to_image,
    latent_to_heatmap,
    load_reverse_viewer_artifacts,
)

st.set_page_config(page_title="Latent Diffusion Playground", layout="wide")


# =========================================================
# Cached model loading
# =========================================================
@st.cache_resource
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_vae(vae_ckpt: str, norm_stats: str, norm_mode: str):
    device = get_device()
    return load_viewer_artifacts(Path(vae_ckpt), Path(norm_stats), norm_mode, device)


@st.cache_resource
def load_diffusion(unet_ckpt: str, vae_ckpt: str, norm_stats: str, norm_mode: str, scaling_json: str):
    device = get_device()
    return load_reverse_viewer_artifacts(
        unet_checkpoint_path=Path(unet_ckpt),
        vae_checkpoint_path=Path(vae_ckpt),
        normalization_stats_path=Path(norm_stats),
        norm_mode=norm_mode,
        device=device,
        latent_scaling_path=Path(scaling_json),
    )


@st.cache_data
def sample_dataset_images(n: int = 12) -> list[str]:
    val_dir = ROOT / "artifacts" / "phase1" / "preprocessed_images" / "val"
    files = sorted(val_dir.glob("*.jpg"))
    if not files:
        return []
    return [str(p) for p in random.sample(files, min(n, len(files)))]


# =========================================================
# Preprocessing (mirrors data/preprocess.py + data/torch_dataset.py exactly)
# =========================================================
def preprocess_image(image: Image.Image, image_size: int, mean: tuple, std: tuple) -> torch.Tensor:
    image = image.convert("RGB")
    w, h = image.size
    crop = min(w, h)
    left, top = (w - crop) // 2, (h - crop) // 2
    image = image.crop((left, top, left + crop, top + crop)).resize(
        (image_size, image_size), Image.Resampling.LANCZOS
    )
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean_t) / std_t


def to_pil(tensor: torch.Tensor, mean: tuple, std: tuple) -> Image.Image:
    return decoded_to_image(tensor.unsqueeze(0) if tensor.ndim == 3 else tensor, mean, std)


# =========================================================
# Sidebar: checkpoint configuration
# =========================================================
st.sidebar.title("⚙️ Checkpoints")
vae_ckpt = st.sidebar.text_input("VAE checkpoint", str(ROOT / "artifacts" / "phase2_lpips" / "vae_last.pt"))
unet_ckpt = st.sidebar.text_input("UNet checkpoint", str(ROOT / "artifacts" / "phase6_v2" / "unet_last.pt"))
scaling_json = st.sidebar.text_input(
    "Latent scaling JSON", str(ROOT / "artifacts" / "phase6_v2" / "latent_scaling.json")
)
norm_stats = str(ROOT / "artifacts" / "phase1" / "normalization_stats.yaml")
norm_mode = "diffusion"

with st.sidebar.expander("ℹ️ 이 프로젝트에 대해", expanded=False):
    st.markdown(
        """
**파이프라인:** 이미지 → VAE 인코더 → 잠재 벡터 → (+노이즈) → U-Net → (-노이즈) → 잠재 벡터 → VAE 디코더 → 이미지

이 모델은 `phase2_train_vae.py`(VAE)와 `phase6_train_latent_diffusion.py`(U-Net)로 학습되었습니다.

⚠️ **중요한 교훈:** 이 프로젝트의 초기 DDPM 버전은 학습 손실이 매우 낮았음에도
항상 완전한 노이즈만 생성했습니다 — 원인은 샘플러에 *x0-clipping*
(dynamic thresholding) 단계가 없었기 때문입니다. t=T 부근에서
cosine 스케줄의 `beta`가 `0.999`에 도달하면서 `1/sqrt(alpha_t)` 증폭 계수가
매우 커지고, 이로 인해 노이즈 제거 몇 단계만에 잠재 벡터가 폭발해버립니다.
자세한 내용과 수치는 `docs/PHASE_PLAN.md`와
`scripts/phase11_scheduler_stability_lab.py`를 참고하세요.
        """
    )

try:
    vae_artifacts = load_vae(vae_ckpt, norm_stats, norm_mode)
    vae_ready = True
except Exception as exc:  # noqa: BLE001
    vae_ready = False
    st.sidebar.error(f"VAE를 불러오지 못했습니다: {exc}")

diffusion_ready = False
if vae_ready:
    try:
        diff_artifacts = load_diffusion(unet_ckpt, vae_ckpt, norm_stats, norm_mode, scaling_json)
        diffusion_ready = True
    except Exception as exc:  # noqa: BLE001
        st.sidebar.warning(f"UNet을 불러오지 못했습니다 (탭 2/3 비활성화): {exc}")

st.title("🌀 Latent Diffusion Playground")
st.caption("VAE, DDPM, U-Net 시각화 데모 (CelebA)")

tab1, tab2, tab3 = st.tabs(["1️⃣ VAE 탐색기", "2️⃣ 순방향 확산", "3️⃣ 역방향 확산 (생성)"])


def _pick_image_widget(key_prefix: str) -> Image.Image | None:
    col_a, col_b = st.columns([2, 1])
    with col_a:
        uploaded = st.file_uploader("얼굴 이미지 업로드", type=["jpg", "jpeg", "png"], key=f"{key_prefix}_upload")
    with col_b:
        if st.button("🎲 CelebA에서 무작위 샘플 가져오기", key=f"{key_prefix}_random"):
            samples = sample_dataset_images(1)
            if samples:
                st.session_state[f"{key_prefix}_path"] = samples[0]
    if uploaded is not None:
        return Image.open(uploaded)
    path = st.session_state.get(f"{key_prefix}_path")
    if path:
        return Image.open(path)
    return None


# =========================================================
# Tab 1 -- VAE Explorer
# =========================================================
with tab1:
    st.header("VAE: 이미지 → 인코더 → μ, σ → 재매개변수화 → 디코더")
    if not vae_ready:
        st.stop()

    image = _pick_image_widget("vae")
    use_sampling = st.checkbox(
        "μ만 사용하는 대신 재매개변수화 트릭(z = μ + σ·ε) 사용",
        value=False,
        help="체크 해제 시 '평균' 재구성(결정론적) 결과를 보여줍니다. 체크하면 무작위 노이즈 ε의 영향을 확인할 수 있습니다.",
    )

    if image is not None:
        x = preprocess_image(image, vae_artifacts.image_size, vae_artifacts.normalization.mean, vae_artifacts.normalization.std)
        x = x.unsqueeze(0).to(vae_artifacts.device)

        with torch.no_grad():
            mu, logvar = vae_artifacts.model.encode(x)
            z = vae_artifacts.model.reparameterize(mu, logvar) if use_sampling else mu
            recon = vae_artifacts.model.decode(z)

        col1, col2 = st.columns(2)
        with col1:
            st.image(to_pil(x[0], vae_artifacts.normalization.mean, vae_artifacts.normalization.std), caption="원본 이미지 (128x128로 크롭/리사이즈됨)")
        with col2:
            st.image(to_pil(recon[0], vae_artifacts.normalization.mean, vae_artifacts.normalization.std), caption="VAE 재구성 결과")

        mse = torch.nn.functional.mse_loss(recon, x).item()
        st.metric("Reconstruction MSE", f"{mse:.5f}")
        st.caption(
            f"Latent shape: {list(mu.shape)} -- μ mean={mu.mean().item():.3f}, "
            f"σ mean={torch.exp(0.5 * logvar).mean().item():.3f}"
        )
    else:
        st.info("이미지를 업로드하거나 '무작위 샘플 가져오기'를 눌러 시작하세요.")


# =========================================================
# Tab 2 -- Forward Diffusion
# =========================================================
with tab2:
    st.header("순방향 확산: 잠재 벡터 → t에 따라 가우시안 노이즈를 점진적으로 추가")
    if not vae_ready:
        st.stop()

    image2 = _pick_image_widget("fwd")
    scheduler_name = st.radio("Scheduler", ["cosine", "linear"], horizontal=True, key="fwd_scheduler")
    timesteps = st.slider("전체 타임스텝 수 T", 100, 1000, 1000, step=100, key="fwd_T")
    t_value = st.slider("타임스텝 t (0 = 원본 이미지, T-1 = 완전한 노이즈)", 0, timesteps - 1, 0, key="fwd_t")

    if image2 is not None:
        x2 = preprocess_image(image2, vae_artifacts.image_size, vae_artifacts.normalization.mean, vae_artifacts.normalization.std)
        x2 = x2.unsqueeze(0).to(vae_artifacts.device)
        schedule = make_schedule(scheduler_name, timesteps, device=vae_artifacts.device)

        with torch.no_grad():
            latent = vae_artifacts.model.encode_feature_map(x2)
            t_tensor = torch.tensor([t_value], device=vae_artifacts.device, dtype=torch.long)
            noisy_latent = q_sample(latent, t_tensor, schedule)
            decoded_noisy = vae_artifacts.model.decode_feature_map(noisy_latent)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(to_pil(x2[0], vae_artifacts.normalization.mean, vae_artifacts.normalization.std), caption="원본 이미지")
        with col2:
            st.image(latent_to_heatmap(noisy_latent), caption=f"t={t_value}에서의 잠재 벡터 (히트맵)")
        with col3:
            st.image(
                to_pil(decoded_noisy[0], vae_artifacts.normalization.mean, vae_artifacts.normalization.std),
                caption=f"노이즈가 섞인 잠재 벡터를 디코딩 (t={t_value})",
            )

        alpha_hat_t = schedule.alpha_hat[t_value].item()
        st.caption(
            f"alpha_hat[t={t_value}] = {alpha_hat_t:.4f}  →  남아있는 신호 비율 ≈ {alpha_hat_t*100:.1f}%, "
            f"노이즈 비율 ≈ {(1-alpha_hat_t)*100:.1f}%"
        )
    else:
        st.info("이미지를 업로드하거나 '무작위 샘플 가져오기'를 눌러 시작하세요.")


# =========================================================
# Tab 3 -- Reverse Diffusion (Generation)
# =========================================================
with tab3:
    st.header("역방향 확산: 완전한 노이즈 → U-Net이 점진적으로 노이즈 제거 → 이미지")
    if not diffusion_ready:
        st.warning("UNet 체크포인트를 불러오지 못했습니다 -- 사이드바의 경로를 확인하세요.")
        st.stop()

    num_samples = st.slider("생성할 이미지 개수", 1, 8, 4)
    sampler = st.radio(
        "Sampler",
        ["ddpm (권장, 1000단계, 안정적)", "ddim (실험적, 더 빠르지만 불안정할 수 있음)"],
        horizontal=False,
    )
    seed = st.number_input("Seed", value=42, step=1)

    if "ddim" in sampler:
        ddim_steps = st.slider("DDIM 스텝 수", 10, 500, 50)
        st.caption(
            "⚠️ DDIM은 스텝 수가 적으면 cosine 스케줄의 끝부분(t가 T에 가까울 때)에서 "
            "여전히 노이즈만 나올 수 있습니다 -- 이는 DDPM에서 수정한 버그와는 별개의 불안정성입니다. "
            "자세한 내용은 `scripts/phase11_scheduler_stability_lab.py`를 참고하세요."
        )

    if st.button("🎨 생성하기", type="primary"):
        device = diff_artifacts.device
        schedule = make_schedule("cosine", 1000, device=device)
        shape = (num_samples, diff_artifacts.unet_in_channels, 16, 16)

        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))

        with st.spinner("노이즈 제거 중..."):
            with torch.no_grad():
                if "ddpm" in sampler:
                    sampled = p_sample_loop(diff_artifacts.unet, shape, schedule, device=device)
                else:
                    step_size = max(1000 // ddim_steps, 1)
                    ddim_seq = list(range(999, 0, -step_size))
                    ddim_seq_prev = ddim_seq[1:] + [0]
                    img = torch.randn(shape, device=device)
                    for t_idx, t_prev_idx in zip(ddim_seq, ddim_seq_prev):
                        t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
                        t_prev = torch.full((shape[0],), t_prev_idx, device=device, dtype=torch.long)
                        epsilon = diff_artifacts.unet(img, t)
                        img = ddim_sample_step(img, t, t_prev, epsilon, schedule, eta=0.0)
                    sampled = img

                decoded = vae_artifacts.model.decode_feature_map(sampled / diff_artifacts.latent_scaling_factor)

        cols = st.columns(num_samples)
        for i in range(num_samples):
            with cols[i]:
                st.image(
                    to_pil(decoded[i], vae_artifacts.normalization.mean, vae_artifacts.normalization.std),
                    caption=f"샘플 {i}",
                )
