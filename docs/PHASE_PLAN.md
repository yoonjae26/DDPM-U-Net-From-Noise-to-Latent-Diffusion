# Latent Diffusion Playground Phase Plan

## Current status

- Phase 1 Dataset Pipeline: implemented
- Phase 2 VAE Development: implemented (baseline training pipeline)
- Phase 3+ modules: scaffolded and ready to build incrementally

## Phase 1 output artifacts

After running the pipeline, generated files are:

- `artifacts/phase1/dataset_summary.json`
- `artifacts/phase1/manifest.csv`
- `artifacts/phase1/manifest.parquet`
- `artifacts/phase1/normalization_stats.yaml`
- `artifacts/phase1/preprocessed_images/<split>/*.jpg`

## Execution

```bash
uv sync
uv run python scripts/phase1_dataset_pipeline.py --data-root ../Data --image-size 128
```

Or run by phase index:

```bash
uv run python scripts/run_phase.py 1 -- --data-root ../Data --image-size 128
```

Quick sanity run on a subset:

```bash
uv run python scripts/phase1_dataset_pipeline.py --data-root ../Data --max-images 2000 --stats-sample-size 1000
```

Phase 2 VAE training:

```bash
uv sync --extra train
uv run python scripts/phase2_train_vae.py --epochs 10 --batch-size 64 --latent-dim 256 --norm-mode diffusion
```

Quick sanity run:

```bash
uv run python scripts/phase2_train_vae.py --epochs 1 --batch-size 8 --max-train-samples 128 --max-val-samples 64
```

## Next implementation order

1. Phase 2 VAE development (encoder, reparameterization, decoder, train loop)
2. Phase 3 latent space viewer for shape/distribution/value inspection
3. Phase 4 latent diffusion scheduler + forward process
4. Phase 5 latent U-Net with time embedding
5. Phase 6 latent diffusion training
6. Phases 8-14 visualization, GIF export, scheduler benchmark, web UI, release
 
đây là dự án tôi muốn thực hiện Latent Diffusion Playground Interactive Visualization of VAE, DDPM and U-Net
Mục tiêu cuối cùng
Xây dựng một nền tảng trực quan giúp người dùng:
- Hiểu cách VAE nén ảnh vào latent space
- Quan sát latent representation được hình thành như thế nào
- Hiểu DDPM hoạt động trong latent space
- Quan sát Forward Diffusion Process
- Quan sát Reverse Diffusion Process
- Hiểu vai trò của U-Net trong việc dự đoán và loại bỏ nhiễu
- So sánh các Noise Scheduler (Linear, Cosine)
- Trực quan hóa Feature Maps của VAE và U-Net
- Xuất GIF minh họa từng giai đoạn
- Tương tác trực tiếp trên Web UI
- Hiểu kiến trúc nền tảng phía sau các mô hình như Stable Diffusion
Cần nghiên cứu
Diffusion
- Forward Process
- Reverse Process
- Gaussian Noise
- Markov Chain
 DDPM
- Noise Schedule
- Sampling Process
- Training Objective
U-Net
- Encoder
- Decoder
- Skip Connection
Time Embedding
- Sinusoidal Embedding
Phase 1 — Dataset Pipeline
tôi sử dụng dataset CelebFaces Attributes Dataset (CelebA) có trong đường dẫn "C:\Users\nguye\Desktop\Deeplearning\DDPM-U-Net\Data"
Phase 2 — VAE Development (Mới)
Mục tiêu
Xây dựng VAE trước khi làm Diffusion.
Module 1
Encoder
Image
 ↓
Conv Layers
 ↓
μ
σ
Module 2
Reparameterization
Công thức trung tâm:
z=μ+σϵz=\mu+\sigma\epsilonz=μ+σϵ
Module 3
Decoder
Latent
 ↓
Upsampling
 ↓
Image
Training
Loss:
Reconstruction Loss
KL Divergence
Phase 3 — Latent Space Viewer (Mới)
Mục tiêu
Trực quan hóa latent space.
Chức năng
Upload ảnh.
Hiển thị:
Image
 ↓
Encoder
 ↓
Latent Vector
Người dùng thấy:
z shape
distribution
latent dimension
Giá trị
Rất ít repo DDPM có phần này.
Phase 2 cũ — Diffusion Core Engine
Mục tiêu
Xây dựng trái tim của DDPM.
Module 1
Scheduler
Xây dựng:
Linear Scheduler
Cosine Scheduler
Tính toán:
beta
alpha
alpha_hat
Module 2
Forward Process
Input:
Image
Output:
Noisy Image
Người dùng có thể xem:
t = 0
t = 100
t = 300
t = 700
t = 1000
Phase 4 — Diffusion Core Engine
(Phase 2 cũ)
Khác biệt
Forward Process không chạy trên Image.
Forward Process chạy trên:
Latent Space
Pipeline
Image
 ↓
VAE Encoder
 ↓
Latent
 ↓
Add Noise
Phase 3 cũ — U-Net Development
Mục tiêu
Tự xây dựng U-Net thay vì dùng thư viện có sẵn.
Thành phần
Encoder
Conv
BatchNorm
ReLU
Downsample
Bottleneck
Feature Compression
Decoder
Upsample
Skip Connection
Time Embedding
Thêm thông tin timestep vào mạng.
Phase 5 — Latent U-Net Development
(Phase 3 cũ)
Mục tiêu
UNet học trên latent.
Input:
Noisy Latent
Output:
Predicted Noise
Thành phần
- Encoder
- Decoder
- Skip Connections
- Time Embedding
Phase 4 cũ— DDPM Training
Mục tiêu
Huấn luyện U-Net dự đoán nhiễu.
Quy trình
Image
 ↓
Random Timestep
 ↓
Add Noise
 ↓
UNet
 ↓
Predict Noise
 ↓
Loss
Logging
Lưu:
Loss
Checkpoint
Generated Samples
Phase 6 — Latent Diffusion Training
(Phase 4 cũ)
Pipeline
Image
 ↓
VAE Encoder
 ↓
Latent
 ↓
Add Noise
 ↓
UNet
 ↓
Predict Noise
 ↓
Loss
Logging
Loss
Latent Visualization
Generated Samples
Phase 8 — Forward Diffusion Viewer
Hiển thị:
Image
 ↓
Encoded Latent
 ↓
Noise Level 10%
 ↓
Noise Level 50%
 ↓
Noise Level 100%
Toggle
Người dùng có thể xem:
Image Space
hoặc
Latent Space
Phase 9 — Reverse Diffusion Viewer
Hiển thị:
Random Latent
 ↓
Denoising
 ↓
Clean Latent
 ↓
Decoded Image
Export GIF
latent_reverse.gif
Phase 10 — Explainable U-Net + VAE
VAE Viewer
Hiển thị:
Input Image
 ↓
Encoder Features
 ↓
Latent Features
 ↓
Decoder Features
 ↓
Reconstructed Image
U-Net Viewer
Hiển thị:
Encoder Layers
Bottleneck
Decoder Layers
Phase 11 — Scheduler Comparison Lab
So sánh:
Linear
Cosine
Đánh giá:
FID
Loss
Sampling Speed
Phase 12 — Interactive Web Application
Tab 1
VAE Explorer
Image
 ↓
Latent
 ↓
Reconstruction
Tab 2
Forward Diffusion
Latent
 ↓
Noise
Tab 3
Reverse Diffusion
Noise
 ↓
Image
Tab 4
U-Net Features
Feature Maps.
Tab 5
Scheduler Comparison
Benchmark.
Phase 13 — GIF Export System
(Cập nhật)
Sinh:
vae_reconstruction.gif
forward.gif
reverse.gif
sampling.gif
Phase 14 — GitHub Release Version
README nên có:
1. VAE Architecture
2. Latent Space Visualization
3. Forward Diffusion GIF
4. Reverse Diffusion GIF
5. VAE Reconstruction
6. U-Net Feature Maps
7. Scheduler Benchmark
8. Interactive Demo
`````
bây giờ hãy tạo môi trường ảo uv và bắt đầu từ việc đọc và phân loại dữ liệu, tiền xử lý dữ liệu, và chuẩn hóa dữ liệu nếu cần sau đó hãy xây dựng thống theo từng phase