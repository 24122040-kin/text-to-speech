# Báo cáo Deploy Piper (vi) — Qualcomm AI Hub / Dragonwing IQ-9075 EVK (NPU)

> TTS tiếng Việt **Piper `vi_VN-vais1000-medium`** — chạy **100% trên Hexagon NPU (HTP v73)** của Snapdragon/Dragonwing IQ-9075 EVK. Pipeline được tách thành 4 sub-model tĩnh theo đúng recipe chính thức của Qualcomm (`qualcomm/ai-hub-models` → `pipertts_*`), mọi suy luận mạng nơ-ron đều chạy trên NPU; host chỉ làm glue không-thần-kinh (phonemize espeak + alignment argmax + ghép cửa sổ vocoder).

---

## 1. Tóm tắt kết quả

| Hạng mục | Kết quả |
|---|---|
| Model | Piper (vi) `vi_VN-vais1000-medium` (VITS end-to-end, 22050 Hz) |
| Thiết bị đích | **Dragonwing IQ-9075 EVK** (Qualcomm QCS9075, Hexagon HTP v73, soc_model 77) |
| Chạy trên | **NPU thuần (qnn_context_binary — HTP), không CPU fallback** |
| 4 sub-model compile | ✅ encoder / sdp / flow / decoder — **SUCCESS cả 4** |
| Precision | float → **fp16 trên HTP** (recipe chính thức Qualcomm cho Piper) |
| Cosine similarity (NPU vs fp32) | **mean = 0.999855** (pass 8/8, ngưỡng > 0.9) |
| — flow z tensor cos | **mean = 0.999999** |
| Đánh giá round-trip ASR (PhoWhisper-small) | WER trung bình = 0.3425, CER = 0.2875 |
| Dung lượng QNN binary | encoder 15.5 MB + sdp 2.0 MB + flow 19.4 MB + decoder 3.5 MB ≈ **40.4 MB** |

---

## 2. Kiến trúc model (VITS — end-to-end text→audio)

Piper (vi) là **VITS** (Variational Inference with adversarial Training for end-to-end TTS): phoneme → mel → waveform. Vì HTP yêu cầu shape tĩnh hoàn toàn và graph động của model gộp (NonZero/Range/ScatterND trong flow decoder, độ dài audio động) không compile được, model được tách thành **4 component shape-tĩnh** — đúng cách Qualcomm chính thức deploy PiperTTS:

```
Text (tiếng Việt)
  │  espeak-ng phonemize (host, phi-thần-kinh)      → phoneme ids [1,512] int32
  ▼
[1] ENCODER (text encoder)      NPU   x,x_lengths → x_encoded, m_p, logs_p, x_mask
  ▼
[2] SDP (duration predictor)    NPU   x_encoded,x_mask,scales → y_lengths, w_ceil
  │  host: generate_path (argmax alignment, phi-thần-kinh) → attn_squeezed
  ▼
[3] FLOW (normalizing flow, reverse)  NPU   m_p,logs_p,y_mask,attn,noise → z
  │  host: chia cửa sổ 64-frame, overlap 12
  ▼
[4] DECODER (HiFi-GAN vocoder)  NPU   z [1,192,64] → audio [1,1,16384]
  │  host: ghép cửa sổ
  ▼
WAV 22050 Hz
```

### 2.1 Encoder (text encoder — 6.3M params)

- Embedding phoneme 256→192 (`sid` [256,192]) + scale √hidden
- 6 lớp FFT (attention 2 heads, 192 chiều, filter_channels 768):
  - Multi-head self-attention tương đối (conv_q/k/v/o kernel 1 + `emb_rel_k/v` [1,9,96])
  - FFN: 2× Conv1d kernel 3 (trong graph: Pad+Conv), GELU, dropout
- Outputs: `x_encoded` [1,192,512], prior `m_p`/`logs_p` [1,192,512], `x_mask` [1,1,512]

### 2.2 SDP (stochastic duration predictor — 0.6M params)

- pre-conv + WaveNet (8 flows đảo, bỏ flow[-2] theo recipe Qualcomm để nhanh) + proj
- **Nhiễu xác định**: constant noise pattern (0.5) × noise_scale_w (thay RandomNormalLike — NPU không hỗ trợ op ngẫu nhiên)
- Outputs: `y_lengths` [1] (tổng duration), `w_ceil` [1,1,512] (duration từng phoneme)

### 2.3 Flow (normalizing flow — 7.4M params)

- 8 coupling layer (chỉ cần 4 WN trong reverse-mode: flows 6,4,2,0), affine coupling + WN (Conv1d kernel 5, dilations)
- Reverse mode: z_p = m_p·attnᵀ + fixed_noise·exp(logs_p)·noise_scale; `fixed_noise` là buffer hằng (deterministic)
- Input `attn_squeezed` [1,1536,512] (attention từ alignment host), `y_mask` [1,1,1536]
- Output: `z` [1,192,1536] (mel prior đã biến đổi)

### 2.4 Decoder (HiFi-GAN vocoder — 1.7M params)

- conv_pre (kernel 7) → 3× ConvTranspose1d upsample (8,8,4; kernel 16,16,8; 256→32 kênh)
- 9 ResBlock2 (MRF): kernel (3,5,7), dilation ((1,2),(2,6),(3,12)) — mỗi block 2×Conv1d
- conv_post (kernel 7) + tanh → audio [1,1,16384] (64 frames × hop 256)
- Cửa sổ: 40 frame + overlap 12 (host chia/ghép; mỗi lần gọi NPU 1 window)

### 2.5 Số liệu graph thực đo (từ ONNX export)

| Component | Nodes | Conv | ConvTranspose | Params |
|---|---|---|---|---|
| encoder | 2596 | 37 | 0 | 6.3M |
| sdp | 2742 | 32 | 0 | 0.6M |
| flow | 752 | 40 | 0 | 7.4M |
| decoder | 70 | 20 | 3 | 1.7M |
| **Tổng** | | 129 | 3 | **≈ 16.0M** |

---

## 3. Bằng chứng chạy thuần NPU (không CPU fallback)

### 3.1 Cách compile (options — theo recipe chính thức Qualcomm)

| Component | Compile options (qnn_context_binary) | Input specs |
|---|---|---|
| encoder | `--truncate_64bit_tensors --truncate_64bit_io` | x int32 [1,512], x_lengths int32 [1] |
| sdp | `--truncate_64bit_tensors --truncate_64bit_io` | x_encoded/x_mask float32, length_scale/noise_scale_w [1] |
| flow | `--truncate_64bit_tensors --truncate_64bit_io` | m_p/logs_p [1,192,512], y_mask [1,1,1536], attn_squeezed [1,1536,512], noise_scale [1] |
| decoder | `--quantize_io` | z float32 [1,192,64] → audio [1,1,16384] |

`qnn_context_binary` = **QNN context binary chạy trực tiếp trên HTP (Hexagon Tensor Processor)** — toàn bộ op nằm trong graph NPU; không có op nào rơi xuống CPU (không `--use_cpu` fallback, không op unsupported). So sánh với asset chính thức `qualcomm/PiperTTS-EN` (qcs9075, QAIRT 2.45): **kích thước binary gần như trùng khớp** — encoder 15.53 MB vs 15.53 MB, sdp 1.97 vs 1.88 MB, flow 19.44 vs 19.44 MB, decoder 3.53 vs 3.36 MB — cùng cấu trúc 4 graph HTP.

### 3.2 QNN context binaries đã tạo

| Component | Job ID | Binary | Size |
|---|---|---|---|
| encoder | `jgnn0rejg` | piper_vi_encoder.iq9075.bin | 15.53 MB |
| decoder | `jp8x2w4zg` | piper_vi_decoder.iq9075.bin | 3.53 MB |
| sdp | `jp0j4ev2g` | piper_vi_sdp.iq9075.bin | 1.97 MB |
| flow | `jgk4vr9yp` | piper_vi_flow.iq9075.bin | 19.44 MB |

### 3.3 Chạy inference trên NPU (chuỗi đầy đủ)

Chuỗi inference trên IQ-9075: **encoder → sdp → flow → decoder**, mọi activation trung gian lấy **từ NPU** (không dùng activation fp32 CPU).

- `enc`: job `j5672qdvp` — https://workbench.aihub.qualcomm.com/jobs/j5672qdvp/
- `sdp`: job `j5mmevdy5` — https://workbench.aihub.qualcomm.com/jobs/j5mmevdy5/
- `flow`: job `jgk4vr4yp` — https://workbench.aihub.qualcomm.com/jobs/jgk4vr4yp/
- `dec`: job `j5672q8vp` — https://workbench.aihub.qualcomm.com/jobs/j5672q8vp/
- `sdp`: job `jpvdqzjjp` — https://workbench.aihub.qualcomm.com/jobs/jpvdqzjjp/
- `flow`: job `j57426yv5` — https://workbench.aihub.qualcomm.com/jobs/j57426yv5/
- `dec`: job `j5w7wxqmg` — https://workbench.aihub.qualcomm.com/jobs/j5w7wxqmg/
- `dec`: job `jpyxznn05` — https://workbench.aihub.qualcomm.com/jobs/jpyxznn05/

---

## 4. Cosine similarity (NPU vs fp32 reference)

- **Ngưỡng yêu cầu: > 0.9** — đạt **8/8**, audio cos mean = **0.999855**

| # | Câu (đầu) | flow z cos | audio cos | Độ dài | Kết quả |
|---|---|---|---|---|---|
| 0 | Nông nghiệp tự cung tự tiêu là một hệ thống đ | 1.0 | **0.999875** | 9.253s | ✅ PASS |
| 1 | Bất kỳ ai muốn lái xe trên khu vực cao hoặc b | 0.999999 | **0.999792** | 6.095s | ✅ PASS |
| 2 | Chuẩn 802.11n hoạt động trên cả hai tần số 2. | 0.999999 | **0.999858** | 5.863s | ✅ PASS |
| 3 | Các công ty chuyển phát được trả tiền để vận  | 1.0 | **0.999876** | 3.796s | ✅ PASS |
| 4 | đối với springboks trận này đã giúp đội tuyển | 0.999999 | **0.999872** | 3.529s | ✅ PASS |
| 5 | Mái ấm được coi là nơi cung cấp những gì thiế | 0.999999 | **0.999797** | 4.714s | ✅ PASS |
| 6 | Nếu không có bàn là hoặc nếu bạn không thích  | 1.0 | **0.999888** | 5.027s | ✅ PASS |
| 7 | Giống như cách mà Paris được gọi là kinh đô t | 0.999999 | **0.99988** | 7.57s | ✅ PASS |

> **Phương pháp:** reference là pipeline 4-component chạy fp32 (ORT) cùng cấu trúc. Vì SDP (duration predictor) có phép `ceil()` — sai số fp16 cỡ 1 frame làm lệch alignment — so sánh audio dùng **matched alignment** (durations/attention lấy từ NPU, chỉ flow+decoder chạy fp32), cô lập đúng sai số lượng tử của NPU. Ở mức tensor (flow z) so sánh raw cùng input: **cos = 0.999999**.

---

## 5. Đánh giá chất lượng TTS (round-trip ASR)

Audio NPU → ASR tiếng Việt (PhoWhisper-small) → so transcript với câu gốc (WER/CER).

| # | Câu gốc | Hypothesis (ASR) | WER | CER | cos_sim |
|---|---|---|---|---|---|
| 0 | Nông nghiệp tự cung tự tiêu là một hệ thống đơn gi | nông nghiệp tự cung tự tiêu là một hệ thống đơn gi | 0.2453 | 0.211 | 0.999875 |
| 1 | Bất kỳ ai muốn lái xe trên khu vực cao hoặc băng đ | bất kỳ ai muốn lái xe chơi khu vực cao hoặc băng đ | 0.1481 | 0.0609 | 0.999792 |
| 2 | Chuẩn 802.11n hoạt động trên cả hai tần số 2.4 Ghz | chuần trăm nghìn hai chấm mười một en ở hoạt động  | 1.2143 | 1.0862 | 0.999858 |
| 3 | Các công ty chuyển phát được trả tiền để vận chuyể | các công ty chuyển phát được trả tiền để vận chuyể | 0.6087 | 0.6083 | 0.999876 |
| 4 | đối với springboks trận này đã giúp đội tuyển kết  | đối với thành trận này đã giúp đội tuyển kết thúc  | 0.125 | 0.1558 | 0.999872 |
| 5 | Mái ấm được coi là nơi cung cấp những gì thiết yếu | máy ấm được coi là nơi cung cấp những gì thiết yếu | 0.0435 | 0.0103 | 0.999797 |
| 6 | Nếu không có bàn là hoặc nếu bạn không thích đi bí | nếu không có bàn hoặc nếu bạn không thích đi biết  | 0.1364 | 0.0652 | 0.999888 |
| 7 | Giống như cách mà Paris được gọi là kinh đô thời t | giống như cách mà tại được gọi là kình đô thời tra | 0.2188 | 0.1026 | 0.99988 |

**Mean WER = 0.3425, Mean CER = 0.2875** (n=8)

---

## 6. Đối chiếu với asset chính thức của Qualcomm

Tham chiếu trực tiếp: `qualcomm/PiperTTS-EN` (HF) → `pipertts_en-voice_ai-float-qualcomm_qcs9075.zip` (QAIRT 2.45):

| File | Official EN (qcs9075) | Của dự án (vi, IQ-9075) |
|---|---|---|
| encoder.bin | 15,527,936 B | 15,532,032 B |
| sdp.bin | 1,970,176 B | 1,880,064 B |
| flow.bin | 19,435,520 B | 19,435,520 B |
| decoder.bin | 3,526,656 B | 3,363,840 B |

Kích thước gần như trùng khớp → cùng loại artifact HTP. Khác biệt chính: bản EN dùng **Charsiu T5 G2P** (`charsiu_encoder/decoder.bin`), bản vi dùng **espeak-ng phonemize** (giống bản `pipertts_it` — cũng build từ ONNX).

---

## 7. Artifacts

| File | Mô tả |
|---|---|
| `outputs/piper_vi_npu/components/*.onnx` | 4 component ONNX shape-tĩnh (fp32) |
| `outputs/piper_vi_npu/components/*.iq9075.bin` | 4 QNN context binary (NPU) |
| `outputs/piper_vi_npu/calib/*.npz` | Calibration + test activations |
| `outputs/piper_vi_npu/hw/` | Hardware outputs + audio + results |
| `outputs/piper_vi_npu/hw/similarity_results.json` | Cosine similarity per test |
| `outputs/piper_vi_npu/hw/evaluation_results.json` | Round-trip ASR eval |
| `src/step4_npu/` | Pipeline scripts (export/compile/run/verify/eval/report) |

---

## 8. Kết luận

1. **Đã deploy Piper (tiếng Việt) lên Qualcomm AI Hub / Dragonwing IQ-9075 EVK thành công** — 4 sub-model (encoder, sdp, flow, decoder) compile thành QNN context binary chạy thuần Hexagon NPU (v73), theo đúng recipe chính thức của Qualcomm; không còn theo đường deploy cũ (model gộp) vốn compile FAIL nhiều tuần.
2. **Chạy thuần NPU**: toàn bộ 129 Conv + 3 ConvTranspose + transformer/flow/vocoder đều nằm trong graph HTP; host chỉ làm phonemize/alignment/ghép cửa sổ (phi-thần-kinh).
3. **Similarity đạt ✅ trên ngưỡng 0.9** — so với fp32 reference.
4. **Chất lượng**: round-trip ASR WER 0.3425 / CER 0.2875.
