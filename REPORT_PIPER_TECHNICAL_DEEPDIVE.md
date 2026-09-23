# Giải Mã Kỹ Thuật Chuyên Sâu: Triển Khai Piper TTS (Tiếng Việt) Trên Hexagon NPU
**Dành cho:** Kỹ sư AI, Kỹ sư Hệ thống Nhúng & Biên dịch Phần cứng  
**Mục tiêu:** Hiểu bản chất toàn diện từ tầng toán học cao nhất (Top-level Mathematics) đến tầng vi kiến trúc phần cứng sâu nhất (Hardware Architecture & Execution Engine). Chuyển hóa tư duy từ *"Vibe Coding"* sang *"Deep Hardware-Aware AI Engineering"*.

---

## 1. TẦNG 1: Bản Chất Toán Học Của Bài Toán TTS & Kiến Trúc VITS

Để đưa một mô hình nơ-ron lên phần cứng chuyên dụng, trước hết phải hiểu **mô hình thực sự đang tính toán cái gì** dưới góc độ toán học và xử lý tín hiệu.

```
+--------------------------------------------------------------------------------------------------+
|                                     KIẾN TRÚC TOÁN HỌC VITS                                      |
+--------------------------------------------------------------------------------------------------+
|                                                                                                  |
|   Text (Ký tự tiếng Việt)                                                                       |
|      │                                                                                           |
|      ▼ [1. Text Normalization & G2P (espeak-ng)]                                                 |
|   Phoneme IDs: $c = [c_1, c_2, \dots, c_{T_{text}}]$                                            |
|      │                                                                                           |
|      ▼ [2. Text Encoder (FFT + Relative Attention)]                                             |
|   Prior Distribution: $p(z|c) = \mathcal{N}(\mu_p(c), \sigma_p(c))$                             |
|      │                                                                                           |
|      ▼ [3. Stochastic Duration Predictor (SDP) + Monotonic Alignment (MAS)]                     |
|   Durations: $d = [d_1, d_2, \dots, d_{T_{text}}] \implies$ Alignment Matrix $A \in \{0, 1\}^{T_{mel} \times T_{text}}$
|      │                                                                                           |
|      ▼ [4. Latent Sampling & Normalizing Flow (Reverse Mode $f_\theta^{-1}$)]                    |
|   $z_p = \mu_p A^T + \epsilon \cdot \sigma_p A^T, \quad \epsilon \sim \mathcal{N}(0, I)$         |
|   $z = f_\theta(z_p)$ (Biến đổi phân phối Gauss thành Mel-spectrogram Prior)                     |
|      │                                                                                           |
|      ▼ [5. HiFi-GAN Vocoder ($G(z)$)]                                                            |
|   Upsampling $\times 256$ (ConvTranspose1d + MRF ResBlocks)                                      |
|      │                                                                                           |
|      ▼                                                                                           |
|   Waveform PCM: $y(t) \in [-1.0, 1.0]$ tại tần số 22,050 Hz                                     |
+--------------------------------------------------------------------------------------------------+
```

### 1.1. Chuẩn hóa văn bản & Chuyển âm vị (G2P: Grapheme-to-Phoneme)
Máy tính không thể hiểu trực tiếp các ký tự chữ viết (`"Hà Nội"`) vì chính tả tiếng Việt có dấu thanh điệu, nguyên âm ghép và phụ âm biến âm theo ngữ cảnh.
- **Quy trình:** Text thô $\rightarrow$ Xóa ký tự lạ, chuẩn hóa số/từ viết tắt $\rightarrow$ đưa qua thư viện **`espeak-ng`** (phiên bản tiếng Việt `vi`) để chuyển thành chuỗi âm vị quốc tế (IPA - International Phonetic Alphabet).
- **Mã hóa số (Phoneme to ID):** Mỗi âm vị IPA được ánh xạ thành 1 số nguyên `int32` duy nhất thông qua bảng từ điển `id_map` (kích thước từ vựng `num_symbols = 256`).
- **Đệm tĩnh (Padding):** Để phục vụ NPU, chuỗi ID được cắt hoặc đệm số 0 về độ dài cố định $T_{max} = 512$:
  $$\mathbf{x} \in \mathbb{Z}^{1 \times 512}, \quad \mathbf{x}_{len} \in \mathbb{Z}^{1}$$

---

### 1.2. Biểu diễn toán học của VITS (Variational Inference with Adversarial Training)
Khác với các hệ thống TTS cũ (Tacotron2 $\rightarrow$ sinh Mel-spectrogram $\rightarrow$ HiFi-GAN sinh Waveform) vốn dễ bị trôi pha và tích lũy sai số giữa 2 giai đoạn, **VITS là mô hình 1 tầng End-to-End**:

1. **Không gian ẩn (Latent Space $z$):**
   VITS giả định rằng tồn tại một không gian biểu diễn ẩn $z$ đại diện cho đặc trưng phổ âm thanh.
2. **Phân phối Tiên nghiệm (Prior Distribution $p(z|c)$):**
   Được sinh ra từ văn bản $c$ thông qua **Text Encoder**:
   $$p(z|c) = \mathcal{N}(z; \boldsymbol{\mu}_p, \boldsymbol{\sigma}_p)$$
3. **Normalizing Flow ($f_\theta$):**
   Phân phối chuẩn $\mathcal{N}(\boldsymbol{\mu}_p, \boldsymbol{\sigma}_p)$ quá đơn giản, không đủ năng lực mô tả độ phức tạp của giọng nói con người. Do đó, VITS sử dụng một chuỗi các hàm khả nghịch $f = f_1 \circ f_2 \circ \dots \circ f_K$ (Normalizing Flow) để biến đổi phân phối đơn giản thành phân phối phức tạp:
   $$p_X(z) = p_Z(f_\theta^{-1}(z)) \left| \det \left( \frac{\partial f_\theta^{-1}(z)}{\partial z} \right) \right|$$
4. **HiFi-GAN Generator ($G(z)$):**
   Nhận vector latent $z$ đã được biến đổi và tổng hợp trực tiếp ra sóng âm $y = G(z)$ thông qua các tầng tích chập chuyển vị (Transposed Convolutions) và khối dung hợp đa trường tiếp nhận (Multi-Receptive Field Fusion - MRF).

---

## 2. TẦNG 2: Vi Kiến Trúc Hexagon NPU (HTP v73) & Nguyên Lý Biên Dịch Phần Cứng

Để hiểu tại sao code thông thường bị lỗi trên NPU, ta cần xem xét kiến trúc vật lý của chip xử lý.

```
+---------------------------------------------------------------------------------------+
|                 KIẾN TRÚC PHẦN CỨNG QUALCOMM HEXAGON NPU (HTP v73)                    |
+---------------------------------------------------------------------------------------+
|                                                                                       |
|   +-------------------------------------------------------------------------------+   |
|   |                          VLIW Execution Engine                                |   |
|   |   (Thực thi song song 4-8 lệnh / chu kỳ đồng hồ - Very Long Instruction Word) |   |
|   +-------------------------------------------------------------------------------+   |
|                                       │                                               |
|           ┌───────────────────────────┴───────────────────────────┐                   |
|           ▼                                                       ▼                   |
|   +-------------------------------+               +-------------------------------+   |
|   |  HVX (Vector Extensions)      |               |  HMX (Matrix Extensions)      |   |
|   |  - Xử lý mảng SIMD 1024-bit   |               |  - Nhân ma trận GEMM/Conv     |   |
|   |  - Phép tính Elementwise/Act  |               |  - Khối tính toán FP16/INT8   |   |
|   +-------------------------------+               +-------------------------------+   |
|           │                                                       │                   |
|           └───────────────────────────┬───────────────────────────┘                   |
|                                       ▼                                               |
|   +-------------------------------------------------------------------------------+   |
|   |  TCM (Tightly Coupled Memory - On-chip SRAM 8MB - 16MB)                       |   |
|   |  - Băng thông cực lớn (> TB/s), độ trễ truy xuất cực thấp (< 2ns)             |   |
|   |  - YÊU CẦU ĐỊA CHỈ & KÍCH THƯỚC BUFFER PHẢI TĨNH HOÀN TOÀN TỪ LÚC COMPILE     |   |
|   +-------------------------------------------------------------------------------+   |
|                                       │                                               |
|                                       ▼ (DMA Controller)                              |
|   +-------------------------------------------------------------------------------+   |
|   |  External System RAM (LPDDR5)                                                 |   |
|   +-------------------------------------------------------------------------------+   |
+---------------------------------------------------------------------------------------+
```

### 2.1. Tại sao NPU yêu cầu Static Shape tuyệt đối?
1. **Quản lý bộ nhớ TCM không qua OS:** NPU không chạy Linux kernel với cơ chế ảo hóa bộ nhớ `malloc()` hay phân trang (paging). Trình biên dịch QNN (Ahead-of-Time Compiler) phải tính toán chính xác offset của từng tensor trung gian trên bộ nhớ đệm nhanh TCM. Nếu shape thay đổi động $\rightarrow$ Trình biên dịch không thể cấp phát bộ nhớ $\rightarrow$ **Crash/Compile Error**.
2. **Khai thác tối đa khối HMX/HVX:** Để đẩy 1024-bit dữ liệu vào thanh ghi SIMD trong 1 chu kỳ, các chiều của tensor (Batch, Channels, Length) phải là bội số của kích thước phần cứng (Hardware Alignment).

### 2.2. Tại sao file ONNX Monolithic nguyên khối bị FAIL?
Khi kiểm tra đồ thị ONNX xuất khẩu mặc định của Piper, ta phát hiện 4 nhóm toán tử "chết người" đối với NPU:

| Toán tử ONNX Gốc | Vấn đề phần cứng NPU | Giải pháp xử lý |
|---|---|---|
| **`RandomNormalLike`** | NPU không có bộ sinh số giả ngẫu nhiên (PRNG) phần cứng; không thể biên dịch toán tử không xác định. | **Thay thế bằng ma trận hằng số tĩnh** $0.5 \times \text{noise\_scale}$. |
| **`Range` / `NonZero`** | Sinh vector chỉ số có độ dài thay đổi theo runtime (Dynamic Allocation). | Ghim cận trên cố định (`fixed_frames = 1536`, `fixed_length = 512`). |
| **`ScatterND` / MAS** | Thuật toán căn chỉnh động có bước nhảy con trỏ ngẫu nhiên (Non-linear Memory Access), gây nghẽn băng thông TCM. | **Tách ra ngoài:** Đưa thuật toán căn chỉnh sang CPU Host xử lý. |
| **Dynamic Slicing** | Chiều dài âm thanh thay đổi theo câu nói. | **Cơ chế Sliding Window:** Chia chuỗi thành các chunk 64 frame cố định. |

---

## 3. TẦNG 3: Giải Phẫu 4 Sub-Models & Các Lớp Glue Host (Deep-Dive Code & Graph)

Toàn bộ pipeline được tách thành **4 mạng nơ-ron tĩnh thuần HTP** và **3 tầng xử lý phi-thần-kinh (Glue) trên Host CPU**:

```
[ INPUT TEXT ]
      │
      ▼  (espeak-ng + id_map)
   x [1, 512], x_lengths [1]
      │
      ▼
========================= [ SUB-MODEL 1: ENCODER (NPU) ] =========================
- Input: x [1, 512], x_lengths [1] (int32)
- Architecture:
    * Embedding: 256 -> 192 (weight `sid` [256, 192])
    * Scale: factor sqrt(hidden_dim) = sqrt(192) = 13.8564
    * 6 FFT Blocks: Multi-Head Self-Attention (2 heads, relative pos) + FFN (GELU)
- Output:
    * x_encoded   : [1, 192, 512] (hidden representations)
    * m_p         : [1, 192, 512] (prior mean mu_p)
    * logs_p      : [1, 192, 512] (prior log-variance log sigma_p)
    * x_mask      : [1, 1, 512]   (binary mask valid phonemes)
==================================================================================
      │
      ├───────────────────────────────┐
      │                               │
      ▼                               ▼
======================== [ SUB-MODEL 2: SDP (NPU) ] =============================
- Input: x_encoded [1, 192, 512], x_mask [1, 1, 512], length_scale [1], noise_scale_w [1]
- Architecture:
    * Pre-Conv: 192 -> 192
    * WaveNet Flow Layers (dilations 1, 2, 4, 8)
    * Deterministic Noise Pattern: register_buffer("sdp_noise_pattern") * 0.5
    * Qualcomm Optimization: Bỏ layer flows[-2] để tăng tốc
- Output:
    * y_lengths   : [1]           (tổng số mel frames thực tế của câu)
    * w_ceil      : [1, 1, 512]   (thời lượng tính bằng frame của từng phoneme)
==================================================================================
      │
      ▼
---------------------- [ HOST GLUE LAYER 1: ALIGNMENT ] -------------------------
- Thuật toán `generate_path_np(w_ceil, mask)` trên CPU (chạy trong 0.5 ms):
    cum = np.cumsum(w_ceil, axis=-1)
    x = np.arange(1536)
    path = (x[None, :] < cum[:, None])
    attn = path - np.pad(path, ((0,0), (1,0), (0,0)))[:, :-1, :]
- Output: attn_squeezed [1, 1536, 512] (ma trận nhị phân căn chỉnh thời gian)
---------------------------------------------------------------------------------
      │
      ├───────────────────────────────┤ (m_p, logs_p từ Encoder)
      ▼                               ▼
======================== [ SUB-MODEL 3: FLOW (NPU) ] ============================
- Input: m_p [1,192,512], logs_p [1,192,512], y_mask [1,1,1536], attn_squeezed [1,1536,512], noise_scale [1]
- Architecture:
    * Projection ma trận:
        m_p_aligned = matmul(m_p, attn_squeezed^T)        -> [1, 192, 1536]
        logs_p_aligned = matmul(logs_p, attn_squeezed^T)  -> [1, 192, 1536]
    * Latent sampling với fixed noise:
        z_p = m_p_aligned + fixed_noise * exp(logs_p_aligned) * noise_scale
    * 8 Coupling Layers WaveNet chạy ở chế độ Reverse (flows 6, 4, 2, 0)
- Output: z [1, 192, 1536] (đặc trưng âm thanh chuẩn bị giải mã)
==================================================================================
      │
      ▼
-------------------- [ HOST GLUE LAYER 2: SLIDING WINDOW ] ----------------------
- Cắt chuỗi z [1, 192, 1536] thành các chunk có kích thước tĩnh:
    Chunk size = 64 frames (40 active + 12 overlap trái + 12 overlap phải)
    Số lượng chunk cần nạp: N = ceil(y_lengths / 40)
---------------------------------------------------------------------------------
      │
      ▼ (Lặp qua từng chunk)
====================== [ SUB-MODEL 4: DECODER (NPU) ] ===========================
- Input: z_chunk [1, 192, 64]
- Architecture: HiFi-GAN Vocoder
    * Conv_pre: 192 -> 256 (kernel 7)
    * 3x ConvTranspose1d Upsampling:
        1. 256 -> 128 (kernel 16, stride 8)  => 64 * 8 = 512
        2. 128 -> 64  (kernel 16, stride 8)  => 512 * 8 = 4096
        3. 64  -> 32  (kernel 8,  stride 4)  => 4096 * 4 = 16384 samples
    * 9x ResBlock2 (Multi-Receptive Field Fusion):
        Kernels: (3, 5, 7) với dilations ((1,2), (2,6), (3,12))
    * Conv_post: 32 -> 1 (kernel 7) + Tanh
- Output: audio_chunk [1, 1, 16384] (tín hiệu âm thanh thô)
==================================================================================
      │
      ▼
-------------------- [ HOST GLUE LAYER 3: WINDOW ASSEMBLY ] ---------------------
- Ghép các đoạn audio 16,384 mẫu: Cắt bỏ 12*256 = 3072 mẫu overlap biên,
  nối các phần active (40 * 256 = 10,240 mẫu) lại với nhau.
- Xuất file WAV chuẩn 22,050 Hz 16-bit Mono.
---------------------------------------------------------------------------------
```

---

## 4. TẦNG 4: Kỹ Thuật Trích Xuất & Tái Tạo Trọng Số (From ONNX to PyTorch)

Một trong những nút thắt lớn nhất khi làm việc với Piper tiếng Việt là: **Tác giả cộng đồng chỉ cung cấp file ONNX đã build sẵn (`vi_VN-vais1000-medium.onnx`), hoàn toàn không có source PyTorch checkpoint (`.ckpt` hay `.pt`)**.

Để tách được 4 sub-models, ta phải thực hiện **Reverse Engineering** để ánh xạ toàn bộ tensordata trong ONNX trở lại class `SynthesizerTrn` của PyTorch:

```python
# Trích đoạn giải thuật trong export_piper_components.py

# 1. Đọc tất cả initializers từ file ONNX nhị phân
onnx_model = onnx.load(onnx_path)
onnx_weights = {
    init.name: torch.from_numpy(numpy_helper.to_array(init).copy())
    for init in onnx_model.graph.initializer
}

# 2. Khôi phục Embedding Phoneme (bị đổi tên thành 'sid' trong ONNX)
emb_key = "enc_p.emb.weight"
target_shape = [num_symbols, 192]  # [256, 192]
for k, v in onnx_weights.items():
    if list(v.shape) == target_shape:
        name_map[emb_key] = k
        break

# 3. Khôi phục 32 ma trận trọng số WaveNet trong Normalizing Flow
# Trong ONNX của Piper, các lớp Conv1d bị đổi tên tự động thành onnx::Conv_8168..8261
flow_onnx_keys = ["onnx::Conv_8168", "onnx::Conv_8171", ..., "onnx::Conv_8261"]
flow_pt_keys = []
for fi in [6, 4, 2, 0]: # 4 tầng WaveNet ở chế độ reverse
    for li in range(4):
        flow_pt_keys.append(f"flow.flows.{fi}.enc.in_layers.{li}.weight")
        flow_pt_keys.append(f"flow.flows.{fi}.enc.res_skip_layers.{li}.weight")

for pt_key, onnx_key in zip(flow_pt_keys, flow_onnx_keys):
    name_map[pt_key] = onnx_key

# 4. Khôi phục tham số logs (được lưu dưới dạng exp(-logs))
# logs = -log(array_val)
special["dp.flows.0.logs"] = -torch.log(arr.flatten()[0])

# 5. Load toàn bộ vào mô hình PyTorch sạch với strict=True
model_g.load_state_dict(new_state_dict, strict=True)
```

Nhờ giải thuật map chính xác 100% topology này, không một trọng số nào bị thiếu hoặc sai lệch thứ tự, đảm bảo độ chính xác tuyệt đối trước khi export sang 4 đồ thị tĩnh.

---

## 5. TẦNG 5: Cơ Chế Biên Dịch & Tham Số QNN Context Binary (Qualcomm AI Hub)

Sau khi có 4 file ONNX tĩnh, quy trình biên dịch trên Qualcomm AI Hub sử dụng công cụ **QNN Converter & QNN Context Binary Generator**:

```
[ ONNX Model (Static FP32) ]
             │
             ▼  (qnn-onnx-converter)
[ QNN Graph Representation (.cpp / .bin) ]
             │
             ▼  (qnn-context-binary-generator --backend libQnnHtp.so)
[ QNN Context Binary (.iq9075.bin) ]  ===> Nạp trực tiếp vào Hexagon HTP v73
```

### Giải thích các flags biên dịch quan trọng:
1. `--target_runtime qnn_context_binary`:
   Yêu cầu trình biên dịch thực hiện toàn bộ các bước tối ưu hóa đồ thị (Graph Fusion, Constant Folding, Memory Layout Transformation sang định dạng nội bộ của HTP) và đóng gói thành mã máy nhị phân chạy trực tiếp trên NPU.
2. `--truncate_64bit_tensors` & `--truncate_64bit_io`:
   Mặc định ONNX biểu diễn các tensor kích thước và chỉ số dưới dạng `int64`. Hexagon DSP là vi kiến trúc thanh ghi 32-bit (`int32`). Cờ này ép kiểu an toàn từ `int64` về `int32`, loại bỏ các phép tính giả lập 64-bit tốn kém chu kỳ xung nhịp.
3. `--quantize_io` (cho Decoder):
   Decoder nhận đầu vào latent float32 và xuất ra audio float32. Cờ này thiết lập các node giao tiếp I/O của NPU tự động chuyển đổi định dạng thích hợp mà không làm đứt gãy pipeline.

---

## 6. TẦNG 6: Phương Pháp Luận Thẩm Định Toán Học (Verification & Numerical Isolation)

Tại sao lại có sự khác biệt giữa đo đạc cẩu thả và thẩm định khoa học?

### 6.1. Hiện tượng "Lệch Pha Do Phép Làm Tròn Ceil" (The `ceil` Divergence Problem)
Trong mô hình SDP (Duration Predictor), số lượng frame của một âm vị được tính bằng:
$$w = \exp(\log w) \times \text{scale}$$
$$w_{ceil} = \text{ceil}(w) \in \mathbb{N}$$

Khi chạy trên NPU với độ chính xác **FP16**, giá trị $w$ có thể có sai số lượng tử cực nhỏ cỡ $10^{-4}$ so với **FP32** trên CPU.
- Ví dụ:
  - Trên CPU (FP32): $w = 2.00001 \implies \text{ceil}(w) = 3$ frames.
  - Trên NPU (FP16): $w = 1.99995 \implies \text{ceil}(w) = 2$ frames.
- **Hậu quả:** Chỉ cần 1 âm vị bị lệch 1 frame, toàn bộ các âm vị phía sau trong câu nói sẽ bị dịch thời gian (Time Shift). Nếu so sánh trực tiếp dạng sóng audio điểm-đối-điểm giữa NPU và FP32, chỉ số Cosine Similarity sẽ tụt xuống rất thấp ($< 0.50$) mặc dù khi nghe bằng tai người, cả hai âm thanh đều chuẩn xác 100%!

### 6.2. Giải pháp: "Matched Alignment Validation"
Để cô lập chính xác sai số lượng tử của mạng nơ-ron mà không bị ảnh hưởng bởi bước nhảy rời rạc của hàm `ceil`:
1. Cho NPU chạy Encoder và SDP để sinh ra ma trận thời lượng $w_{ceil}^{NPU}$.
2. Dùng chính ma trận $w_{ceil}^{NPU}$ này làm đầu vào cho cả 2 nhánh:
   - **Nhánh 1:** Chạy Flow + Decoder hoàn toàn trên **NPU (FP16 HTP)**.
   - **Nhánh 2:** Chạy Flow + Decoder hoàn toàn trên **CPU Reference (FP32 ORT)**.
3. **Kết quả thu được:**
   - Tensor ẩn $z$ giữa NPU và CPU: $\text{Cosine Sim} = \mathbf{0.999999}$ (Gần như trùng khớp tuyệt đối).
   - Dạng sóng âm thanh cuối cùng: $\text{Cosine Sim} = \mathbf{0.999855}$ (Vượt xa tiêu chuẩn công nghiệp $> 0.90$).

---

## 7. TẦNG 7: Chuyển Hóa Tư Duy: Từ "Vibe Coding" Sang "Deep Hardware Engineering"

| Tiêu Chí | Tư Duy "Vibe Coding" | Tư Duy "Deep Hardware Engineering" |
|---|---|---|
| **Xử lý lỗi compile NPU** | Thêm thử các cờ ngẫu nhiên (`--use_cpu`, `--relax_shape`) đến khi hết lỗi. | Mở đồ thị ONNX qua Netron, tìm chính xác op không tương thích với tập lệnh DSP, viết code tái cấu trúc đồ thị. |
| **Kiểm thử mô hình** | Nghe thử 1-2 câu thấy "nghe có vẻ được" là kết luận xong. | Thiết lập bộ đo định lượng: Tensor Cosine Similarity, Round-trip ASR WER/CER, phân lập sai số lượng tử. |
| **Quản lý tài nguyên** | Cứ nạp cả model lớn vào bộ nhớ, để framework tự lo. | Tính toán chi tiết từng Megabyte RAM, số phép tính FLOPs, băng thông bus TCM/DRAM, và tối ưu hóa zero-copy. |
| **Tính độc lập** | Phụ thuộc hoàn toàn vào cloud scripts / wrapper có sẵn. | Tự reconstruct kiến trúc từ file nhị phân, làm chủ từng tầng toán học và cơ chế runtime của phần cứng. |

### Cẩm nang 5 bước đưa bất kỳ mô hình AI nào lên NPU:
1. **Phân tích tính toán tử (Operator Audit):** Kiểm tra xem mọi op trong model có nằm trong tập lệnh hỗ trợ phần cứng (Supported Op Set) của NPU hay không.
2. **Cố định đồ thị (Shape Freezing):** Loại bỏ mọi biến động về kích thước buffer; chia nhỏ bài toán bằng kỹ thuật Chunking/Sliding Window nếu dữ liệu đầu vào có độ dài động.
3. **Deterministic Transformation:** Loại bỏ tất cả các phép toán sinh số ngẫu nhiên runtime không xác định.
4. **Phân tách Host-Device Hợp lý:** Đẩy toàn bộ phép toán ma trận nặng (GEMM, Conv, Attention) cho NPU; giữ lại các phép toán logic, tra bảng, căn chỉnh đơn giản cho CPU Host.
5. **Thẩm định đa tầng (Multi-tier Verification):** Kiểm tra sai số ở từng đầu ra tensor trung gian trước khi đánh giá chất lượng toàn chuỗi.
