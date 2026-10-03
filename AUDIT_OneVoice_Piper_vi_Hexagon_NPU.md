# BÁO CÁO AUDIT — OneVoice / Piper vi trên Hexagon NPU

## 0. Phán quyết thẳng

Bạn đã làm được phần khó nhất và nó là thật: 4 graph neural (encoder / sdp / flow / decoder) đã compile thành `qnn_context_binary` chạy trên IQ-9075, và 7 stage có job inference thật trên device (enc / sdp / align / flow / dec / ola / resample). Chất lượng số học tốt: cos(NPU, fp32) = 0.999978, flow-z cos = 0.9999997.

Nhưng **"100% NPU" chưa đạt**, và có 3 lỗi chặn cứng khiến hiện tại pipeline đang chạy **sai** chứ không phải chỉ "còn CPU":

| # | Vấn đề | Mức |
|---|--------|-----|
| P0-1 | Không tồn tại runtime NPU nào trong repo — mọi `InferenceSession` là CPU EP | CHẶN |
| P0-2 | Tokenizer đang là map ký tự tự chế, mất toàn bộ dấu tiếng Việt | CHẶN |
| P0-3 | Chain NPU đang được nuôi bằng output của tokenizer hỏng đó → audio ngắn 1.8–7.8× và vẫn PASS metric | CHẶN |

---

## P0-1. Không có NPU runtime: mọi session là CPU

| File:line | Nội dung | Bằng chứng |
|-----------|----------|------------|
| `zero_cpu_driver.py:73-81` | 9× `ort.InferenceSession(path, opts)` — không có `providers=` | ORT mặc định `CPUExecutionProvider` |
| `zero_cpu_driver.py:70` | `opts.intra_op_num_threads = 4` | knob của CPU |
| `zero_cpu_driver.py:327` | tự log `"RTF: %.4f (ONNX CPU Emulator)"` | script tự nhận là emulator CPU |
| `evaluate_e2e_pipeline.py:74-80` | 7 session, không `providers=` | CPU |
| `piper_components_pipeline.py:102-117` | 8 session ghi thẳng `providers=["CPUExecutionProvider"]` | CPU |
| `zero_cpu_driver.py:110-229` | 19× `bind_cpu_input(...)`, 12× `.numpy()` | "zero-copy device chaining" không tồn tại |

Grep toàn repo: không có `QNNExecutionProvider`, không `OrtValue` device, không allocator ở bất kỳ đâu. Các file `.iq9075.bin` / `.dlc` chỉ được upload lên AI Hub job, **không driver nào load chúng**. Vậy nên "Zero-CPU / 0.0% phụ thuộc CPU" trong `DISCUSSIONS.md:1,5,251,392` là **sai**.

---

## P0-2. Tokenizer giả — lỗi nặng nhất cho phần Piper

`piper_components_pipeline.py:44-58` (`phonemize_vi`) duyệt từng ký tự của text và tra vào `phoneme_id_map` (map này keyed bằng phoneme IPA):

```python
for ch in text.lower():
    if ch in id_map: ...
    elif ch in (" ", "\t", "\n"): ...
    # KHÔNG có else → ký tự lạ bị bỏ im lặng
```

Đo thực tế:

| Input | `phonemize_vi` trả về | Piper thật (`piper_vi_npu_data.npz`) |
|-------|----------------------|--------------------------------------|
| "Nông nghiệp tự cung tự tiêu" | `'nn nhip t cun t tiu'` | `'^nˈoŋ ŋˈiɛ6p t̪ˈy6 kˈuŋ '` |
| "đường" | `'n'` | có ɗ, ɨ, ə + thanh điệu |
| "ăâêôơưđáàảãạếệ" | `'_'` (1 id) | — |
| "Chuẩn" | `'chun'` | tɕ + thanh hỏi |

`ă â đ ê ô ơ ư` và mọi dấu thanh **không tồn tại** trong `phoneme_id_map` (154 entry, ASCII) → bị xóa sạch. Không có espeak, không BOS `^`, không EOS `$`, không separator `0` (Piper chèn `0` giữa các phoneme — dữ liệu thật `[1,0,26,0,120,0,...]`, tokenizer giả cho `[26,26,3,26,20,...]`).

Đây là thay đổi **chưa commit** (`git diff` xác nhận: `phonemize_vi` / `prepare_input` được thêm vào working tree của `piper_components_pipeline.py`).

Bằng chứng nó đã lan vào đường chạy thật: `phonemize_vi(test_texts[0])` → 164 ids, và `calib/test_encoder.npz` có `x_lengths[0] = 164`. Khớp chính xác.

---

## P0-3. Chain NPU đang chạy bằng token giả → duration sụp, metric vẫn PASS

Chuỗi nhân quả đã kiểm chứng:

1. `generate_component_calibration.py:78` dùng `phonemize_vi` → regenerate `calib/*.npz` lúc 2026-10-03 10:25.
2. `run_hw_pipeline.py:181` — `stage_enc` đọc `calib/test_encoder.npz`, **không phải** dữ liệu thật `piper_vi_npu_data.npz`.
3. Nên toàn bộ hardware chain nhận token giả: `x_lengths=[164,85,50,169]` thay vì `[608,325,349,249,219,279,273,417]`.

Kết quả: cùng câu 240 ký tự, cùng model, cùng weights:

| Run | Tokens vào | y_lengths | Audio |
|-----|-----------|-----------|-------|
| 09-22 (báo cáo) | 608 (Piper thật) | 797 | 9.253 s ✓ |
| 10-03 (hiện tại) | 164 (tokenizer giả) | 164 | 1.904 s ✗ |

Tôi đã tự chạy lại SDP host trên chính hw/enc output: với mapping hiện tại cho đúng `[164,88,65,181]` khớp NPU; với encoder-output đúng nghĩa cho `[164,86,55,173]` (lệch) → xác nhận input đã đổi, không phải SDP hỏng.

Câu "Chuẩn 802.11n…" = 28 âm tiết, audio 0.755 s (7.8× quá nhanh). Không TTS nào đọc được.

**Metric vẫn PASS vì** `verify_hw_pipeline.py:105-115` so sánh NPU-vs-fp32 với **cùng alignment của NPU** và cắt cả hai về cùng `yl` → bất biến với duration sai. Đó là lý do `similarity_results.json` ghi `mean_audio_cos 0.999978`, `n_pass 4/4` cho audio hỏng. Reference còn là host CPU (`providers=["CPUExecutionProvider"]`) → repo không có phép so sánh NPU-vs-NPU nào.

---

## P1. Các tuyên bố "100% NPU" không đứng vững

| Claim | Thực tế (đo trên artifact) |
|-------|---------------------------|
| Byte Encoder là giải pháp thay G2P trên NPU (`DISCUSSIONS.md:136,251`) | **Chưa từng chạy trên NPU**: không có `byte_enc` trong `job_log_hw.json`, không có `hw/byte_enc/`; `stage_all` (`:452-458`) không gọi `stage_byte_enc`. Trọng số random (`byte_emb.weight` std=0.9981 vs encoder đã train 0.0454). Và `stage_byte_enc:139-146` đang feed phoneme IDs vào `byte_indices` (file bytes thật `test_byte_text_encoder.npz` bỏ không dùng). |
| `streaming_slicer.onnx` = "DMA ring-buffer slicer 100% NPU" (`DISCUSSIONS.md:184,256`) | 1 node Identity, 168 byte, I/O `[1,192,64]→[1,192,64]`. Slicing thật là numpy host (`run_hw_pipeline.py:325-334`). Không có compile job, không inference job. |
| `overlap_add` khử clicks/pops trên NPU | `run_hw_pipeline.py:364` đặt `prev_tail = np.zeros(...)` cho mọi chunk → mỗi chunk ≥1 bị fade từ im lặng 13.9 ms; `next_tail` bị vứt. Audio được chấm điểm còn không dùng OLA/resampler NPU (host `np.concatenate` + `np.interp` ở `verify_hw_pipeline.py:110-113`, `evaluate_hw_pipeline.py:46-50`). |
| "SDP toàn op tĩnh, 0 rẽ nhánh" (`DISCUSSIONS.md:377`) | `piper_vi_sdp.onnx` có 36 Range + 30 ScatterND + 21 NonZero + 15 GatherND — đúng các op mà `DISCUSSIONS.md:116` nói HTP không compile được. |
| "ép toàn bộ Conv1d → Conv2d H=1" | Conv1d vẫn còn: encoder 37, sdp 32, flow 40, decoder 20 Conv + 3 ConvTranspose (đều rank 3). Chỉ `byte_text_encoder` có Conv2d H=1. |
| "không CPU fallback" (`REPORT_PIPER_IQ9075.md:13`) | **Chưa được chứng minh**: không có `submit_profile_job` nào trong `src/`, không có log job/placement trên đĩa. |

### Về 102 op "khó" trong SDP — đã trace ngược từng op

Kết quả backward-reachability (script `outputs/_audit/sdp_datadep_audit.py`):

```
GatherND  x15  DATA-DEPENDENT      Range     x36  DATA-DEPENDENT
NonZero   x21  DATA-DEPENDENT      ScatterND x30  DATA-DEPENDENT
TOTAL data-dependent: 102       TOTAL constant-foldable: 0
```

102/102 đều trace về graph input (`x_encoded`, `x_mask`, `noise_scale_w`), không fold được → buộc phải thực thi lúc inference (chúng nằm trong `/flows.7/` = spline searchsorted của ConvFlow). Vì compile vẫn SUCCESS, khả năng cao QAIRT decompose chúng thành primitive HTP — nhưng đó là **suy luận, chưa có bằng chứng**. Muốn khẳng định 100% HTP thì bắt buộc phải chạy profile job. Đây là việc rẻ nhất và nên làm đầu tiên.

Thêm: `patch_piper_npu.py` (viết để sửa chính các op này) **không hề được áp dụng** — nó patch graph monolithic (`input_name = "input"`, `:92`) ra `piper_vi_npufix.onnx` (63 MB, không ai dùng), còn 4 component deploy thì export thẳng từ `export_piper_components.py`.

---

## P2. Kết quả/báo cáo không tái lập được

- `hw/` trộn 2 run khác nhau: `hw_0..3.wav` (10-03: 1.904 / 1.022 / 0.755 / 2.101 s) + `hw_4..7.wav` (09-28: 3.529 / 4.714 / 5.027 / 7.570 s). `load_hw_outputs` (`:414`) chỉ lấy h5 mới nhất mỗi stage → ghép run âm thầm, không kiểm tra số mẫu.
- `similarity_results.json` (10-03, n=4, mean 0.999978) vs `evaluation_results.json` (09-22, n=8, WER 0.3425) vs `REPORT_PIPER_IQ9075.md:128-139` (bảng 8 dòng, duration 9.253…7.57 s). Bảng trong báo cáo không khớp artifact nào.
- Sai số liệu: report ghi `sdp.bin` = 1,880,064 B / `decoder.bin` = 3,363,840 B; đĩa thật 1,974,272 / 3,526,656. `REPORT_PIPER_STATUS…` ghi flow ONNX 29.8 MB; đĩa thật 74.45 MB.
- `generate_report.py:60,159-163` hardcode row encoder (job `jgnn0rejg`) trong khi `job_log_components.json` ghi job đó chỉ "submitted", không có SUCCESS.
- Node counts trong `DISCUSSIONS.md` sai 5/9 dòng: encoder 3,904→2,596; aligner 12→22; flow 1,135→752; decoder 116→70; ola 18→35.
- Compile option lệch: 4 component lõi build bằng `qnn_context_binary`, nhưng `deploy_piper_components.py:85-92` hiện dùng `qnn_dlc` → chạy lại hôm nay sẽ không tái lập được `.bin` cũ (và sẽ trộn `.dlc`/`.bin` trong cùng pipeline).
- Dead artifact: `piper_vi_npufix.onnx` 63 MB (không dùng), `calib/test_*.npz` ~20 MB (không stage nào đọc), `piper_vi_monotonic_aligner.iq9075.bin.dlc` (double extension), không có file `*w8a16*` nào → nhánh quantize chưa từng chạy cho components.

---

## P3. Lỗi đúng đắn (correctness) — đã đo bằng số

| # | Lỗi | Bằng chứng đo được |
|---|-----|--------------------|
| C15 | **Trần 1536 frame cắt im lặng.** Test: 512 token × 4 frame = 2048 → aligner chỉ giữ 1536, token cuối cùng được 0 frame → mất hẳn audio của nó. Trần audio = 1536×256/22050 = 17.83 s, và chỉ cho 3 frame/token ở input dài nhất. | `attn_sum=1536` (mong đợi 2048), last token frames kept = 0.0 |
| C16 | **MAX_SEQ_LEN 512 vs fixed_length 608.** `test_input[0]` rộng 608, token thật tới index 606 → cắt âm thầm 96 slot (15.8%). `min(phoneme_length, 512)` ở `zero_cpu_driver.py:102`, `prepare_input:64` — không warning. | `data_stats.json`: fixed_length 608; encoder ONNX input `x: [1,512]` |
| C17 | **Vòng lặp cửa sổ decoder bỏ frame cuối.** Guard `min(yl, z.shape[2]-40-12)` = `min(yl,1484)` → `y_lengths=1536` chỉ emit 1520 frame, mất 16 frame (0.186 s). | Mô phỏng vòng lặp: MISSING=16 |
| C18 | **`ENC_OUT` là quả mìn.** Thứ tự ONNX thật = `[x_encoded, m_p, logs_p, x_mask]`; thứ tự AI Hub h5 = `[m_p, logs_p, x_encoded, x_mask]` (h5 có attr `name='output_N'`: group0=output_1, group1=output_2, group2=output_0). `load_hw_outputs` bỏ qua attr `name`, sort theo key số → `ENC_OUT` hiện "đúng" do ăn may. 3 tensor cùng shape `[1,192,512]` → recompile là swap m_p/logs_p/x_encoded không báo lỗi. | `hw/enc/*.h5` attrs; ORT run cho cos(x_encoded)=1.000000 ở index 0 |
| C19 | **Calibration của OLA sai hợp đồng graph:** `prev_tail` toàn zeros, `is_first=[1,0,0,…]` → nếu ai chạy `--step quantize` thì calibration không bao giờ thấy tail thật. | `generate_component_calibration.py:111-131` |
| C20 | **`y_lengths` dtype:** bản HEAD là float32, aligner ONNX khai báo int32 → ORT raise `Unexpected input data type`. Working tree đã sửa (uncommitted). | diff `run_hw_pipeline.py:245` |

---

## ĐÃ KIỂM TRA VÀ OK (đừng mất thời gian lo những cái này)

- **Weight mapping Piper ĐÚNG:** chạy thật → `Loaded 349 / 620 parameters` + `All inference-critical weights loaded!`. Các key unmapped đều nằm trong nhánh không dùng ở inference: `enc_q.*` (posterior encoder, chỉ train), `dp.post_*` (nhánh train theo w thật — `models.py:77-82`), `dp.flows.1` (bị `SDP.forward:236-237` bỏ có chủ đích). `dp.proj` (dùng thật ở `:231`) có weight thật. → Không có thảm họa random-weight.
- **`overlap_add.onnx` nội bộ đúng:** `w_in = sin²`, `w_out = cos²`, power-complementary; slice ranges khớp timeline stride-40 (đã trích xuất toàn bộ hằng số slice: `[0:10240]`, `[10240:13312]`, `[3072:6144]`, `[6144:13312]`, `[13312:16384]` — tất cả khớp). Lỗi nằm ở caller, không ở graph.
- **`monotonic_aligner.onnx` logic khớp `generate_path_np`:** sane case cho `attn_sum=117` đúng 117 frame, mỗi token ≥1 frame, không cột nào rỗng.
- `RandomNormalLike` đã bị loại sạch (monolith 2 → components 0); If/Loop/Scan = 0 ở cả 9 graph.
- Mọi input/output của 8 component static (3 output encoder khai báo `dim_param` nhưng thực tế `[1,192,512]`).
- Activation chaining NPU→NPU qua h5 là thật (enc→sdp→align→flow→dec), và wav được chấm điểm thật sự đến từ decoder trên NPU (host chỉ cắt cửa sổ / nối / int16 / write).
- `audio_resampler.onnx` là 1 MatMul tĩnh hợp lệ về mặt shape — nhưng **304 MB** (76,083,200 tham số cho ma trận `[10240,7430]`), nặng gấp 7.5× cả 4 model Piper cộng lại, và bị bỏ khỏi mọi bảng "tổng dung lượng ≈40 MB".

---

## Roadmap để thực sự 100% NPU (theo đúng thứ tự)

1. **Sửa tokenizer trước tiên** — dùng espeak thật như `prepare_piper_data.py:68-71` (`from piper.voice import phonemes_to_ids` + `EspeakPhonemizer`), làm `phonemize_vi` **raise** khi gặp ký tự không có trong map, rồi regenerate data. Thêm assert so id sequence với `piper_vi_npu_data.npz`. Không có bước này thì mọi số đo sau đó vô nghĩa.
2. Đổi `stage_enc` đọc `piper_vi_npu_data.npz` (8 mẫu thật) thay vì `calib/test_encoder.npz`.
3. **Chạy 1 profile job trên AI Hub** cho encoder/sdp/flow/decoder → có bằng chứng placement từng op. Nếu SDP rớt CPU ở 102 op spline → phải thay/rewrite spline.
4. Chạy **QNN EP thật + OrtValue device buffer**, hoặc bỏ hẳn nhánh `zero_cpu_driver` / `evaluate_e2e_pipeline` khỏi mọi tuyên bố NPU.
5. Chạy OLA/resampler/slicer trên device và nối `prev_tail = next_tail`; thay `streaming_slicer` Identity bằng graph Gather/Where tĩnh thật.
6. **Byte-level:** hoặc train `ByteLevelTextEncoder` (distill từ Piper encoder), hoặc xóa khỏi mọi tuyên bố. Hiện tại nó không giải quyết được vấn đề phoneme-CPU vì (a) chưa chạy trên device, (b) trọng số random, (c) espeak vẫn nằm trong đường chạy thật.
7. Sửa `MAX_SEQ_LEN` 512→608 rồi re-export/recompile; sửa guard vòng lặp decoder; nâng trần 1536 frame hoặc cắt câu ở host.
8. Dọn artifact (`piper_vi_npufix.onnx`, `test_*.npz`, file `.bin.dlc`), regenerate toàn bộ report từ **1 run duy nhất có run-id**.

---

## Ghi chú về quá trình audit

- **Cảnh báo:** artifact thay đổi trong lúc audit (bạn đang chạy pipeline song song — `job_log_hw.json` mtime 11:17, `similarity_results.json` 11:27). Timestamp rất quan trọng khi đọc lại báo cáo này.
- Script kiểm chứng tạo tại `OneVoice/outputs/_audit/` (6 file: `onnx_audit.py`, `livepath_audit.py`, `sdp_datadep_audit.py`, `encoder_order_test.py`, `encout_mapping_test.py`, `unmapped_weights.py`) — thư mục `outputs/` đã gitignored, xóa lúc nào cũng được.
- Các file `_audit_*.py` do sub-agent tạo ở workspace root đã bị xóa sạch; không có file nào trong `OneVoice/src/` bị sửa.
- Đã dừng 2 sub-audit (graph-surgery chi tiết, quantization) sau khi tự phủ phần lớn nội dung của chúng — nếu muốn đào sâu thêm 2 nhánh đó, có thể resume.
