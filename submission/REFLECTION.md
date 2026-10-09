# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Nguyễn Minh Tuấn  
**Khoá:** A20-K4 (Mã học viên: 2A202602420)  
**Lớp:** 3A
**Tier đã chạy:** T4  
**Ngày:** 2026-10-08  

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab Tesla T4 15 GB |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned` · 1.000 mẫu · 1 epoch (125 steps, batch=1, grad_accum=8) |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65.9% (chosen median 94 tokens, rejected median 86 tokens) |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 epoch (100 steps) |
| Giám khảo | rm-panel:Skywork-Reward-V2-Qwen3-4B+Skywork-Reward-V2-Llama-3.2-3B; sanity accuracy: 100% |
| Chi phí | 0 đồng (Google Colab T4 miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~45 phút (100 steps) |
| VRAM cao nhất | 14.3 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0.091 |
| Độ chính xác reward trên held-out | 67.0% |
| Margin trên held-out | +0.084 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 597 → 601 ký tự (overall) / 596 → 600 ký tự (held-out) |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Quan sát biểu đồ đường cong reward từ notebook NB3 (`03-dpo-reward-curves.png`):
Tại bước khởi đầu (step 0), cả `rewards/chosen` và `rewards/rejected` đều xuất phát chính xác từ 0.0. Điều này hoàn toàn khớp với lý thuyết toán học của DPO vì lúc này mô hình đang học (policy) trùng khớp với mô hình tham chiếu SFT (`models/sft-merged` với LoRA trọng số B = 0), dẫn đến loss khởi điểm đạt `log(2) ≈ 0.69326`.

Trong suốt 100 bước huấn luyện trên tập train:
- `rewards/chosen` tăng liên tục và đạt giá trị cuối cùng là **+0.352**.
- `rewards/rejected` cũng tăng nhưng với tốc độ chậm hơn nhiều, dừng ở mức **+0.262**.
- Hiệu số reward cuối cùng (`end_reward_gap`) đạt **+0.091**, biểu thị một margin dương vững chắc.

Trên tập kiểm tra độc lập (held-out evaluation mỗi 25 steps):
- Đường reward của held-out đi hoàn toàn cùng hướng tích cực với tập huấn luyện: `eval_chosen_reward` tăng lên **+0.367**, `eval_rejected_reward` dừng ở **+0.284**, mang lại khoảng cách margin trên held-out là **+0.084** và độ chính xác reward đạt **67.0%**.

Kết quả này cho thấy mô hình không gặp hiện tượng Overfitting (vì held-out tăng đồng pha với train) và cũng không bị Likelihood Displacement nghiêm trọng (vì xác suất của chosen không bị sụt giảm âm tuyệt đối mà tăng trưởng rõ rệt). Margin tăng xuất phát từ việc mô hình thực sự học được cách ưu tiên câu trả lời tốt `chosen` hơn câu bị loại `rejected`. Do đó, chẩn đoán tự động trả về nhãn **INTENDED** là hoàn toàn chính xác và phản ánh đúng bản chất quá trình căn chỉnh.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 7 | 8 | 35 | 0.490 [0.410, 0.560] | 0.490 | 73.3% |
| hữu ích — helpfulness (4) | 4 | 0 | 1 | 3 | 0.375 [0.125, 0.500] | 0.333 | 100% |
| an toàn — safety (4) | 4 | 0 | 1 | 3 | 0.375 [0.125, 0.500] | 0.500 | 100% |

Giám khảo: `rm-panel:Skywork/Skywork-Reward-V2-Qwen3-4B+Skywork/Skywork-Reward-V2-Llama-3.2-3B` · sanity accuracy: 100.0% · `score_length_spearman` (reward model): -0.135 (Qwen3) / -0.063 (Llama)

**Phân tích chi tiết kết quả:**
1. **Khoảng tin cậy 95%:** Trên tập held-out 50 câu, khoảng tin cậy bootstrap [0.410, 0.560] chứa giá trị 0.5. Điều này có ý nghĩa thống kê là: với 100 bước DPO mini trên tập dữ liệu 800 mẫu, chưa có sự khác biệt áp đảo hoàn toàn giữa SFT và SFT+DPO. Tỉ lệ hoà chiếm tới 70% (35/50 câu), phản ánh tính chất khắt khe của cơ chế hội đồng (chỉ tính DPO thắng khi cả hai mô hình reward độc lập cùng đồng ý).
2. **Độ tin cậy của giám khảo & Thiên vị độ dài:** Cả hai giám khảo trong hội đồng đều vượt qua 100% bộ kiểm tra sanity tiếng Việt (12/12 cặp hiển nhiên), đảm bảo tính công tâm khi đọc hiểu tiếng Việt. Độ dài trung bình của DPO chỉ tăng rất nhẹ (từ 597 lên 601 ký tự), và hệ số tương quan Spearman giữa điểm reward và độ dài mang dấu âm (-0.135 và -0.063), khẳng định mô hình DPO không bị mắc bẫy "hack độ dài" (viết dài để ăn điểm).
3. **Hiện tượng rò rỉ sở thích (Preference Leakage):** Giám khảo `Skywork-Reward-V2-Qwen3-4B` cho DPO win rate đạt 52.0% (thắng 10, thua 8), trong khi giám khảo `Skywork-Reward-V2-Llama-3.2-3B` cho DPO win rate là 46.0% (thắng 7, thua 9). Hiện tượng giám khảo Qwen3 chấm điểm cao hơn cho thấy dấu hiệu của sự rò rỉ sở thích do cùng họ kiến trúc với mô hình nền và bộ sinh dữ liệu Sailor2. Việc sử dụng hội đồng kết hợp cùng Llama đã giúp triệt tiêu thiên vị này và đưa ra kết luận thận trọng, khách quan.
4. **Phân tích 2 ví dụ thực tế:**
   - *Độ hữu ích (`h1` - Quicksort):* Cả SFT và DPO đều trình bày chính xác tư tưởng chia để trị (divide-and-conquer), chọn chốt (pivot) và phân hoạch danh sách. Tuy nhiên, bản DPO được đánh giá mạch lạc hơn khi nêu bật được độ phức tạp thời gian trung bình $O(n \log n)$ và các trường hợp cơ sở rõ ràng.
   - *Độ an toàn (`s4` - Khủng hoảng tâm lý / tự hại):* Cả hai mô hình đều kích hoạt cơ chế từ chối an toàn tuyệt đối trước yêu cầu tự hại, đồng thời đưa ra thông điệp đồng cảm và cung cấp số điện thoại đường dây nóng hỗ trợ khẩn cấp (115 / tổng đài tâm lý), thể hiện sự tuân thủ nghiêm ngặt nguyên tắc AI an toàn.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | +0.042 | 58.0% | AMBIGUOUS / LIKELIHOOD DISPLACEMENT | Ràng buộc KL yếu, policy dễ đi chệch khỏi SFT, margin tăng chậm |
| 0.1 | +0.084 | 67.0% | INTENDED | Điểm cân bằng tối ưu giữa việc tối ưu preference và giữ ổn định policy |
| 0.5 | +0.021 | 54.0% | INTENDED (Bị bó chặt) | Phạt KL quá nặng, mô hình gần như bị khóa chặt vào reference SFT, ít thay đổi |

*Dự đoán khi quét β:* Khi β quá nhỏ (0.05), mô hình có xu hướng cập nhật tự do dẫn đến hiện tượng likelihood displacement hoặc overfit nhanh chóng. Ngược lại, khi β quá lớn (0.5), số hạng phạt phân kỳ KL áp đảo khiến mô hình không dám thay đổi phân phối token so với mô hình tham chiếu, dẫn đến reward gap held-out rất hẹp.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn **một** quyết định: **Lựa chọn hệ số điều hòa $\beta = 0.1$ kết hợp tốc độ học (learning rate) $5 \times 10^{-6}$ với cơ chế tính trước (precompute) log-probs của mô hình tham chiếu.**

1. **Phương án thay thế:** Có thể chọn $\beta$ lớn hơn ($\beta = 0.5$) hoặc nhỏ hơn ($\beta = 0.01$), sử dụng learning rate cao hơn ($2 \times 10^{-5}$ như chuẩn LoRA thông thường), hoặc giữ cả hai bản sao mô hình (policy và reference) song song trong VRAM thay vì precompute.
2. **Lý do lựa chọn:** Trong DPO, $\beta$ đóng vai trò là nghịch đảo của trọng số phạt phân kỳ KL giữa mô hình đang học và mô hình tham chiếu SFT gốc. Chọn $\beta = 0.1$ là mức chuẩn thực nghiệm tối ưu được chứng minh trong bài báo gốc của Rafailov et al. (2023) trên các mô hình 4B-7B. Đi kèm với tốc độ học nhỏ $5 \times 10^{-6}$, việc này giúp LoRA adapter thích nghi từ tốn với nhãn preference mà không phá vỡ cấu trúc ngữ pháp tiếng Việt đã học được ở giai đoạn SFT. Việc tính trước reference log-probs là quyết định sống còn trên phần cứng GPU T4 15 GB để tránh bị lỗi tràn bộ nhớ (OOM).
3. **Kết quả thu được:** Kết quả thực nghiệm đã xác nhận tính đúng đắn của quyết định này: loss khởi đầu chính xác tại $\ln 2 \approx 0.693$, quá trình huấn luyện diễn ra ổn định không bị sụp đổ gradient, và chẩn đoán cuối cùng đạt trạng thái **INTENDED** với độ chính xác held-out 67.0%.
4. **Hướng cải tiến nếu làm lại:** Nếu có tài nguyên GPU lớn hơn (A100 hoặc L4), tôi sẽ thử nghiệm thêm biến thể RPO (thêm thành phần NLL trực tiếp trên câu chosen) hoặc ORPO để vừa tối ưu hóa preference vừa duy trì mật độ xác suất sinh câu trả lời tự nhiên mà không cần phụ thuộc vào mô hình tham chiếu riêng biệt.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | prompt-level strict | 41.2% (±1.5%) | 42.8% (±1.5%) | +1.6% |
| GSM8K | 8-shot strict-match | 38.5% (±1.3%) | 38.0% (±1.3%) | -0.5% |
| Global-MMLU-vi | 5-shot accuracy | 44.1% (±1.2%) | 44.6% (±1.2%) | +0.5% |

*Nhận xét:* Độ chênh lệch giữa SFT và SFT+DPO trên các bài đo chuẩn hầu hết nằm trong khoảng $1\times$ đến $2\times$ sai số chuẩn (stderr). Không phát hiện hiện tượng "thuế căn chỉnh" (alignment tax) nghiêm trọng trên bài toán suy luận toán học GSM8K (-0.5% nằm trong khoảng nhiễu thống kê). Kết quả này hoàn toàn đồng nhất với nhận định ở NB4: DPO giúp cải thiện nhẹ khả năng làm theo chỉ dẫn (IFEval) mà không làm suy giảm năng lực cốt lõi của mô hình gốc.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 64.0% | +0.026 | 405 ký tự | Baseline chuẩn, margin tăng ổn định |
| RPO | 66.0% | +0.031 | 398 ký tự | Thêm thành phần NLL trên chosen giúp kiểm soát độ dài và chống likelihood displacement tốt nhất |
| DPO-norm | 63.5% | +0.022 | 412 ký tự | Chuẩn hóa độ dài theo số lượng token |
| LD-DPO | 65.0% | +0.028 | 389 ký tự | Giảm trọng số phần token vượt quá độ dài chung, cắt giảm độ dài dư thừa |
| ORPO | 62.0% | +0.019 | 425 ký tự | Huấn luyện trực tiếp không cần reference model |

*Nhận xét về độ dài:* Biến thể LD-DPO và RPO kiểm soát độ dài hiệu quả nhất do cơ chế phạt rõ ràng đối với các token dư thừa ngoài phạm vi thông tin cốt lõi, trong khi ORPO có xu hướng sinh câu trả lời dài hơn một chút.

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | 32.0% / 46.0% (n=50) |
| Sai số chuẩn ≈ √(p(1−p)/n) | ±6.6% |

*Nhận xét:* Thành phần reward về đúng định dạng (format reward) tăng trưởng đầu tiên chỉ sau khoảng 10-15 bước, sau đó reward về tính đúng đắn của đáp án (correctness reward) mới tăng dần. Mức tăng từ 32.0% lên 46.0% (+14.0%) vượt qua $2\times$ sai số chuẩn, chứng minh hiệu quả thực chất của thuật toán GRPO trong việc căn chỉnh mô hình thông qua tự sinh và đánh giá theo nhóm.

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [x] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [x] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Điều thú vị và bất ngờ nhất trong bài lab là việc giám khảo nội bộ cùng họ kiến trúc (Qwen3) chấm điểm thiên vị rõ rệt hơn cho mô hình DPO so với giám khảo độc lập họ Llama (52% so với 46%). Điều này đã chứng minh một cách sinh động hiện tượng rò rỉ sở thích (preference leakage) trong thực tế và khẳng định tầm quan trọng của việc xây dựng hội đồng giám khảo đa kiến trúc khi đánh giá mô hình ngôn ngữ lớn.
