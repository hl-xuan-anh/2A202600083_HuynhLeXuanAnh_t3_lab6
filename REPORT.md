# Lab 21 — Evaluation Report

**Học viên**: Huỳnh Lê Xuân Ánh — 2A202600083  
**Ngày nộp**: 2026-05-07  
**Submission option**: B (HF Hub)

## 1. Setup
- **Base model**: `unsloth/Llama-3.2-3B-Instruct`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 200 samples (180 train + 20 eval, seed=42)
- **max_seq_length**: 1024 (p95 = 536, rounded up + cap 1024)
- **GPU**: Tesla T4, 16 GB VRAM
- **Training cost**: ~$0.07 (~12.12 phút @ $0.35/hr)
- **HF Hub link** (Option B): https://huggingface.co/HLXA/llama-3.2-3b-vi-lab21-r16

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-----------------|------------|-----------|-----------|------------|
| 8    | 2,293,760       | 3.99 min   | 11.11 GB  | 1.6637    | 5.2786     |
| 16   | 4,587,520       | 4.03 min   | 10.34 GB  | 1.6491    | 5.2023     |
| 64   | 18,350,080      | 4.09 min   | 12.09 GB  | 1.6481    | 5.1973     |

## 3. Loss Curve Analysis
![Training loss curve (r=16)](loss_curve.png)

- Quan sát: đường training loss giảm rõ rệt từ khoảng **~1.61** xuống **~1.38–1.40** theo các step, cho thấy mô hình học được pattern từ dataset.
- Có một vài nhịp “nhấp nhô” giữa chừng (loss giảm rồi tăng nhẹ rồi lại giảm), nhưng biên độ dao động nhỏ và không có dấu hiệu diverge/“loss bùng lên”.
- Ở đoạn cuối, loss tăng nhẹ quanh **~1.41–1.42** rồi giảm trở lại **~1.39**, gợi ý quá trình tối ưu đã gần hội tụ; mức dao động cuối training có thể do batch nhỏ (T4 batch=1) và dữ liệu ít (200 samples).
- Vì không có đường **eval loss theo step** trong hình, chưa thể kết luận chắc chắn overfitting chỉ từ training curve; tuy nhiên shape “giảm dần và ổn định” là tín hiệu training tương đối healthy.

## 4. Qualitative Comparison (5 examples)

### Example 1
**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.  
**Base**: Machine learning là một phương pháp xử lý dữ liệu tự động… (trả lời khá chung chung).  
**Fine-tuned (r=16)**: Machine learning là một phần của trí tuệ nhân tạo… (mạch lạc hơn, nêu được vài ứng dụng).  
**Nhận xét**: improved (flow tốt hơn, ít lặp ý).

### Example 2
**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.  
**Base**: Đưa ví dụ “Fibonacci thứ 10 là 55” + nhận xét độ phức tạp, nhưng không đưa code hoàn chỉnh.  
**Fine-tuned (r=16)**: Có đưa khung code Python (dù phần formatting/cú pháp còn bị cắt ngắn trong log).  
**Nhận xét**: improved (đúng hướng hơn, “actionable” hơn).

### Example 3
**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.  
**Base**: Bắt đầu liệt kê được 1 nguyên tắc, nội dung dài và dễ lan man.  
**Fine-tuned (r=16)**: Trả lời theo dạng list rõ hơn, nhưng vẫn có hiện tượng lặp/cụt ý do giới hạn max tokens khi log.  
**Nhận xét**: slightly improved (format tốt hơn, nội dung vẫn chung chung).

### Example 4
**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.  
**Base**: Có nhắc LoRA/QLoRA trong NLP nhưng lẫn thông tin không chuẩn (ví dụ mốc/nguồn).  
**Fine-tuned (r=16)**: Trả lời sai trọng tâm (diễn giải nhầm thuật ngữ), chất lượng giảm.  
**Nhận xét**: degraded (cần dataset domain/labeling tốt hơn cho kiến thức kỹ thuật).

### Example 5
**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.  
**Base**: Trả lời tương đối đúng định nghĩa Prompt Engineering, thiếu cấu trúc cho 2 phần còn lại.  
**Fine-tuned (r=16)**: Có chia bullet nhưng xuất hiện nội dung sai/lệch (ví dụ diễn giải RAG không chuẩn).  
**Nhận xét**: same → degraded (format tốt hơn nhưng factuality chưa tốt).

## 5. Conclusion về Rank Trade-off

Trên dataset 200 mẫu (180/20) này, **r=16** cho ROI tốt nhất. Lý do: so với r=8, r=16 cải thiện perplexity rõ rệt (5.2786 → 5.2023) trong khi thời gian train gần như tương đương (~4 phút) và peak VRAM thậm chí thấp hơn trong run này (10.34 GB vs 11.11 GB). Khi tăng lên **r=64**, perplexity chỉ cải thiện rất nhỏ (5.2023 → 5.1973) dù trainable params tăng mạnh (4.6M → 18.35M) và VRAM peak tăng (12.09 GB). Đây là dấu hiệu **diminishing returns**: từ khoảng **r=16 trở lên**, chất lượng (perplexity) cải thiện không đáng kể so với chi phí tham số/VRAM. Nếu deploy production (đặc biệt trong bối cảnh T4/edge GPU), mình chọn **r=16** vì cân bằng tốt giữa chất lượng và tài nguyên; r=64 chỉ đáng cân nhắc khi có dataset lớn hơn/khó hơn và cần tối đa hóa metric, hoặc khi GPU/VRAM không phải là ràng buộc.

## 6. What I Learned
- Rank lớn không đồng nghĩa “tốt hơn” rõ rệt: cần nhìn vào diminishing returns qua perplexity và VRAM/time.
- Với GPU nhỏ (T4), thiết kế pipeline (batch=1, tắt eval-steps, safe eval) quan trọng không kém hyperparams.
- Qualitative eval có thể reveal lỗi factuality mà perplexity không phản ánh; dataset chất lượng/đúng domain là yếu tố quyết định.
