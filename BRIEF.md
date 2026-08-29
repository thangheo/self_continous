# Semantic drift khi LLM viết lại nhiều lần - thí nghiệm làm gì và không làm gì

*(Bản bàn giao. Đủ để viết bài wiki mà không cần context cũ.)*

---

# PHẦN 1 - THÍ NGHIỆM NÀY LÀM GÌ, VÀ KẾT LUẬN GÌ

## Câu hỏi

Đưa một đoạn text cho LLM viết lại, rồi đưa bản vừa viết cho nó viết lại tiếp,
sáu vòng. Text có trôi xa dần khỏi bản gốc theo kiểu tam sao thất bản không?

Giả thuyết ban đầu: có, drift cộng dồn theo số vòng - một random walk trong
semantic space. **Kết quả bác giả thuyết này.**

## Cách đo

Mỗi phiên bản `xᵢ` được biến thành vector `zᵢ = E(xᵢ)`, chuẩn hoá về unit sphere,
đo bằng angular distance. Chuỗi `z₀ … z₆` là một trajectory.

Ba điều kiện prompt, khác nhau ở mức ràng buộc:

| | prompt | vai trò |
|---|---|---|
| A_natural | "viết lại cho rõ ràng hơn" | cách người ta thật sự dùng LLM |
| B_preserve | "viết lại nhưng giữ mọi claim và quan hệ logic" | ràng buộc trung gian |
| C_style_only | "chỉ sửa ngữ pháp, không đổi mệnh đề" | **baseline stylistic** |

C là bắt buộc chứ không phải phụ: mọi sentence embedding trộn style với content,
nên `drift_A − drift_C` mới xấp xỉ được phần semantic. Drift của A đứng một mình
không kết luận được gì.

Hai chế độ:
- **chained**: `xᵢ = rewrite(xᵢ₋₁)` - có tích luỹ không?
- **direct**: `xᵢ = rewrite(x₀)` - control, mỗi điểm là một bước độc lập từ gốc.

Cấu hình: 3 text, 3 seed, 6 iteration, temperature 1.0. Rewriter Claude Haiku 4.5,
judge Claude Sonnet 5 (khác model với rewriter). 324 rewrite call + 216 judge call,
tổng ~1.4 USD.

## Kết quả 1 - drift bão hoà, không tích luỹ

Fit `Rₙ ~ n^α` với `Rₙ` là angular displacement từ `x₀`:

| condition | α | R² | step autocorrelation |
|---|---|---|---|
| A_natural | 0.15 | 0.95 | **−0.280** |
| B_preserve | 0.16 | 0.86 | −0.127 |
| C_style_only | 0.22 | 0.86 | −0.024 |

Mốc: α = 0.5 là random walk thuần, α = 1.0 là ballistic. Cả ba đều xa dưới 0.5.

Step autocorrelation **âm** là con số đáng chú ý nhất: mỗi bước có xu hướng đi
ngược bước trước. Đó là chữ ký của dao động quanh một điểm cố định - vọt qua rồi
bị kéo về, chứ không phải đi lang thang tự do. Độ lớn của nó xếp hạng đúng theo
cường độ ràng buộc của prompt.

## Kết quả 2 - chained gần như không hơn direct

Angular displacement từ `x₀`, condition A:

| n | chained | direct |
|---|---|---|
| 1 | 0.145 | 0.160 |
| 3 | 0.182 | 0.143 |
| 6 | 0.191 | 0.140 |

Sáu vòng chỉ hơn một vòng khoảng **32%**. Random walk thuần phải cho √6 ≈ 2.4 lần.
Direct phẳng hoàn toàn từ n=1, và nó rơi vào gần đúng chỗ mà chained cuối cùng
dừng lại.

Gần như toàn bộ dịch chuyển nằm ở **cú nhảy đầu tiên** vào vùng phân bố của model.

## Kết quả 3 - giếng có bề rộng hữu hạn

Ensemble dispersion (khoảng cách trung bình giữa 3 seed tại mỗi iteration) tăng
nhanh ở n=1 rồi phẳng: A ≈ 0.16, B ≈ 0.105, C ≈ 0.03. Không nở ra (không phải
diffusion), không co về 0 (không hội tụ về một điểm duy nhất).

Style-corrected drift `A − C` cũng bão hoà: +0.099 tại n=1, +0.122 tại n=6.

## Kết quả 4 - cú nhảy đầu tiên gồm những gì

Độ dài so với bản gốc (chained), tỉ lệ số từ:

| condition | n=1 | n=2 | n=4 | n=6 |
|---|---|---|---|---|
| A_natural | 0.93 | 0.90 | 0.87 | 0.86 |
| B_preserve | 1.00 | 0.99 | 0.99 | 0.99 |
| C_style_only | 1.00 | 1.00 | 1.00 | 1.00 |

A cắt 7% ngay vòng đầu rồi gần như dừng. **Độ nén cũng bão hoà**, cùng hình dạng
với đường displacement.

## Kết luận

Iterative LLM rewriting **không** phải quá trình khuếch tán tích luỹ. Nó là một
cú nhảy vào vùng phân bố của model, sau đó dao động anti-persistent trong một
giếng có bề rộng hữu hạn.

Cường độ ràng buộc của prompt điều khiển **độ sâu giếng**, không điều khiển tốc độ
trôi.

Cú nhảy đầu tiên là **định dạng lại về khuôn model ưa** - ngắn hơn, giọng của nó -
chứ không phải khởi đầu của một quá trình mất mát dần.

## Một phản biện đã được kiểm và bác

Codex (OpenAI) nêu: rewrite làm text ngắn đi, nên cái gọi là drift có thể chỉ là
compression. Đã kiểm.

Gộp cả ba condition: `corr(|Δđộ dài|, displacement) = +0.845`. Trông như confound
rõ ràng.

Nhưng đó là **Simpson's paradox**. A vừa nén nhiều nhất vừa drift nhiều nhất, C
vừa không nén vừa drift ít nhất - tương quan gộp chỉ đang phản ánh khác biệt giữa
các condition, do cùng một biến thứ ba điều khiển: mức tự do prompt cho phép.

Tính lại **trong từng condition**, sau khi khử hiệu ứng text và iteration:

| condition | r riêng phần | sd độ dài trong ô |
|---|---|---|
| A_natural | **−0.156** | 0.019 |
| B_preserve | −0.440 | 0.011 (quá nhỏ, đang tương quan trên nhiễu) |
| C_style_only | +0.132 | 0.001 (không đủ dữ liệu) |

Dấu **âm** ở A: nén nhiều hơn thì drift *ít* hơn một chút - ngược chiều với điều
giả thuyết compression đòi hỏi. Thêm nữa, C giữ độ dài đúng 1.00 suốt sáu vòng mà
vẫn drift 0.069, nên có một sàn drift hoàn toàn độc lập với độ dài.

Compression không phải cơ chế. Kết luận đứng.

---

# PHẦN 2 - THÍ NGHIỆM NÀY KHÔNG LÀM GÌ, VÀ KHÔNG KẾT LUẬN GÌ

## Không đo được ý nghĩa có được giữ nguyên không

Đây là giới hạn quan trọng nhất. Nhánh đo claim preservation (rút mệnh đề nguyên tử
từ `x₀` rồi chấm từng bản viết lại) đã chạy và **hỏng**. Bằng chứng:

- Với `text_02`, judge nói bản gốc không chứa một claim nào trong 38 claim rút ra
  từ chính nó (anchor 0/38). Lỗi cơ chế, không phải sai số ngẫu nhiên.
- `A_natural` bị chấm 0% claim tại n=4, nhưng đọc tay bản n=4 thấy nó giữ gần như
  toàn bộ luận điểm của bản gốc.
- `C_style_only` đi 0.83 → 0.67 → 0.83. Claim không sống lại được.
- Taxonomy 5 nhãn sụp thành nhị phân: WEAKENED + MERGED + CONTRADICTED chỉ chiếm
  0.5% trên 7 560 nhãn.

Nguyên nhân: nhồi 26–41 claim cùng cả bài vào **một** call, bắt model xuất mảng
nhãn đúng thứ tự và đúng độ dài. Model bỏ cuộc và gán nhãn đồng loạt.

Cách sửa đúng là một call cho một claim, câu hỏi nhị phân, bỏ taxonomy 5 nhãn.
Chi phí ≈ 2 800 call, vượt ngân sách nên đã cắt khỏi phạm vi.

**Hệ quả: không được viết rằng "LLM giữ nguyên ý nghĩa qua nhiều vòng rewrite".**
Thí nghiệm này chỉ cho thấy representation không tiếp tục trôi. Trôi ít không đồng
nghĩa với giữ đúng nội dung. Một bản có thể mất một con số, một điều kiện, hoặc đảo
chiều một quan hệ nhân quả mà embedding gần như không nhúc nhích.

## Không tách được "đổi văn phong" khỏi "đổi nội dung" ở condition A

Có condition C làm sàn stylistic, nhưng A vừa đổi giọng vừa cắt độ dài vừa có thể
đổi nội dung, và pilot này không bóc riêng ba thứ đó. Cách bóc ở lần chạy sau:
thêm điều kiện "viết lại, giữ nguyên số từ ±5%".

## Không khái quát hoá được

- **3 text, 3 seed.** Bootstrap trên 3 đơn vị gần như vô nghĩa. Kết quả có thể chỉ
  phản ánh chủ đề và văn phong của đúng ba đoạn này.
- **Một rewriter duy nhất, và là model nhỏ** (Haiku 4.5). Model nhỏ có thể bão hoà
  sớm hơn, nên α **không đại diện** cho LLM nói chung.
- **Cả pipeline nằm trong một hệ sinh thái**: rewriter và judge đều là model
  Anthropic. Đây là thiếu độc lập về phương pháp, không phải kết quả bị thao túng,
  nhưng phải nêu. Cách chữa là chạy lại với rewriter của hãng khác.

## Không tái lập được từng trajectory

API không nhận seed. `temperature=1.0` cho ba mẫu độc lập từ phân phối sinh của
model - đủ tốt để ước lượng variance, nhưng chạy lại sẽ không ra đúng chuỗi text cũ.

## Không phát hiện ra hiện tượng drift

Chuyện rewrite lặp gây drift **đã được công bố**. Đừng viết như phát hiện mới:

- **REISD** (2025): LLM viết lại chính output của nó thì thay đổi ngữ nghĩa nhỏ hơn
  hẳn so với khi viết lại text người, vì text đó đã nằm trong phân phối sinh của
  model. Kết quả "cú nhảy đầu tiên rồi bão hoà" ở trên khớp với quan sát này.
- **PADBen** (arXiv 2511.00416): phân tích iterative paraphrasing trong
  representation space, mô tả "intermediate laundering region".
- **LLM as a Broken Telephone** (2025): iterative rephrasing làm suy giảm factuality;
  có khảo sát ảnh hưởng của nhiệt độ và mức ràng buộc của prompt.

Đóng góp riêng hẹp nhưng có thật: **phân loại regime bằng số** - fit α, step
autocorrelation, ensemble dispersion, và so chained với direct. Chưa thấy ai fit
exponent một cách tường minh. Viết như một bài replication có thêm một phép đo.

## Không dùng được hình UMAP làm bằng chứng

Trong F1, các trajectory `B_preserve` bay xa hơn `A_natural`, ngược hẳn với số đo
(B = 0.110, A = 0.191). UMAP không bảo toàn khoảng cách. Bỏ hình, hoặc để kèm ghi
chú rõ rằng nó chỉ minh hoạ.

## Ghi chú kỹ thuật

`Judge self-agreement: 1.000` mà script in ra là **số giả** - với `judge_votes=1`
thì hàm đó trả 1.0 theo định nghĩa, không đo gì cả.

---

## Câu kết luận nên dùng nguyên văn

> Trên ba text này, với một rewriter này, iterative rewriting không tích luỹ drift
> mà bão hoà ngay sau bước đầu tiên.

## Cần tự xác nhận trước khi đăng

- **Corpus thật sự gồm những gì.** `text_01` là bài về LLM thiết kế sản phẩm, không
  phải bài SI-SDR như dự định ban đầu. Liệt kê lại `texts/` và mô tả đúng trong
  phần Method.
- **Embedding model nào đã dùng** (`config.embed_model`, dim = 1024). Nếu corpus có
  tiếng Việt mà dùng model tiếng Anh thì phải nêu như một giới hạn.