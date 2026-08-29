# Semantic drift under iterative LLM rewriting

Đo xem viết lại nhiều lần bằng LLM thì text trôi trong semantic space theo kiểu gì,
và thông tin mất đi có tích luỹ không.

## Chạy

```bash
pip install -r requirements.txt
source .env 
# bỏ 5-10 đoạn text kỹ thuật/lập luận (300-800 từ) vào texts/*.txt

python run.py          # sinh trajectories -> embed -> chấm claim
python analyze.py      # metrics + 5 figure trong results/
```

`run.py --no-judge` để bỏ phần chấm claim (phần đắt nhất).
Mọi call LLM được cache trong sqlite, crash giữa chừng chạy lại không mất tiền.

## Sizing

Số rewrite call = `texts × 3 conditions × 2 modes × seeds × iterations`

| | texts | seeds | iters | rewrite calls | judge calls |
|---|---|---|---|---|---|
| pilot | 3 | 4 | 8 | 576 | ~500 |
| full | 8 | 20 | 10 | 9 600 | ~6 700 |

Chạy pilot trước. Nếu pilot không thấy tách biệt giữa A và C thì thiết kế sai,
scale lên chỉ đốt tiền.

## Bốn quyết định thiết kế quan trọng

**1. Condition C là baseline, không phải điều kiện phụ.**
Mọi sentence embedding đều trộn style và content. Drift của A một mình không
chứng minh được gì. `drift_A − drift_C` mới là ước lượng phần semantic.

**2. Câu hỏi thật là regime, không phải "có tăng không".**
Distance chắc chắn tăng. Cái đáng đo là exponent α trong `R_n ~ n^α`:

- α ≈ 0.5 → random walk thuần, không hướng
- α ≈ 1.0 → ballistic: model kéo text về prior của nó, có hướng ưu tiên
- α → 0 → saturation: có attractor, text hội tụ về "house style" rồi đứng im

Ba kết luận này khác nhau hoàn toàn. Ba metric phân biệt chúng, đều có trong
`analyze.py`: exponent α (F3), step autocorrelation, và ensemble dispersion (F4).
Dispersion co lại theo n là bằng chứng mạnh cho attractor — và theo tôi đó là
kết quả có khả năng xảy ra nhất, chứ không phải random walk vô hạn.

**3. Bootstrap trên text, không trên seed.**
20 seed của cùng một đoạn text vẫn chỉ là một quan sát về generalization.
CI tính trên seed sẽ hẹp giả tạo.

**4. LLM judge là điểm yếu nhất của thiết kế.**
Dùng cùng một model để vừa rewrite vừa chấm claim là gần như circular.
Khi chạy thật: đặt `judge_model` khác `rewrite_model`, đặt `judge_votes=3`,
và xem `Judge self-agreement` mà `analyze.py` in ra. Dưới 0.80 thì bỏ toàn bộ
kết quả claim, chỉ giữ phần embedding.

## Kết quả đọc thế nào

| figure | trả lời câu hỏi |
|---|---|
| F1 trajectories | quỹ đạo có thật sự tách khỏi gốc không, hay chỉ nhiễu |
| F2 drift vs iteration | chained có tích luỹ hơn direct không (nếu bằng nhau → không có compounding) |
| F3 scaling | drift thuộc regime nào |
| F4 dispersion | diffusion hay attractor |
| F5 claims | preservation giảm trong khi embedding similarity vẫn cao → kết quả mạnh nhất |

Kết hợp đáng giá nhất: **F5 giảm mạnh trong khi F2 của cùng condition tăng chậm**.
Nghĩa là bài vẫn "trông giống" bản gốc trong embedding space nhưng các luận điểm
cụ thể đã bốc hơi — đúng thứ mà cosine similarity không bắt được.

## Chỗ thiết kế còn hở

- Claim extraction chạy trên original, không re-extract mỗi vòng → không bắt được
  claim mới sinh ra rồi lại mất. Chấp nhận được cho v1.
- Chưa control độ dài. LLM rewrite hay làm ngắn lại; độ dài giảm cũng gây drift.
  Nên log `len(text)` và regress drift theo Δlength trước khi kết luận.
- Chỉ một rewrite model. Muốn nói "LLM nói chung" thì cần ≥2 nhà cung cấp.
