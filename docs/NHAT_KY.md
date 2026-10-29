# Nhật ký

Mọi quyết định, thay đổi quy tắc và kết quả được ghi ở đây theo ngày, trong quá trình làm.

## 2026-10-29 — Khởi tạo

Codebase dựng lại từ mã của giai đoạn thử tính khả thi (không công khai; không kèm dữ liệu,
kết quả hay tài liệu của giai đoạn đó). Giữ lại:

- Dựng và đánh giá taxonomy (tự động và bởi người), gợi ý sản phẩm.
- Ba bộ dữ liệu: Amazon Fashion, Amazon Baby Products, DBLP (bản phát hành của TaxoGen).
  Dữ liệu thô dùng lại bản đã tải; sha256 kiểm lại khi tải/đọc.
- Phương pháp: `improved`; `taxogen` = TaxoGen **code gốc** (commit 929a7534, dựng bằng
  `tools/taxogen_original/setup_original.py`), kèm hai điểm đã có từ giai đoạn thử khả thi:
  - `driver.py`: nhãn con bị CaseOLAP lọc khỏi từ khoá của nút thì lấy vector từ
    `embeddings.txt` đầy đủ của nút (code gốc báo KeyError);
  - `taxonomy.tuning.method_rules.taxogen.min_term_coverage: 0`: cây chuyển đổi chỉ giữ
    10 thuật ngữ mỗi nút nên độ phủ thuật ngữ không so được.
- Baseline thêm: `sbert_kmeans` (tách phần đóng góp của nhúng SBERT khỏi HDBSCAN, mỏ neo và
  chấm quan hệ), `bertopic` (tuỳ chọn, chủ đề phẳng).

Bỏ:

- TaxoGen bản viết lại (ở giai đoạn thử khả thi đã thấy yếu hơn code gốc về NPMI); test toy giờ
  chạy TaxoGen code gốc và tự bỏ qua nếu chưa dựng môi trường Python 3.7.
- Handmade Products, arXiv cs (bộ thu thập OAI-PMH), các công cụ chọn bộ dữ liệu và kiểm tra
  độ trung thành của bản viết lại.
- Tài liệu của giai đoạn thử khả thi (quy tắc, đề cương và phiếu chấm); phiếu chấm sinh lại từ
  đầu. Biểu mẫu nhận mã người chấm dạng một chữ hoa + 2 chữ số, thu hẹp khi đã cấp mã.

Lưu ý về TaxoGen code gốc:

- Phân cụm của code gốc không cố định seed. Trên dữ liệu toy, với 3 cụm và filter_thre 0.2,
  khoảng 1/3 số lần chạy có một nút tầng 3 mà CaseOLAP trả về rỗng, và `case_ranker.py`
  lỗi ZeroDivisionError. Bộ toy vì vậy dùng 2 cụm và filter_thre 0 (4/4 lần chạy qua). Trên dữ
  liệu thật (hơn 30 lần chạy ở giai đoạn thử khả thi) lỗi này chưa xảy ra; nếu xảy ra, cách xử
  lý được ghi ở đây trước khi xem kết quả. Không thêm sửa đổi nào vào hành vi của code gốc vì
  lỗi này.

## 2026-10-29 — Vai trò bộ dữ liệu, chia lại tài liệu

Vai trò (chốt):

- **Bộ phát triển: Amazon Fashion, DBLP.** Đã dùng trong giai đoạn thử tính khả thi; thiết kế
  `improved` được xem lại trên train/val của hai bộ này. Bài nêu rõ vai trò này.
- **Bộ xác nhận: Amazon Baby Products.** Không chạy bước nào trên Baby cho tới khi thiết kế
  `improved`, các lưới tham số và quy tắc chọn được chốt và ghi ở đây. Ở giai đoạn thử khả
  thi, Baby đã chạy bước tune trên val của cách chia cũ; test của Baby chưa từng được đọc.

Thay đổi:

- Tài liệu được chia lại bằng seed riêng `taxonomy.split.seed: 2026` (trước: seed chính 42),
  tách khỏi seed dựng cây. Lý do: tài liệu test của cách chia cũ đã được xem trong giai đoạn
  thử khả thi.
- Bỏ ngày tháng khỏi comment trong config, giữ lý do kỹ thuật; lịch sử quyết định ghi ở đây.
- Baby Products: `evaluation.gold: true`. Comment cũ ("as in amazon_fashion.yaml") sai: 18.925
  trong 20.000 sản phẩm đầu của metadata Baby có đường dẫn danh mục; Fashion có 0/20.000.
- `rec.taxonomy_top_k` và `evaluation.assign_top_k`: 20 → 10. TaxoGen gốc chỉ giữ 10 thuật ngữ
  mỗi nút; gán item theo 20 thuật ngữ thiệt cho TaxoGen.

## 2026-10-29 — TaxoGen gốc: cố định seed

Code gốc không cố định seed; cùng tham số, cây toy dao động 2–13 nút, và test toy lỗi 1/6 lần
chạy. Nguồn ngẫu nhiên và cách xử lý (chỉ trong `driver.py` / `models/taxogen.py`, code gốc
và `original_code.patch` không đổi; thuật toán không đổi):

- k-means của spherecluster (`random_state=None`, tức RNG toàn cục của numpy): `driver.py run`
  gọi `np.random.seed(seed)` và `random.seed(seed)` trước `recur`.
- word2vec.c đa luồng không tất định (các luồng cập nhật vector song song). Embedding cục bộ gán
  cứng `-threads 20`; driver thay bằng `taxogen_original.threads` (mọi đối số khác giữ nguyên).
  `threads: 1` (trước: 20). Đo trên 2.053.580 từ DBLP: 12 luồng 30 s, 1 luồng 204 s và 197 s;
  hai lần 1 luồng trùng từng byte, lần 12 luồng khác.
- Embedding cục bộ ghi tài liệu theo thứ tự duyệt `set`: `PYTHONHASHSEED=0` cho tiến trình con.
- Embedding toàn cục không có seed, nên dùng chung cho mọi seed trên cùng tài liệu
  (`data-<corpus>-t<threads>`); mỗi seed một lần chạy (`run-...-seed<seed>`). Độ lệch chuẩn qua
  5 seed của TaxoGen giờ chỉ phản ánh khởi tạo k-means.
- Kiểm tra: cùng seed, 1 luồng, hai phiên độc lập cho cây giống hệt; cùng seed, 12 luồng thì
  khác. Test mới `test_original_code_is_deterministic_for_a_seed_with_one_thread`.
- Toy: với word2vec 1 luồng, `n_expand 15` cho cây 2 nút (corpus cục bộ của nút con quá nhỏ);
  `n_expand: 30` cho cây 10–12 nút, độ sâu 3.
- Nguyên nhân lỗi chập chờn 1/6 của test toy: TaxoGen gốc có lúc cho một nút con có 10 thuật ngữ
  trùng hệt nút cha (`root/0/0/0` = `root/0/0`, nay tái hiện được với seed 42). Đây là đầu ra
  của code gốc, được giữ nguyên; quy tắc từ chối nút trùng cha chỉ áp dụng cho các builder, và
  test được sửa theo đó. Số nút trùng của TaxoGen gốc có trong `n_degenerate_edges` và được báo
  cáo. Trên DBLP (12 cây của giai đoạn thử khả thi) không có nút nào như vậy.
- Quy tắc chấm (chốt trước khi có phiếu nào): cạnh có nút con trùng cha của TaxoGen gốc **được
  giữ** trong tập lấy mẫu như mọi cạnh khác (đánh giá đúng đầu ra của baseline); số cạnh như vậy
  trong mỗi cây và trong mẫu được báo cáo.
- Chi phí: mỗi lần gọi word2vec chậm khoảng 6–7 lần. Với 20 luồng, một lần chạy TaxoGen gốc
  mất khoảng 34 phút (DBLP) và 10 phút (Baby Products) ở giai đoạn thử khả thi; thời gian mới
  sẽ đo khi chạy.
- Bù chi phí: các ô lưới tune và các seed của TaxoGen gốc chạy song song
  (`taxogen_original.workers: 6`, một tiến trình mỗi nhân), mỗi tiến trình 1 luồng; kết quả
  không phụ thuộc số tiến trình (test: hai seed chạy song song trên cùng thư mục = chạy tuần tự).
  Embedding toàn cục được chuẩn bị một lần (có khoá). Bộ nhớ khi 6 lần chạy DBLP đồng thời chưa
  đo; nếu thiếu RAM thì giảm `workers`, kết quả không đổi.

## 2026-10-29 — Rà soát trước khi chạy full

Chưa có lần chạy full nào; mọi thay đổi dưới đây đều trước khi đọc bất kỳ tập test nào của
bản này và trước khi có phiếu chấm.

Protocol và công bằng giữa các phương pháp:

- Phiếu chấm: mọi nút được cắt về **top 10 thuật ngữ của chính phương pháp** (như các chỉ số tự
  động, `evaluation.top_k`) rồi mới xếp lại bằng CaseOLAP chung. Trước đó, nút của `improved`
  (giữ toàn bộ pool, hàng trăm thuật ngữ) được chọn 5 từ hiển thị, từ cha và tập "kẻ lạc" trong
  một pool lớn hơn nhiều so với TaxoGen gốc (10 thuật ngữ).
- DBLP: bỏ embedding word2vec của bản phát hành (huấn luyện trên mọi tiêu đề, gồm cả tài liệu
  test); word2vec luôn huấn luyện trên đúng tài liệu mà phương pháp được dựng từ đó.
- CLI tự chạy lại với `PYTHONHASHSEED=0` khi biến này chưa đặt (trước: chỉ khuyến cáo trong
  README); `env.json` của mỗi lần chạy ghi giá trị này.
- Bỏ khoá `human_eval.force` trong config (bỏ qua kiểm tra toàn vẹn mà không lộ ra); chỉ còn
  `--force` trên dòng lệnh, được ghi trong `argv` của lần chạy.

Baseline thêm (nhận xét về số baseline còn ít):

- Taxonomy: `hclus` (gom cụm phân cấp Ward trên vector word2vec, đệ quy k nút con) và `hlda`
  (hierarchical LDA, tomotopy 0.12.7, seed cố định, 1 luồng) — hai baseline của bài TaxoGen gốc.
  Lưới tune: hclus k ∈ {3, 4, 5, 6, 8}; hlda eta ∈ {0.05, 0.1, 0.2} × gamma ∈ {0.1, 1.0}; cùng
  quy tắc chọn; hlda như TaxoGen gốc chỉ giữ 10 từ mỗi nút nên `min_term_coverage: 0`. Cả hai vào
  so sánh cùng kích thước (`evaluation.size_matched_methods`). Người chấm vẫn chỉ chấm
  `improved` và `taxogen`.
- Gợi ý: RP3beta (lưới alpha ∈ {0.5, 1.0} × beta ∈ {0, 0.3, 0.6}, topk 200) và Sem+CF (đối
  chứng của Sem+Tax(Improved)+CF: cùng phép kết hợp, không có taxonomy).

Cấu trúc code: mô hình được gom vào `src/taxorec/models/` (`taxogen_improved.py`,
`taxogen.py`, `hlda.py`, `bertopic_flat.py`, `recommenders.py`); bỏ nhánh `local: w2v` không
dùng tới. `statsmodels` chỉ còn là phụ thuộc của test (đối chiếu Wilson, Fleiss' kappa).

Sửa lỗi tìm thấy khi chạy thử `tools/run_chain.sh` trên toy: thư mục `logs/` chưa tồn tại; tên
phương pháp cuối bị dính `\r` (Python trên Windows), làm hỏng đường dẫn thư mục chạy; khi một
bước lỗi, bước tune TaxoGen chạy nền không bị dừng.

Tiền xử lý (thấy khi chạy thử trên mẫu 5% review Fashion, bộ phát triển, chỉ bước train/val):
cụm từ gồm một từ lặp lại cạnh nhau (`wallet_wallet`, `love_love_love`, `bracelet_bracelet`; 4
trong 226 cụm, do bỏ dấu câu giữa hai câu) bị loại ở bước đào cụm từ. Áp dụng như nhau cho mọi
phương pháp (cùng từ vựng ứng viên).

## 2026-10-29 — Xem lại thiết kế `improved` trên Fashion và DBLP (train/val)

Thay đổi trước khi chẩn đoán:

- Từ vựng ứng viên của Amazon chỉ gồm danh từ hoặc cụm từ có từ cuối là danh từ
  (`corpus.noun_headed: true`; từ loại theo nhãn spaCy chiếm đa số của mỗi lemma). Tính từ vẫn ở
  trong tài liệu và trong cụm từ (long_sleeve). Lý do: trong phiếu chấm thử có nút gồm tính từ
  khen, dáng người, màu sắc ("green, yellow, lovely, delicate, beautiful"). Fashion: bỏ 2.498
  thuật ngữ (`corpus/removed_not_noun_headed.json`), còn 10.548 (2.870 cụm từ), 735.986 tài liệu.
  Có vài danh từ bị tính là tính từ (accordion); chấp nhận. DBLP dùng từ khoá của bản phát hành
  (vốn là cụm danh từ), không đổi.
- Mỗi nút ghi lại mọi nút con ứng viên (cỡ, điểm quan hệ, giữ hay bỏ); tune, build và ablation
  báo số nút con được chấm / bị bỏ (`relation_summary`).
- UMAP được ghi nhớ theo đầu vào trong một tiến trình (tất định, kết quả không đổi).

Chẩn đoán: `tune-taxonomy --method improved` (18 cấu hình, dựng trên train, chấm trên val;
không đọc test). DBLP: 377.931 / 566.896 tài liệu train / val, 1 giờ 46 phút; Fashion: 147.197 /
220.795, 46 phút.

- Giả thuyết cần kiểm tra: hồ sơ nút cha là top-10 thuật ngữ của chính nó, vốn rơi vào nút con
  lớn nhất, nên bộ chấm quan hệ có thể chỉ loại nút con nhỏ. **Không được xác nhận.** Ở tau 0.4,
  bộ chấm bỏ 6–22 trong 50–80 nút con được chấm (DBLP) và 5–9 trong 34–55 (Fashion); trung vị cỡ
  (số tài liệu) của nút con bị bỏ lớn hơn của nút con được giữ ở 24/32 cấu hình có nút bị bỏ; tương quan
  Spearman giữa điểm quan hệ và cỡ nằm trong [−0,39; 0,22], phần lớn âm. Ở tau 0.2 bộ chấm gần như
  không bỏ gì (0–5).
- Trên val, tau cao hơn cho subsumption cao hơn ở gần như mọi tổ hợp; NPMI tăng theo tau ở mọi
  tổ hợp của Fashion, không đều trên DBLP. Cấu hình được chọn:
  DBLP divisor 20, frac 0.02, tau 0.4 (54 nút); Fashion divisor 30, frac 0.01, tau 0.4 (58 nút).
  Cả hai chọn tau ở đầu trên của lưới.
- Fashion: ràng buộc `min_nodes: 50` loại mọi cấu hình frac 0.02 (36–49 nút, NPMI 0.03–0.10),
  nên cấu hình được chọn thuộc nhóm frac 0.01 (NPMI quanh 0). Ràng buộc này có từ trước, vì số
  câu chấm; không đổi sau khi thấy nó bất lợi cho `improved`.

Quyết định: giữ hồ sơ nút cha là top thuật ngữ (`relation_profile: top`); phương án "general"
được thêm thành một biến thể ablation để báo cáo.

## 2026-10-29 — `min_nodes` 50 → 45; chốt thiết kế

- `taxonomy.tuning.min_nodes: 45` cho mọi bộ dữ liệu (trước: 50). Quyết định sau khi thấy, trên
  val của Fashion, ràng buộc 50 loại mọi cấu hình frac 0.02 và chọn cấu hình có NPMI −0,012
  (58 nút). Quy tắc chọn áp lại trên các dòng tune đã có (không dựng lại): Fashion 50 → divisor 30,
  frac 0.01, tau 0.4 (58 nút, NPMI −0,012); 45 → divisor 40, frac 0.02, tau 0.3 (46 nút, NPMI
  0,057); 40 → divisor 40, frac 0.02, tau 0.4 (40 nút, đúng mép ràng buộc). DBLP: cả ba mức cho
  cùng cấu hình (54 nút). 45 cân bằng chất lượng và số câu chấm (khoảng 40 câu quan hệ mỗi
  phương pháp). Bài nêu rõ thay đổi này được quyết định trên val của bộ phát triển.
- **Chốt thiết kế** (tại thời điểm ghi mục này): phương pháp `improved`, các baseline, lưới tune,
  quy tắc chọn, ablation và giao thức đánh giá là cấu hình trong `configs/` cùng code hiện tại.
  Từ đây không sửa thêm dựa trên kết quả; nếu phát hiện lỗi lập trình, cách sửa và ảnh hưởng
  được ghi ở đây, kết quả trước và sau đều được báo cáo. Baby Products (bộ xác nhận) được chạy
  lần đầu sau mục này.
