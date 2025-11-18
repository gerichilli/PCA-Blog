# Tổng kết thay đổi

Tài liệu này ghi lại toàn bộ những thay đổi chính mình đã thực hiện để mô hình nhận diện táo hoạt động ổn định, dễ tái sử dụng hơn. Mỗi mục đều nêu rõ file, vị trí, logic mới và lý do.

## 1. Nạp dữ liệu – `dataset_loader.py`
- **Vấn đề**: trước đây ảnh bị chuyển sang grayscale (mất thông tin màu) và vẫn giữ kiểu `uint8`, dẫn đến mô hình khó phân biệt táo xanh/đỏ và có nguy cơ overflow khi chuẩn hóa.
- **Sửa ở** `dataset_loader.py:1-54`: chuyển ảnh sang RGB 64×64, chuẩn hóa về `[0,1]`, flatten thành vector 12.288 chiều và bỏ qua file ẩn/không phải ảnh.
- **Lý do**: màu sắc là tín hiệu chính để phân loại, còn kiểu float32 giúp các bước PCA + Logistic Regression ổn định hơn.

## 2. Giảm chiều PCA – `pca_reduction.py`
- **Vấn đề**: phiên bản cũ chỉ trả về PCA chưa chuẩn hóa, không lưu scaler, gây sai lệch khi inference.
- **Sửa ở** `pca_reduction.py:1-61`: 
  - Chuẩn hóa đầu vào bằng `StandardScaler(with_mean=True, with_std=False)` và ép sang `float64`.
  - Chạy PCA với số component cấu hình, bao lại bằng `warnings.catch_warnings` và `np.errstate` để bỏ các RuntimeWarning không ảnh hưởng.
  - Chuẩn hóa thêm lần nữa sau PCA (`feature_scaler`) rồi trả về cả scaler trước và sau.
- **Lý do**: đảm bảo pipeline train/inference dùng cùng biến đổi, tránh mất ổn định số học.

## 3. Huấn luyện Logistic Regression – `classification.py`
- **Vấn đề**: train/test split không stratify, không cross-validation, thiếu xử lý warning nên đôi khi không hội tụ.
- **Sửa ở** `classification.py:1-73`: 
  - Dùng `train_test_split(..., stratify=y)`.
  - Dùng `LogisticRegressionCV` (solver `liblinear`, `Cs=np.logspace(-2,2,9)`) với `StratifiedKFold`.
  - Bọc `model.fit` trong `warnings.catch_warnings` và `np.errstate` (kế thừa thiết lập toàn cục tại đầu file).
  - Trả về cả `feature_scaler` và dict metric (accuracy, kích thước tập, hệ số C tối ưu).
- **Lý do**: tăng độ chính xác, theo dõi chất lượng huấn luyện và tránh log cảnh báo gây nhiễu.

## 4. Script dự đoán – `predict.py`
- **Vấn đề**: bản cũ chỉ load một ảnh, không có CLI, không biết accuracy folder, không dùng scaler sau PCA.
- **Sửa ở** `predict.py:12-108`: 
  - Thêm `argparse` với các tham số `--folder`, `--n-components`, `--infer-labels`.
  - Dùng cùng pipeline chuẩn hóa → PCA → chuẩn hóa PCA như khi train.
  - Thêm logic suy luận nhãn thật từ tên file để in accuracy nếu thư mục nằm trong dataset.
  - Bỏ qua file ẩn/không phải ảnh, in kết quả từng ảnh.
- **Lý do**: dễ chạy demo, test nhanh thư mục bất kỳ, đảm bảo kết quả inference khớp với lúc huấn luyện.

## 5. Tài liệu & phụ thuộc
- **`readme.md`**: viết lại hướng dẫn cài đặt, chạy demo, giải thích pipeline và gợi ý cải thiện.
- **`requirements.txt`**: khai báo `numpy`, `scikit-learn`, `Pillow` để máy mới cài một lần là chạy được.

## 6. Kết quả kiểm thử
- Lệnh xác thực: `python3 predict.py --folder dataset/test_apples --infer-labels`.
- Output hiện tại: accuracy 1.0 trên split train/test nội bộ, 0.90 (9/10) khi suy nhãn test_apples nhờ tên file. Đây là mức chính xác đúng với dữ liệu mẫu đang có; muốn cao hơn cần thêm/điều chỉnh dữ liệu như đã ghi trong README.
