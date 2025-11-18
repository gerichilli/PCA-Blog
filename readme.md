Tạo môi trường: python3 -m venv .venv && source .venv/bin/activate
Cài thư viện: pip install -r requirements.txt
Huấn luyện + demo test set: python predict.py --folder dataset/test_apples --infer-labels
Xem kết quả từng ảnh và độ chính xác suy ra từ tên file; thông tin training (train/test size, C tối ưu) in ra đầu log.
Muốn dùng thư mục khác: python predict.py --folder path/to/images --no-infer-labels; cần ảnh 64×64 RGB (script tự resize) với tên tùy ý.



# Apple PCA Classifier

Dự án huấn luyện mô hình Logistic Regression (kết hợp PCA) để nhận diện táo xanh / táo đỏ từ bộ ảnh đã cho.

## 1. Dataset

```
dataset/
├─ green_apples/   # ảnh dùng để huấn luyện (nhãn 0)
├─ red_apples/     # ảnh dùng để huấn luyện (nhãn 1)
└─ test_apples/    # ảnh kiểm thử, tên file đã chứa nhãn mong đợi
```

*Ảnh được resize 64×64, giữ nguyên 3 kênh RGB, chuẩn hóa về [0,1] rồi flatten (12.288 chiều) trước khi đưa vào PCA.*

## 2. Chuẩn bị môi trường

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Các thư viện chính: `numpy`, `scikit-learn`, `Pillow`.

## 3. Huấn luyện & đánh giá

`predict.py` vừa huấn luyện mô hình vừa test trên thư mục ảnh bất kỳ.

```bash
python predict.py \
  --folder dataset/test_apples \
  --n-components 100 \
  --infer-labels
```

- PCA giảm số chiều xuống `--n-components` (mặc định 100).
- Logistic Regression được tinh chỉnh hệ số phạt C bằng cross-validation (StratifiedKFold 5 phần) để tối ưu độ chính xác.
- Nếu dùng `--infer-labels`, script tự suy ra nhãn thật từ tên file chứa từ khóa `green` / `red` và in accuracy.

Ví dụ output:

```
Training metrics: {'test_accuracy': 0.92, 'train_size': 5177, 'test_size': 1295, 'best_C': 2.1544}
Testing folder: dataset/test_apples
green_apple_01.jpg ➝ Green
...
Inferred folder accuracy: 1.0000 (10 labeled samples)
```

## 4. Dự đoán ảnh mới

```
python predict.py --folder path/to/your_images --no-infer-labels
```

Thư mục có thể chứa bất kỳ ảnh `.jpg/.jpeg/.png/.bmp`. Script sẽ bỏ qua file ẩn.

## 5. Mẹo cải thiện độ chính xác

1. **Tăng dữ liệu**: bổ sung thêm ảnh (nhiều bối cảnh / ánh sáng) ở cả hai lớp.
2. **Điều chỉnh PCA**: thử `--n-components` nằm trong khoảng 80–200, giữ lại nhiều thông tin hơn nếu kích thước dataset đủ lớn.
3. **Tiền xử lý**: đảm bảo ảnh không bị nhiễu, crop tập trung vào quả táo, cân bằng sáng.
4. **Bổ sung augmentations**: nhân bản dữ liệu bằng cách xoay nhẹ, thay đổi độ sáng trước khi đưa vào pipeline (có thể mở rộng trong `dataset_loader.py` nếu cần).
