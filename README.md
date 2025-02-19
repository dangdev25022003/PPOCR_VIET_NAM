<p align="center">
<H1>PPOCR_VIET_NAM</H1>
</p>

<div align="center">
    <img src="https://github.com/PaddlePaddle/PaddleOCR/releases/download/v2.8.0/demo.gif" width="800">
</div>
# Vision 1 - Trích Xuất Thông Tin Từ Giấy Tờ và Biển Hiệu

## Giới thiệu
Dự án này sử dụng PaddleOCR để phát hiện và trích xuất văn bản tiếng Việt từ hình ảnh của các loại giấy tờ và biển hiệu. Chương trình sẽ thực hiện các bước xử lý ảnh, nhận diện chữ và xuất kết quả dưới dạng văn bản.

## Yêu cầu hệ thống
- Python >= 3.7
- PaddlePaddle
- PaddleOCR
- OpenCV
- NumPy



## Cài đặt
```bash
# Cài đặt PaddleOCR và các thư viện cần thiết
pip install paddlepaddle
pip install paddleocr
pip install opencv-python numpy
```
```bash 
pip install -r requirements.txt ```
## Huấn luyện mô hình
1. **Chuẩn bị dữ liệu**: Thu thập hình ảnh và nhãn đi kèm (dưới dạng file `.txt`).
2. **Tiền xử lý dữ liệu**: Chuyển đổi hình ảnh về định dạng phù hợp, gán nhãn.
3. **Huấn luyện**: Chạy lệnh sau để huấn luyện mô hình:
   ```bash
   python tools/train.py -c configs/rec/ch_PP-OCRv3.yml
   ```
4. **Lưu mô hình**: Mô hình sẽ được lưu trong thư mục `output`.

## Dự đoán trên ảnh
```bash
python tools/infer/predict_system.py --image_dir="path_to_image.jpg" --det_model_dir="path_to_det_model" --rec_model_dir="path_to_rec_model" --use_angle_cls=True
```

## Kết quả
Kết quả nhận diện sẽ được hiển thị trên ảnh đầu vào và xuất dưới dạng văn bản.

## Liên hệ
Nếu có bất kỳ vấn đề nào, vui lòng liên hệ qua email: `your_email@example.com`.
