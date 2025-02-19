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
Cài các thư viện cần thiết chạy:
```bash 
pip install -r requirements.txt
```
## Huấn luyện mô hình
1. **Chuẩn bị dữ liệu**: Thu thập hình ảnh và nhãn đi kèm (dưới dạng file `.txt`):
   Bạn có theerr sử dụng trực tiêp dữ liệu cảu VINAI: https://github.com/VinAIResearch/dict-guided

2. **Tiền xử lý dữ liệu**: Chuyển đổi hình ảnh về định dạng phù hợp, gán nhãn.
   Sau khi tải về và giải nén ra ta sẽ có:
    Folder labels – chứa các file annotation của từng image,
    Folder train_images – chứa 1200 ảnh từ im0001 đến im1200,
    Folder test_image – chứa 300 ảnh từ im1201 đến im1500,
    Folder unseen_test_images – chứa 500 ảnh từ im1501 đến im2000,
    File general_dict.txt,
    File vn_dictionary.txt
   ![image](https://github.com/user-attachments/assets/466faca4-9a21-4bae-9789-59552f464e53)

4. **Huấn luyện**: Chạy lệnh sau để huấn luyện mô hình:
   ![image](https://github.com/user-attachments/assets/73382023-29d1-402d-a030-7a8f06c9a221)

   ```bash
    import os
    import numpy as np
    from tqdm import tqdm
    import pandas as pd
    import json
    import glob
    
    root_path = glob.glob('./vietnamese/labels/*')
    
    train_label = open("train_label.txt","w")
    test_label = open("test_label.txt","w")
    useen_label = open("useen_label.txt","w")
    for file in root_path:
        with open(file) as f:
          content = f.readlines()
          f.close()
        content = [x.strip() for x in content]
        text = []
        for i in content:
          label = {}
          i = i.split(',',8)
          label['transcription'] = i[-1]
          label['points'] = [[i[0],i[1]],[i[2],i[3]], [i[4],i[5]],[i[6],i[7]]]
          text.append(label)
    
        content = []
        text = json.dumps(text, ensure_ascii=False)
    
        img_name = os.path.basename(file).split('.')[0].split('_')[1]
        int_img = int(img_name)
        img_name = 'im' + "{:04n}".format(int(img_name)) + '.jpg'
        if int_img &gt; 1500:
          useen_label.write( img_name+ '\t'+f'{text}' + '\n')
        elif int_img &gt; 1200:
          test_label.write( img_name+ '\t'+f'{text}' + '\n')
        else:
          train_label.write( img_name+ '\t'+f'{text}' + '\n')
   ```
   Sau khi chuyển đổi ta cần thực hiện truyền các đường dẫn dữ liệu và thông số phù hợp vào file config: SAST + ResNet50_vd
   Đầu tiên download pretrained model đã được train sẵn với dataset tiếng anh đạt kết quả cao với ICDAR2015 dataset:
   href="https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_sast_icdar15_v2.0_train.tar"
   Huấn luyện mô hình:
   ```bash	
    python tools/train.py -c ./configs/det/SAST.yml
   ```
   
6. **Lưu mô hình**: Mô hình sẽ được lưu trong thư mục `output`.

## Dự đoán trên ảnh
```bash
python tools/infer/predict_system.py --image_dir="path_to_image.jpg" --det_model_dir="path_to_det_model" --rec_model_dir="path_to_rec_model" --use_angle_cls=True
```

## Kết quả
Kết quả nhận diện sẽ được hiển thị trên ảnh đầu vào và xuất dưới dạng văn bản.

## Liên hệ
Nếu có bất kỳ vấn đề nào, vui lòng liên hệ qua email: `your_email@example.com`.
