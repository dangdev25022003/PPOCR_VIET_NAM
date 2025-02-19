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
<H1>Model Detection SAST</H1>
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
Dưới đây là toàn bộ file cần thiết cho configs:
```bash
https://drive.google.com/drive/folders/1DLTX7q04cJOYYO0LDdsyYuY7FHryukDT?usp=sharing
```
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
   
5. **Lưu mô hình**: Mô hình sẽ được lưu trong thư mục `output`.
# Convert mô hình dectection
    ```bash
    python tools/export_model.py -c ./configs/det/SAST.yml  \
                                 -o Global.pretrained_model=./output/SAST/latest \
                                    Global.save_inference_dir=./inference/SAST
    ```
## Dự đoán trên ảnh
    ```bash
    python tools/infer/predict_det.py  --det_algorithm="SAST" \
                                       --use_gpu=False \
                                       --det_model_dir="./inference/SAST"  \
                                       --image_dir="your_test_image_path"
    ```
<H1>Model Recognition SAST</H1>
1. **Chuẩn bị dữ liệu**:
![image](https://github.com/user-attachments/assets/ea7b1a62-27b3-4406-a7c9-6cb8b7a92ac6)
Có thể thấy rằng data để train model recognition là mỗi ảnh chứa một chữ trong khi data đã download lại là ảnh chứa nhiều chữ. Do vậy ta phải cắt nhỏ ảnh đã có từ các box chứa text.
    ```bash
    import json
    import os
    import cv2
    import copy
    import numpy as np
    from paddleocr.tools.infer.utility import draw_ocr_box_txt, get_rotate_crop_image
    
    def print_draw_crop_rec_res( img_crop_list, img_name):
            bbox_num = len(img_crop_list)
            for bno in range(bbox_num):
              crop_name=img_name+'_'+str(bno)+'.jpg'
              crop_name_w = "./data/vietnamese/train/img_crop/{}".format(crop_name)
              cv2.imwrite(crop_name_w, img_crop_list[bno])
              crop_label.write("{0}\t{1}\n".format(crop_name, text[bno]))
    
    if not os.path.exists('./data/vietnamese/train/'):
      os.mkdir('./data/vietnamese/train/') 
    
    crop_label = open('./data/vietnamese/train/crop_label.txt','w', encoding='utf8')
    with open('./data/train_label.txt','r', encoding='utf8') as file_text:
      img_files=file_text.readlines()
      
    count=0
    for img_file in img_files:
      content = json.loads(img_file.split('\t')[1].strip())
    
      dt_boxes=[]
      text=[]
      
      for i in content:
        content = i['points']
        if i['transcription'] == "###":
          count+=1
          continue
        bb = np.array(i['points'],dtype=np.float32)
        dt_boxes.append(bb)
        text.append(i['transcription'])
    
      image_file = './data/vietnamese/train_images/' +img_file.split('\t')[0]
      img = cv2.imread(image_file)
      ori_im=img.copy()
      img_crop_list=[]
    
      for bno in range(len(dt_boxes)):
        tmp_box = copy.deepcopy(dt_boxes[bno])
        img_crop = get_rotate_crop_image(ori_im, tmp_box)
        img_crop_list.append(img_crop)
      img_name = img_file.split('\t')[0].split('.')[0]
      
      if not os.path.exists('./data/vietnamese/train/img_crop'):
        os.mkdir('./data/vietnamese/train/img_crop') 
      print_draw_crop_rec_res(img_crop_list,img_name)
    ```
Để dự đoán được tiếng việt thì ta cũng cần một file dictionary dành cho tiếng Việt chứa tất cả các kí tự. Do PaddleOCR chưa hỗ trợ tiếng Việt nên ta sẽ dùng file dictionary riêng để train:
Dưới đây là toàn bộ file cần thiết cho configs:
```bash
https://drive.google.com/drive/folders/1DLTX7q04cJOYYO0LDdsyYuY7FHryukDT?usp=sharing
```
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
Chuẩn bị file Config
Cách chỉnh sửa file config model recognition cũng tương tự như model detection. Chỉ có một số điểm khác là character_dict_path – đường dẫn file dictionary, use_space_char – dự đoán khoảng trắng hay không, max_text_length – độ dài tối đa kí tự     trong một box.
Đầu tiên download pretrained model đã được train sẵn với dataset tiếng anh đạt kết quả cao với ICDAR2015:
    href="https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r50_vd_srn_train.tar"
Thêm đường dẫn pretrain model vừa tải vào Global.pretrained_model trong file config (configs/rec/SRN.yml)
pretrained_model: ./pretrain_models/rec_r50_vd_srn_train/best_accuracy

Tiếp theo chỉnh sửa đường dẫn file dictionary
character_dict_path: ./ppocr/utils/dict/vi_vietnam.txt
Huấn luyện mô hình:
   ```bash	
    python tools/train.py -c ./configs/rec/SRN.yml 
   ```
   
6. **Lưu mô hình**: Mô hình sẽ được lưu trong thư mục `output`.
# Convert mô hình dectection
    ```bash
    python tools/export_model.py -c ./configs/rec/SRN.yml \
                             -o Global.pretrained_model=./output/SRN/latest \
                                Global.save_inference_dir=./inference/SRN
    ```
## Dự đoán trên ảnh
    ```bash
    python tools/infer/predict_rec.py  --image_dir="your_test_image_path" \
                                       --use_gpu=False \
                                       --rec_algorithm="SRN" \
                                       --rec_model_dir="./inference/SRN"  \
                                       --rec_image_shape="1, 64, 256"  \
                                       --rec_char_type="ch"   \
                                       --rec_char_dict_path="./ppocr/utils/dict/vi_vietnam.txt"
    ```
##Có thể chạy song song cùng lúc 2 model bằng cách
    ```bash
        python ./tools/infer/predict_system.py \
        --use_gpu=False \
        --det_algorithm="SAST"  \
        --det_model_dir="./inference/SAST"  \
        --rec_algorithm="SRN" \
        --rec_model_dir="./inference/SRN"  \
        --rec_image_shape="1, 64, 256"  \
        --rec_char_type="ch"  \
        --rec_char_dict_path="./ppocr/utils/dict/vi_vietnam.txt" \
        --drop_score=0.5  \
        --vis_font_path="./font-times-new-roman.ttf" \
        --image_dir="your_test_image_path"
    ```
## Kết quả
Kết quả nhận diện sẽ được hiển thị trên ảnh đầu vào và xuất dưới dạng văn bản.

## Liên hệ
Nếu có bất kỳ vấn đề nào, vui lòng liên hệ qua email: `pdang0952@gmail.com`.
