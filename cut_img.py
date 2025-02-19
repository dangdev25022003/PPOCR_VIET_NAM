# import json
# import os
# import cv2
# import copy
# import numpy as np
# from paddleocr.tools.infer.utility import draw_ocr_box_txt, get_rotate_crop_image
 
# def print_draw_crop_rec_res( img_crop_list, img_name):
#         bbox_num = len(img_crop_list)
#         for bno in range(bbox_num):
#           crop_name=img_name+'_'+str(bno)+'.jpg'
#           crop_name_w = "/home/dangph/Downloads/vietnamese_original/vietnamese/data3/img_crop/{}".format(crop_name)
#           cv2.imwrite(crop_name_w, img_crop_list[bno])
#           crop_label.write("{0}\t{1}\n".format(crop_name, text[bno]))
 
# if not os.path.exists('/home/dangph/Downloads/vietnamese_original/vietnamese/data3'):
#   os.mkdir('/home/dangph/Downloads/vietnamese_original/vietnamese/data3') 
 
# crop_label = open('/home/dangph/Downloads/vietnamese_original/vietnamese/data3/crop_label.txt','w', encoding='utf8')
# with open('/home/dangph/Downloads/vietnamese_original/vietnamese/data/train_label.txt','r', encoding='utf8') as file_text:
#   img_files=file_text.readlines()
  
# count=0
# for img_file in img_files:
#   content = json.loads(img_file.split('\t')[1].strip())
 
#   dt_boxes=[]
#   text=[]
  
#   for i in content:
#     content = i['points']
#     if i['transcription'] == "###":
#       count+=1
#       continue
#     bb = np.array(i['points'],dtype=np.float32)
#     dt_boxes.append(bb)
#     text.append(i['transcription'])
 
#   image_file = '/home/dangph/Downloads/vietnamese_original/vietnamese/data/train_images/' +img_file.split('\t')[0]
#   img = cv2.imread(image_file)
#   ori_im=img.copy()
#   img_crop_list=[]
 
#   for bno in range(len(dt_boxes)):
#     tmp_box = copy.deepcopy(dt_boxes[bno])
#     img_crop = get_rotate_crop_image(ori_im, tmp_box)
#     img_crop_list.append(img_crop)
#   img_name = img_file.split('\t')[0].split('.')[0]
  
#   if not os.path.exists('/home/dangph/Downloads/vietnamese_original/vietnamese/data3/img_crop'):
#     os.mkdir('/home/dangph/Downloads/vietnamese_original/vietnamese/data3/img_crop') 
#   print_draw_crop_rec_res(img_crop_list,img_name)
import json
import os
import cv2
import copy
import numpy as np
from tools.infer.utility import draw_ocr_box_txt, get_rotate_crop_image

# Hàm lưu ảnh cắt và nhãn
def save_cropped_images_and_labels(img_crop_list, img_name, text, split):
    crop_dir = f"/home/dangph/Downloads/vietnamese_original/vietnamese/data3/img_crop/{split}/"
    
    # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir)

    bbox_num = len(img_crop_list)
    for bno in range(bbox_num):
        crop_name = img_name + '_' + str(bno) + '.jpg'
        crop_name_w = os.path.join(crop_dir, crop_name)
        
        # Kiểm tra trước khi lưu
        if img_crop_list[bno] is None or img_crop_list[bno].size == 0:
            print(f"Crop image at index {bno} is None or empty. Cannot save this image.")
            continue
        
        print(f"Attempting to save image: {crop_name_w} with shape {img_crop_list[bno].shape} and dtype {img_crop_list[bno].dtype}")
        success = cv2.imwrite(crop_name_w, img_crop_list[bno])
        if not success:
            print(f"Failed to save crop image: {crop_name_w}")
        else:
            crop_label.write("{0}\t{1}\n".format(crop_name, text[bno]))

# Kiểm tra và tạo thư mục gốc
base_dir = '/home/dangph/Downloads/vietnamese_original/vietnamese/data3'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

# Tạo thư mục img_crop
img_crop_dir = os.path.join(base_dir, 'img_crop')
if not os.path.exists(img_crop_dir):
    os.mkdir(img_crop_dir)

# Tạo các thư mục con cho train, val, test
for split in ['train', 'val', 'test']:
    split_dir = os.path.join(img_crop_dir, split)
    if not os.path.exists(split_dir):
        os.mkdir(split_dir)

crop_label = open('/home/dangph/Downloads/vietnamese_original/vietnamese/data3/crop_label.txt','w', encoding='utf8')
with open('/home/dangph/Downloads/vietnamese_original/vietnamese/data3/train_label.txt','r', encoding='utf8') as file_text:
    img_files = file_text.readlines()

# Phân chia tỷ lệ
total_images = len(img_files)
train_count = int(total_images * 0.7)
val_count = int(total_images * 0.2)
test_count = total_images - train_count - val_count

# Tạo danh sách cho các chỉ số
train_indices = list(range(train_count))
val_indices = list(range(train_count, train_count + val_count))
test_indices = list(range(train_count + val_count, total_images))

# Chạy qua từng chỉ số và xử lý ảnh
for idx in range(total_images):
    img_file = img_files[idx]
    content = json.loads(img_file.split('\t')[1].strip())

    dt_boxes = []
    text = []

    for i in content:
        if i['transcription'] == "###":
            continue
        bb = np.array(i['points'], dtype=np.float32)
        dt_boxes.append(bb)
        text.append(i['transcription'])

    image_file = '/home/dangph/Downloads/vietnamese_original/vietnamese/data3/' + img_file.split('\t')[0]
    img = cv2.imread(image_file)
    ori_im = img.copy()
    img_crop_list = []

    for bno in range(len(dt_boxes)):
        tmp_box = copy.deepcopy(dt_boxes[bno])
        img_crop = get_rotate_crop_image(ori_im, tmp_box)
        img_crop_list.append(img_crop)

    img_name = img_file.split('\t')[0].split('.')[0]

    # Xác định tập (train, val, test) dựa trên chỉ số
    if idx in train_indices:
        split = "train"
    elif idx in val_indices:
        split = "val"
    else:
        split = "test"
    
    save_cropped_images_and_labels(img_crop_list, img_name, text, split)

crop_label.close()

# Kiểm tra nội dung các thư mục
for split in ["train", "val", "test"]:
    folder_path = f"/home/dangph/Downloads/vietnamese_original/vietnamese/data3/img_crop/{split}/"
    if os.path.exists(folder_path):
        files = os.listdir(folder_path)
        print(f"{split.capitalize()} folder contains {len(files)} files.")
    else:
        print(f"{split.capitalize()} folder does not exist.")
