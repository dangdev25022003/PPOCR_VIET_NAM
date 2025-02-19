import os

# Đường dẫn tới thư mục chứa ảnh và file txt
folder_path = '/home/dangph/Downloads/vietnamese_original/vietnamese/data3/img_crop/val/train_images'  # Thay đổi thành đường dẫn tới thư mục chứa ảnh của bạn
txt_file_path = '/home/dangph/Downloads/vietnamese_original/vietnamese/data3/crop_label.txt'   # Đường dẫn tới file txt chứa tên ảnh và thông tin
output_file_path = '/home/dangph/Downloads/vietnamese_original/vietnamese/data3/val.txt'  # Đường dẫn tới file txt để lưu kết quả

# Lấy danh sách các tên file ảnh trong thư mục
image_files = {f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')}  # Có thể thêm các định dạng file khác nếu cần

# In danh sách file ảnh để kiểm tra
print("Danh sách file ảnh:", image_files)

# Mở file txt để đọc và file output để ghi
with open(txt_file_path, 'r', encoding='utf-8') as txt_file, open(output_file_path, 'w', encoding='utf-8') as output_file:
    for line in txt_file:
        # Lấy tên ảnh từ dòng txt (phần trước tab)
        image_name = line.split('\t')[0].strip()  # Thêm strip để loại bỏ khoảng trắng
        
        # Chỉ lấy tên file mà không có đường dẫn
        base_image_name = os.path.basename(image_name)

        # Kiểm tra xem tên ảnh có nằm trong danh sách ảnh hay không
        if base_image_name in image_files:
            # Nếu có, ghi dòng đó vào file output
            output_file.write(line)

print("Hoàn thành việc ghi vào file mới!")
