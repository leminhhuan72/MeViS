# use_ytvos.py
from ytvos import YTVOS

# Đường dẫn đến tệp chú thích của bạn
annotation_file = '/root/MeViS/datasets/ytvis_2019/train/meta.json'

# Khởi tạo đối tượng YTVOS với tệp chú thích
ytvos = YTVOS(annotation_file)

# Gọi các phương thức khác nhau để kiểm tra dữ liệu
ytvos.info()  # Hiển thị thông tin về tệp chú thích

# Lấy danh sách các video IDs
video_ids = ytvos.getVidIds()
print(f"Video IDs: {video_ids}")

# Lấy danh sách các danh mục IDs
category_ids = ytvos.getCatIds()
print(f"Category IDs: {category_ids}")

# Lấy danh sách các chú thích IDs cho một video cụ thể
ann_ids = ytvos.getAnnIds(vidIds=video_ids[0])
print(f"Annotation IDs for video {video_ids[0]}: {ann_ids}")

# Nạp các chú thích
annotations = ytvos.loadAnns(ids=ann_ids)
print(f"Annotations: {annotations}")

# Nạp danh mục
categories = ytvos.loadCats(ids=category_ids)
print(f"Categories: {categories}")

# Nạp video
videos = ytvos.loadVids(ids=video_ids)
print(f"Videos: {videos}")