import os
import shutil
import random

base = r"C:\Users\Şevval\Downloads\Plaka.yolov11"

src_images = os.path.join(base, "train", "images")
src_labels = os.path.join(base, "train", "labels")

# Hedef klasörler
for split in ["valid", "test"]:
    os.makedirs(os.path.join(base, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(base, split, "labels"), exist_ok=True)

# Görselleri listele
images = [f for f in os.listdir(src_images) if f.endswith((".jpg", ".png", ".jpeg"))]
random.shuffle(images)

total = len(images)
valid_count = int(total * 0.15)
test_count = int(total * 0.05)

valid_imgs = images[:valid_count]
test_imgs = images[valid_count:valid_count + test_count]

def move_files(file_list, split):
    for img in file_list:
        # Görseli taşı
        shutil.move(
            os.path.join(src_images, img),
            os.path.join(base, split, "images", img)
        )
        # Label'ı taşı
        label = os.path.splitext(img)[0] + ".txt"
        label_src = os.path.join(src_labels, label)
        if os.path.exists(label_src):
            shutil.move(label_src, os.path.join(base, split, "labels", label))

move_files(valid_imgs, "valid")
move_files(test_imgs, "test")

print(f"Train: {total - valid_count - test_count} görsel")
print(f"Valid: {valid_count} görsel")
print(f"Test: {test_count} görsel")
print("Bölme tamamlandı!")