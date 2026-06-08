import random
import shutil
from pathlib import Path

RAW_DIR = Path('dataset_raw')
OUT_DIR = Path('dataset')
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

random.seed(42)

images = []
for ext in IMAGE_EXTS:
    images.extend(RAW_DIR.glob(f'*{ext}'))
    images.extend(RAW_DIR.glob(f'*{ext.upper()}'))

images = sorted(set(images))
random.shuffle(images)

n = len(images)
train_end = int(n * TRAIN_RATIO)
val_end = train_end + int(n * VAL_RATIO)

splits = {
    'train': images[:train_end],
    'val': images[train_end:val_end],
    'test': images[val_end:]
}

for split, files in splits.items():
    (OUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

    for img_path in files:
        label_path = img_path.with_suffix('.txt')
        shutil.copy2(img_path, OUT_DIR / 'images' / split / img_path.name)

        if label_path.exists():
            shutil.copy2(label_path, OUT_DIR / 'labels' / split / label_path.name)
        else:
            print(f'Không thấy nhãn cho ảnh: {img_path.name}')

# copy classes.txt nếu có
classes = RAW_DIR / 'classes.txt'
if classes.exists():
    shutil.copy2(classes, OUT_DIR / 'classes.txt')

print('Đã chia dataset xong!')
print(f'Train: {len(splits["train"])} ảnh')
print(f'Val: {len(splits["val"])} ảnh')
print(f'Test: {len(splits["test"])} ảnh')
