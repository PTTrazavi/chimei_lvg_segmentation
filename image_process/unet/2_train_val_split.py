import os
import random
# set folder paths
train_folder = "chimei_t_v/train/"
train_label_folder = "chimei_t_v/trainannot/"
val_folder = "chimei_t_v/val/"
val_label_folder = "chimei_t_v/valannot/"

image_list = [i for i in os.listdir(train_folder) if not i.startswith('.')]
print(image_list[:4])
random.shuffle(image_list)
print(image_list[:4])

print("total images:", len(image_list))
val_ratio = int(0.1 * len(image_list))

for i in image_list[:val_ratio]:
    os.rename(os.path.join(train_folder, i), os.path.join(val_folder, i))
    os.rename(os.path.join(train_label_folder, i), os.path.join(val_label_folder, i))

print("total train images:", len([i for i in os.listdir(train_folder) if not i.startswith('.')]))
print("total train label images:", len([i for i in os.listdir(train_label_folder) if not i.startswith('.')]))
print("total val images:", len([i for i in os.listdir(val_folder) if not i.startswith('.')]))
print("total val label images:", len([i for i in os.listdir(val_label_folder) if not i.startswith('.')]))
