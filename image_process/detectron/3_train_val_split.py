import os
import random
import json
# set folder paths
label = "via_region_data.json"
train_folder = "chimei_t_v/train/"
train_label = "via_region_data_train.json"
val_folder = "chimei_t_v/val/"
val_label = "via_region_data_val.json"
# initial
train_dict = {}
val_dict = {}
val_keys = []
# shuffle all images
image_list = [i for i in os.listdir(train_folder) if not i.startswith('.')]
print(image_list[:4])
random.shuffle(image_list)
print(image_list[:4])

print("total images:", len(image_list))
val_ratio = int(0.1 * len(image_list))
if (len(image_list) - val_ratio) % 2 != 0:
    val_ratio = val_ratio - 1
# move val images to val folder
for i in image_list[:val_ratio]:
    os.rename(os.path.join(train_folder, i), os.path.join(val_folder, i))
    val_keys.append(i)
# open the original label json file
with open(label) as json_file:
    data = json.load(json_file)
# seperate train and val dict
for k,v in data.items():
    if k in val_keys:
        val_dict[k] = v
    else:
        train_dict[k] = v
# generate train and val label json file
with open(train_label, 'w') as outfile:
    json.dump(train_dict, outfile)

with open(val_label, 'w') as outfile:
    json.dump(val_dict, outfile)

print("total train images:", len([i for i in os.listdir(train_folder) if not i.startswith('.')]))
print("total train label images:", len(train_dict))
print("total val images:", len([i for i in os.listdir(val_folder) if not i.startswith('.')]))
print("total val label images:", len(val_dict))
