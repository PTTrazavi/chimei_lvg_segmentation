### switch between line 22 and 23 for different image folders
from PIL import Image, ImageDraw
import os
import json
from shutil import copyfile

total_image_count = 0
empty_json = []
# set labels
background = 0
heart = 1 # 1
# set folder paths
train_folder = "chimei_t_v/train/"
train_label_folder = "chimei_t_v/trainannot/"
image_root = "images/"

# get folder names
image_root_list = [f for f in os.listdir(image_root) if not f.startswith('.')]
print("total image folders:", len(image_root_list))
# loop through each folder
for folder in image_root_list:
    image_folder = os.path.join(image_root, folder)
    # image_folder = os.path.join(image_root, folder, "images")
    # don't include hidden file in mac
    image_list = [f for f in os.listdir(image_folder) if not f.startswith('.')]
    print("total images ", folder, ":", len(image_list)/2)
    # loop through each image
    for i in image_list:
        if i.endswith('.json'):
            # get label shape
            with open(os.path.join(image_folder,i)) as f:
                jf = json.load(f)
                # print("generating",folder,i)
                # make sure there is shape in the dictionary
                if len(jf['shapes']) != 0:
                    points_list = jf['shapes'][0]['points']
                    points_tuple = []
                    for p in points_list:
                        points_tuple.append(tuple(p))
                else:
                    empty_json.append(str(folder + '/' + i))
                    continue
            # draw a png label image
            originalImg = Image.open(os.path.join(image_folder,i.split('.')[0]+'.jpg'))
            labelImg = Image.new("L", originalImg.size, background)
            drawer = ImageDraw.Draw(labelImg)
            drawer.polygon(points_tuple, fill=heart)
            labelImg.save(os.path.join(train_label_folder,folder + '_' + i.split('.')[0]+'.png'))
            # copy the original image to training folder
            copyfile(os.path.join(image_folder,i.split('.')[0]+'.jpg'),
                    os.path.join(train_folder,folder + '_' + i.split('.')[0]+'.png'))
            total_image_count = total_image_count + 1

print("converted", total_image_count, "images")
print("empty json file:", empty_json)
