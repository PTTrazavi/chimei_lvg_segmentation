### switch between line 23 and 24 for different image folders
from PIL import Image, ImageDraw
import os
import json
from shutil import copyfile

total_image_count = 0
label_json = {}
empty_json = []
# set labels
# background = 1
heart = 0 # 0
# set folder paths
train_folder = "chimei_t_v/train/"
train_label_file = "via_region_data5.json"
image_root = "images5/"

# get folder names
image_root_list = [f for f in os.listdir(image_root) if not f.startswith('.')]
print("total image folders:", len(image_root_list))
# loop through each folder
for folder in image_root_list:
    # image_folder = os.path.join(image_root, folder)
    image_folder = os.path.join(image_root, folder, "images")
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
                    all_points_x = []
                    all_points_y = []
                    for p in points_list:
                        all_points_x.append(p[0])
                        all_points_y.append(p[1])
                else:
                    empty_json.append(str(folder + '/' + i))
                    continue
            # append label image to json label_json dict
            label_json[folder + '_' + i.split('.')[0]+'.png'] = {"filename": folder + '_' + i.split('.')[0]+'.png',
                                                                 "regions":
                                                                 {"0":
                                                                    {"shape_attributes":
                                                                        {"name":"polygon",
                                                                            "all_points_x":all_points_x,
                                                                            "all_points_y":all_points_y,
                                                                        },
                                                                     "region_attributes":
                                                                        {"heart":heart
                                                                        }
                                                                    }
                                                                  }
                                                                 }
            # copy the original image to training folder
            copyfile(os.path.join(image_folder,i.split('.')[0]+'.jpg'),
                    os.path.join(train_folder,folder + '_' + i.split('.')[0]+'.png'))
            total_image_count = total_image_count + 1
# generate the label json file
with open(train_label_file, 'w') as outfile:
    json.dump(label_json, outfile)

print("converted", total_image_count, "images")
print("empty json file:", empty_json)
