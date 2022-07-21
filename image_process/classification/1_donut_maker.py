from PIL import Image, ImageDraw
import os
import json
from skimage import io
import numpy as np

# utils
def draw_combined_color(mask_image):
    colors = {0:0,      #bg 0
              1:250,    #diastole 1
              -1:100}    #systole -1

    out_image = np.zeros_like(mask_image)

    for k,v in colors.items():
        class_mask = (mask_image==k) # get the layer of class
        class_mask = class_mask * colors[k]
        out_image = out_image + class_mask

    out_image= out_image.astype(int) # make it int so plt.imshow can show rgb 0-255, float will show 0-1
    return out_image

target = "images5/"

folder_list = [f for f in os.listdir(target) if not f.startswith('.')]

for folder in folder_list:
    print("processing folder ", folder , "...")
    # set folder paths
    image_folder = os.path.join(target, folder, "images")
    donut_name = folder + ".png"

    total_image_count = 0
    empty_json = []
    # set labels
    background = 0
    heart = 1 # 1

    # get image names
    image_list = [f for f in os.listdir(image_folder) if not f.startswith('.')]
    print("total images:", len(image_list)/2)
    # loop through the folder
    for i in image_list:
        if i.endswith('.json'):
            # get label shape
            with open(os.path.join(image_folder,i)) as f:
                jf = json.load(f)
                # make sure there is shape in the dictionary
                if len(jf['shapes']) != 0:
                    points_list = jf['shapes'][0]['points']
                    points_tuple = []
                    for p in points_list:
                        points_tuple.append(tuple(p))
                else:
                    empty_json.append(str(i))
                    continue
            # draw a png label image
            originalImg = Image.open(os.path.join(image_folder,i.split('.')[0]+'.jpg'))
            labelImg = Image.new("L", originalImg.size, background)
            drawer = ImageDraw.Draw(labelImg)
            drawer.polygon(points_tuple, fill=heart)
            labelImg.save(os.path.join(image_folder, i.split('.')[0]+'.png')) # folder + '_' +

            total_image_count = total_image_count + 1

    print("converted", total_image_count, "images")
    print("empty json file:", empty_json)

    if total_image_count > 1:
        # get mask names
        mask_list = [f for f in os.listdir(image_folder) if '.png' in f]
        mask_list.sort()
        mask_dict = dict(enumerate(mask_list))
        # get the mask size
        image = io.imread(os.path.join(image_folder, mask_list[0]))
        width = image.shape[0]
        height = image.shape[1]
        # count the pixel of heart
        heart_max = 0
        heart_min = width*height
        max_file = str()
        min_file = str()
        max_key = int()
        min_key = int()

        # get the max area image
        for k,i in enumerate(mask_list):
            mask = io.imread(os.path.join(image_folder, i)) # novel image
            # check if the size is 512*512
            if mask.shape[0] != 512 or mask.shape[1] != 512:
                print("the size is not right!")

            # count the pixels of heart(label==1)
            heart = np.sum(mask==1)
            # find max
            if heart > heart_max:
              heart_max = heart
              max_file = i
              max_key = k

        # get the min area image
        for k,i in enumerate(mask_list):
            mask = io.imread(os.path.join(image_folder, i)) # novel image
            # check if the size is 512*512
            if mask.shape[0] != 512 or mask.shape[1] != 512:
                print("the size is not right!")

            # count the pixels of heart(label==1)
            heart = np.sum(mask==1)
            # find min
            if heart < heart_min: # and k > max_key
              heart_min = heart
              min_file = i
              min_key = k

        # check the results
        print("max_file is:", max_file)
        print("min_file is:", min_file)

        if len(min_file) < 4:
            print("skip this folder because there is no min image!!!!!!!!!")
            continue
        else:
            # min image process
            mask_min = io.imread(os.path.join(image_folder, min_file))
            mask_min = mask_min * -1 # make the systole label as -1 for later calculation
            unique, counts = np.unique(mask_min, return_counts=True)
            print("min:", dict(zip(unique, counts)))

            # max image process
            mask_max = io.imread(os.path.join(image_folder, max_file))
            unique, counts = np.unique(mask_max, return_counts=True)
            print("max:", dict(zip(unique, counts)))

            # combine two images
            mask_combined = mask_min + mask_max
            unique, counts = np.unique(mask_combined, return_counts=True)
            print("combined:", dict(zip(unique, counts)))

            # draw the image
            mask_combined_draw = draw_combined_color(mask_combined)
            io.imsave(os.path.join(target, donut_name), mask_combined_draw)
