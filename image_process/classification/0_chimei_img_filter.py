from PIL import Image
import os

# get folder names
image_folder = "images3/C4530 inferior hypo/C4530[0]_frames/"
image_list = [i for i in os.listdir(image_folder) if not i.startswith('.')]
print("total images:", len(image_list))

print(image_list[:3])
count = 0
# loop through each image
for i in image_list:
    if i.endswith('.jpg') and (i.split('.')[0]+'.json') not in image_list:
        os.remove(os.path.join(image_folder, i))
        count = count + 1
print("deleted", count, "images in folder", image_folder)
