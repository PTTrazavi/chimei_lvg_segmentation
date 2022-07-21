from PIL import Image
import os

# get folder names
image_folder = "images5/"
image_folder_list = [f for f in os.listdir(image_folder) if not f.startswith('.')]
print("total image folders:", len(image_folder_list))

print(image_folder_list[:3])

# loop through each folder and delete jpg without json
for folder in image_folder_list:
    image_list = [i for i in os.listdir(os.path.join(image_folder, folder)) if not i.startswith('.')]
    # image_list = [i for i in os.listdir(os.path.join(image_folder, folder, "images")) if not i.startswith('.')]
    count = 0
    for i in image_list:
        if i.endswith('.jpg') and (i.split('.')[0]+'.json') not in image_list:
            os.remove(os.path.join(image_folder, folder, i))
            # os.remove(os.path.join(image_folder, folder, "images", i))
            count = count + 1
    print("deleted", count, "images in folder", folder)

# # loop through each folder and delete png
# for folder in image_folder_list:
#     image_list = [i for i in os.listdir(os.path.join(image_folder, folder)) if not i.startswith('.')]
#     # image_list = [i for i in os.listdir(os.path.join(image_folder, folder, "images")) if not i.startswith('.')]
#     count = 0
#     for i in image_list:
#         if i.endswith('.png'):
#             os.remove(os.path.join(image_folder, folder, i))
#             # os.remove(os.path.join(image_folder, folder, "images", i))
#             count = count + 1
#     print("deleted", count, "images in folder", folder)
