from PIL import Image
import os

# get folder names
root_folder = "images5/"
image_folders = [i for i in os.listdir(root_folder) if not i.startswith('.')]
print("total folders:", len(image_folders))

print(image_folders[:3])
count = 0

# # loop through each folder and delete .dcm files
# for image_folder in image_folders:
#     files = [i for i in os.listdir(os.path.join(root_folder, image_folder)) if not i.startswith('.')]
#     for file in files:
#         if file.endswith('.dcm'):
#             os.remove(os.path.join(root_folder, image_folder, file))
#             count = count + 1
# print("deleted", count, "dcm files")

# loop through each folder and non-labelled files
for image_folder in image_folders:
    files = [i for i in os.listdir(os.path.join(root_folder, image_folder, 'images')) if not i.startswith('.')]
    for file in files:
        if file.endswith('.jpg') and (file.split('.')[0]+'.json') not in files:
            os.remove(os.path.join(root_folder, image_folder, 'images', file))
            count = count + 1
    print("deleted", count, "images in folder", image_folder)
    count = 0
