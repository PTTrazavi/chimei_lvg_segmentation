import os
import json

# set file paths
json_file_1 = "via_region_data1.json"
json_file_2 = "via_region_data2.json"
json_file_4 = "via_region_data4.json"
json_file_5 = "via_region_data5.json"

with open(json_file_1) as json_file_1:
    data_1 = json.load(json_file_1)
    print("total images:", len(data_1))

with open(json_file_2) as json_file_2:
    data_2 = json.load(json_file_2)
    print("total images:", len(data_2))

with open(json_file_4) as json_file_4:
    data_4 = json.load(json_file_4)
    print("total images:", len(data_4))

with open(json_file_5) as json_file_5:
    data_5 = json.load(json_file_5)
    print("total images:", len(data_5))

# Merge contents of data_2 in data_1
data_1.update(data_2)
print("total images:", len(data_1))

# Merge contents of data_4 in data_1
data_1.update(data_4)
print("total images:", len(data_1))

# Merge contents of data_5 in data_1
data_1.update(data_5)
print("total images:", len(data_1))

# generate the label json file
with open("via_region_data.json", 'w') as outfile:
    json.dump(data_1, outfile)
