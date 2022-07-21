# Chimei hospital heart semantic segmentation project
## colab_code folder  
The code in this folder works on google colab.  

## video_to_images folder  
The code in this folder can extract images from .dcm file or .avi file.  

## image_process folder  
The code in this folder can generate the required images and label files for each model.  

## checkpoints folder  
This folder contains the trained model files for each model.  

## How to use Docker
1. build image from Dockerfile
```bash
docker build [path to a Dockerfile] -t [tag name]
```
2. creat a container from image
```bash
docker run -itd --gpus all --shm-size 16g --rm --name [container name] -v [local folder path]:/workspace/chimei/ -p 8000:8888 [tag name]
```
3. enter the terminal of the container
```bash
docker exec -it [container ID] bash
```
4. check the jupyter notebook address and token
```bash
jupyter notebook list
```
