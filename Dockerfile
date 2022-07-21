# Start FROM Nvidia TensorFlow image https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow
FROM nvcr.io/nvidia/tensorflow:20.12-tf1-py3
# Install linux packages
RUN apt-get update
# Install python dependencies
RUN pip install --upgrade pip
RUN pip --no-cache-dir install keras==2.3.1
RUN pip install -U git+https://github.com/albu/albumentations --no-cache-dir
RUN pip install -U --pre segmentation-models --user
RUN mkdir /workspace/chimei
WORKDIR /workspace/chimei
RUN echo 'jupyter notebook' > /run.sh && \
    chmod 755 /run.sh
CMD /run.sh
