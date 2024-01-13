FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

# Set the timezone non-interactively, othherwise tzdata will ask for input
ARG TZ=UTC
ENV DEBIAN_FRONTEND=noninteractive
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Set the working directory
WORKDIR /app

# Install aptitude dependencies
RUN apt-get -y update && \
    apt-get -y install software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get -y update && \
    apt-get -y install python3.8

# Set Python 3.8 as the default python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --set python3 /usr/bin/python3.8

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y libgl1-mesa-dev cmake python3-pip libopencv-dev git

# Install Python dependencies
RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip3 install dlib imutils scipy pandas opencv-python tqdm pretrainedmodels imgaug efficientnet-pytorch && \
    pip3 install -U retinaface-pytorch && \
    apt-get clean

# Copy the rest of the files
COPY . /app/detector

# Set the working directory for the application
WORKDIR /

# Set the entrypoint
ENTRYPOINT [ "sh", "-c", "python3 /app/detector/run.py && exit 0"]