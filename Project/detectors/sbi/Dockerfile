# Use prebuilt pytorch image as base, change tag as needed if you want to use cuda
# Source: https://github.com/cnstark/pytorch-docker

# Available tags:
#  - 1.7.0-py3.8.13-cuda11.0.3-ubuntu18.04
#  - 1.7.0-py3.8.13-cuda11.0.3-devel-ubuntu18.04
#  - 1.7.0-py3.8.13-ubuntu18.04

FROM cnstark/pytorch:1.7.0-py3.8.13-cuda11.0.3-devel-ubuntu18.04

# Set the timezone non-interactively, othherwise tzdata will ask for input
ARG TZ=UTC
ENV DEBIAN_FRONTEND=noninteractive
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Set the working directory
WORKDIR /app

# Install aptitude dependencies
RUN apt-get -y update && \
    apt-get -y install python3

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y libgl1-mesa-dev && \
    apt-get install -y cmake && \
    apt-get -y install python3-pip && \
    apt-get -y update && apt-get install -y libopencv-dev && \
    apt-get install -y git

# Install python dependencies
RUN pip3 install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip3 install dlib imutils scipy pandas opencv-python tqdm pretrainedmodels imgaug efficientnet_pytorch scikit-learn&&\
    pip3 install -U retinaface_pytorch &&\
    apt-get clean

# Copy the rest of the files
COPY . /app/detector

# Set the working directory for the application
WORKDIR /

# Set the entrypoint
ENTRYPOINT [ "sh", "-c", "python3 /app/detector/run.py && exit 0"]