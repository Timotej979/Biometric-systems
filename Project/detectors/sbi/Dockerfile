# Use the default image with some additional commands
FROM nvidia/cuda:11.4.1-runtime-ubuntu20.04

# Install python3
RUN apt-get -y update && \
    apt-get -y install python3

# Install pip, system dependencies and git
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y libgl1-mesa-dev && \
    apt-get install -y cmake && \
    apt-get -y install python3-pip && \
    apt-get -y update && apt-get install -y libopencv-dev \
    apt-get install -y git

# Install pytorch
RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip3 install dlib imutils scipy pandas opencv-python tqdm pretrainedmodels imgaug efficientnet_pytorch&&\
    pip3 install -U retinaface_pytorch &&\
    apt-get clean
    
# Change back to the original working directory
WORKDIR /app

# Copy the requirements
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy the rest of the files
COPY . /app/detector

# Set the working directory for the application
WORKDIR /

# Set the entrypoint
ENTRYPOINT [ "sh", "-c", "python3 /app/detector/run.py && exit 0"]