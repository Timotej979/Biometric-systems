FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

# Set the working directory
WORKDIR /app

# Install CUDA dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8=8.2.2.26-1+cuda11.2 \
    libcudnn8-dev=8.2.2.26-1+cuda11.2 \
    && rm -rf /var/lib/apt/lists/*

# Install aptitude dependencies
RUN apt-get -y update && \
    apt-get -y install python3

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y libgl1-mesa-dev && \
    apt-get install -y cmake && \
    apt-get -y install python3-pip && \
    apt-get -y update && apt-get install -y libopencv-dev && \
    apt-get install -y git

# Copy the requirements
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip3 install -r requirements.txt && \
    pip3 install -U retinaface_pytorch && \
    apt-get clean

# Copy the rest of the files
COPY . /app/detector

# Set the working directory for the application
WORKDIR /

# Set the entrypoint
ENTRYPOINT [ "sh", "-c", "python3 /app/detector/run.py && exit 0"]