# Use prebuilt pytorch image as base, change tag as needed if you want to use cuda
# Source: https://github.com/cnstark/pytorch-docker

# Available tags:
#  - 1.7.0-py3.8.13-cuda11.0.3-ubuntu18.04
#  - 1.7.0-py3.8.13-cuda11.0.3-devel-ubuntu18.04
#  - 1.7.0-py3.8.13-ubuntu18.04

FROM cnstark/pytorch:1.7.0-py3.8.13-ubuntu18.04

# Set the working directory
WORKDIR /app

# Install python dependencies
RUN apt-get update -y && \
    apt-get upgrade -y && \
    python3 -m pip install --upgrade pip

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