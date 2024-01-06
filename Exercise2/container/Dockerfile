FROM ubuntu:16.04

# Install OpenCV and OpenBR dependencies
RUN apt-get update && apt-get upgrade -y && \
    # Install Python 2.7 and 3.7
    apt-get install -y python2.7-dev python3-dev python3-tk \
	  # Git for version control
    git \
	  # Build essentials for compiling code and software properties common
    build-essential \
    software-properties-common \
	  # Build libraries
    cmake \
	  # Package configuration tool
    pkg-config \
    # Development libraries for LDAP/SMI/FFI
    libldap2-dev libsasl2-dev \
    libsmi2-dev \
    libffi-dev \
    # Development libraries for handling TIFF/PNG/JPEG/scaling images
    libtiff-dev \
    libpng-dev \
    libjpeg-dev \
    libswscale-dev \
    # Unzip utility for extracting compressed files
    unzip \
    # Development libraries for TBB (Threading Building Blocks)
    libtbb-dev libtbb2 \
    # Install required Qt5 libraries
    qt5-default \
    libqt5svg5-dev \
    qtcreator \
    # Wget for downloading files and curl for transferring data
    wget \
    curl \
    # Utility for creating Debian packages from source code
    checkinstall \
    # Development libraries for readline support
    libreadline-gplv2-dev \
    # Development libraries for handling terminal input
    libncursesw5-dev \
    # Additional packages for compiling Python 3.8.10 from source
    libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev && \
    # Install python2.7-pip and python3.7-pip
    curl https://bootstrap.pypa.io/pip/3.5/get-pip.py -o ./get-pip3.5.py && \
    curl https://bootstrap.pypa.io/pip/2.7/get-pip.py -o ./get-pip2.7.py && \
    python2.7 get-pip2.7.py && \
    python3.5 get-pip3.5.py

# Install OpenCV
RUN cd \
  && wget https://github.com/opencv/opencv/archive/2.4.11.zip \
	&& unzip 2.4.11.zip \
	&& cd opencv-2.4.11 \
	&& mkdir build \
	&& cd build \
	&& cmake .. \
	&& make -j10 \
	&& make install \
	&& cd \
	&& rm 2.4.11.zip

# Install OpenBR
RUN cd \
  && git clone https://github.com/biometrics/openbr.git \
	&& cd openbr \
  && git checkout v1.1.0 \
  && git submodule init \
  && git submodule update \
	&& mkdir build \
	&& cd build \
	&& cmake -DCMAKE_BUILD_TYPE=Release -DBR_INSTALL_BRPY=ON .. \
	&& make -j10 \
	&& make install

# Copy the requirements
COPY requirements.txt /requirements.txt

RUN pip3 install --upgrade pip

# Install Python 3.5 dependencies
RUN pip3 install -r requirements.txt