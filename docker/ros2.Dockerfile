FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

# setup timezone (taken from a ros image)
RUN echo 'Etc/UTC' > /etc/timezone && \
  ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
  apt-get update && \
  apt-get install -q -y --no-install-recommends tzdata && \
  rm -rf /var/lib/apt/lists/*

# setup environment
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# ENV DEBIAN_FRONTEND=noninteractive
# ENV PYTHON_VERSION=3.10.14
# ENV PYTHON_PREFIX=/opt/python3.10
# 
# RUN apt-get update && apt-get install -y \
#     wget build-essential libssl-dev zlib1g-dev \
#     libncurses5-dev libncursesw5-dev libreadline-dev \
#     libsqlite3-dev libffi-dev libbz2-dev liblzma-dev \
#     tk-dev uuid-dev curl xz-utils \
# 
# RUN wget "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz" && \
#     tar -xzf Python-{PYTHON_VERSION}.tgz && \
#     rm Python-{PYTHON_VERSION}.tgz
# 
# # build and install Python
# RUN cd Python-${PYTHON_VERSION} && \
#     ./configure --prefix=${PYTHON_PREFIX} --enable-optimizations && \
#     make -j$(nproc) && \
#     make install
# 
# ENV PATH="${PYTHON_PREFIX}/bin:${PATH}"

# basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential git wget curl unzip ninja-build clangd \
  lsb-release gnupg2 dirmngr \
  python3 python3-venv python3-pip python3-dev python3-setuptools python-is-python3 \
  zsh tmux \
  libegl1 libgl1 libgomp1 \  
  && rm -rf /var/lib/apt/lists/*

# setup ROS2 repo
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2.list

# install ROS2 and related packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-kilted-ros-base \
    python3-colcon-common-extensions python3-rosdep python3-vcstool \
    && rm -rf /var/lib/apt/lists/*
RUN rosdep init 2>/dev/null || true && rosdep update

# create a python virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# python dependencies (torch)
RUN python3 -m pip install --no-cache-dir \
    "numpy<=1.26.4" \
    networkx==3.1 \
    torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 torchmetrics \
    --index-url https://download.pytorch.org/whl/cu126
RUN python3 -m pip install --break-system-packages --no-cache-dir torch-scatter \
    -f https://data.pyg.org/whl/torch-2.9.1+cu126.html

# this is horrible i'm sorry
RUN python3 -m pip install --no-cache-dir catkin-pkg 
# other python deps
RUN python3 -m pip install --no-cache-dir spconv-cu126 addict lightning tensorboard typer open3d

# once it gets merged in, use the master
RUN sh -c "$(curl -L https://raw.githubusercontent.com/balancedscorpion/zsh-in-docker/new_user/zsh-in-docker.sh)" -- \
  -u root \
  -x \
  -p git -p z -p history -p copybuffer -p copypath -p copyfile \
  -p https://github.com/zsh-users/zsh-autosuggestions \
  -p https://github.com/zsh-users/zsh-completions

ENV SHELL=/usr/bin/zsh
SHELL ["/bin/zsh", "-lc"]

# this is a DISGUSTING hack to colcon to work with a virtual environment
# we call this python script to get the appropriate site-packages path in a version-agnostic way
RUN python3 -c 'import site; print([p for p in site.getsitepackages() if p.endswith("site-packages")][0])' > /tmp/packages \
    && echo "export PYTHONPATH=$(cat /tmp/packages)" >> /root/.zshrc \
    && echo "export HISTFILE=/root/.zsh_history" >> /root/.zshrc \
    && echo "ros2-env() {" >> /root/.zshrc \
    && echo "    source /opt/ros/kilted/setup.zsh" >> /root/.zshrc \
    && echo "    source /root/ros_ws/install/setup.zsh 2>/dev/null || true" >> /root/.zshrc \
    && echo "}" >> /root/.zshrc \
    && echo "ros2-env" >> /root/.zshrc

ENV PATH="/root/.local/bin:${PATH}"

# ros2 workspace
RUN mkdir -p /root/ros_ws/src
WORKDIR /root/ros_ws

# entrypoint
CMD ["/bin/zsh"]