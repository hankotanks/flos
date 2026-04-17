FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# setup timezone (taken from a ros image)
RUN echo 'Etc/UTC' > /etc/timezone && \
  ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
  apt-get update && \
  apt-get install -q -y --no-install-recommends tzdata && \
  rm -rf /var/lib/apt/lists/*

# setup environment
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-setuptools python-is-python3 \
    lsb-release curl gnupg2 dirmngr \
    build-essential git wget sudo unzip ninja-build \
    zsh tmux \
    && rm -rf /var/lib/apt/lists/*

# the following is from osrf noetic image
# setup keys
RUN set -eux; \
  key='C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654'; \
  export GNUPGHOME="$(mktemp -d)"; \
  gpg --batch --keyserver keyserver.ubuntu.com --recv-keys "$key"; \
  mkdir -p /usr/share/keyrings; \
  gpg --batch --export "$key" > /usr/share/keyrings/ros1-latest-archive-keyring.gpg; \
  gpgconf --kill all; \
  rm -rf "$GNUPGHOME"

# setup sources.list
RUN echo "deb [ signed-by=/usr/share/keyrings/ros1-latest-archive-keyring.gpg ] http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros1-latest.list

# add ROS repository
#RUN curl -sSL "http://packages.ros.org/ros.key" | apt-key add - \
#    && echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list

# install minimal ROS noetic + catkin + python tools
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     ros-noetic-ros-core \
#     python3-catkin-tools python3-rosdep python3-rosinstall python3-vcstools \
#     && rm -rf /var/lib/apt/lists/*

# install ROS and related packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-core \
    python3-catkin-tools python3-rosdep python3-rosinstall python3-vcstools \
    ros-noetic-roscpp \
    ros-noetic-std-msgs \
    ros-noetic-tf2 \
    ros-noetic-tf2-ros \
    ros-noetic-sensor-msgs \
    ros-noetic-nav-msgs \
    ros-noetic-geometry-msgs \
    ros-noetic-message-generation \
    ros-noetic-message-runtime \
    ros-noetic-rospy \
    ros-noetic-rosbag \
    && rm -rf /var/lib/apt/lists/*

# init rosdep
RUN rosdep init || true
RUN rosdep update

# setup ROS environment
RUN echo "source /opt/ros/noetic/setup.bash" >> /etc/bash.bashrc

# upgrade pip and install PyTorch for CUDA 11.8
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install --no-cache-dir \
    "numpy<=1.25" \
    "networkx<=3.1" \
    torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
    --index-url https://download.pytorch.org/whl/cu118

# create a non-root user
ARG UNAME=dev
ARG UID
ARG GID
RUN addgroup --gid $GID $UNAME \
    && adduser --disabled-password --uid $UID --gid $GID --gecos '' $UNAME \
    && adduser $UNAME sudo \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER $UNAME
WORKDIR /home/$UNAME
ENV HOME=/home/$UNAME

# once it gets merged in, use the master
RUN sudo sh -c "$(curl -L https://raw.githubusercontent.com/balancedscorpion/zsh-in-docker/new_user/zsh-in-docker.sh)" -- \
  -u ${UNAME} \
  -x \
  -p git -p z -p sudo -p history -p copybuffer -p copypath -p copyfile \
  -p https://github.com/zsh-users/zsh-autosuggestions \
  -p https://github.com/zsh-users/zsh-completions

RUN sudo chsh -s $(which zsh) ${UNAME}
ENV SHELL=/usr/bin/zsh
SHELL ["/bin/zsh", "-lc"]

RUN echo "export HISTFILE=/home/${UNAME}/.zsh_history" >> /home/${UNAME}/.zshrc \
  && echo "ros-env() {" >> /home/${UNAME}/.zshrc \
  && echo "    source /opt/ros/noetic/setup.zsh" >> /home/${UNAME}/.zshrc \
  && echo "    source /home/${UNAME}/ros_ws/devel/setup.zsh 2>/dev/null || true" >> /home/${UNAME}/.zshrc \
  && echo "}" >> /home/${UNAME}/.zshrc \
  && echo "ros-env" >> /home/${UNAME}/.zshrc \
  && echo 'export ROS_LANG_DISABLE=genlisp:gennodejs:geneus' >> /home/${UNAME}/.zshrc 

ENV PATH="/home/${UNAME}/.local/bin:${PATH}"

# create catkin workspace
RUN mkdir -p /home/${UNAME}/ros_ws/src
WORKDIR /home/${UNAME}/ros_ws
RUN /bin/zsh -c "catkin init && catkin config -DPYTHON_EXECUTABLE=$(which python3)"

CMD ["/bin/zsh"]
