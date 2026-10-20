FROM ros:humble

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y \
    python3-pip \
    python3-opencv \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-rqt-image-view \
    ros-humble-realsense2-camera \
    ros-humble-realsense2-camera-msgs \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install "numpy<2"

WORKDIR /workspace

CMD ["/bin/bash"]

