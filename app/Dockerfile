FROM ros:humble-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/app/.cache/matplotlib \
    ROS_LOG_DIR=/app/.cache/ros_log \
    QT_QPA_PLATFORM=offscreen \
    QT_X11_NO_MITSHM=1 \
    PYTHONPATH=/app/src:/app

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libgl1 \
    libglib2.0-0 \
    libusb-1.0-0 \
    procps \
    psmisc \
    python3-flask \
    python3-matplotlib \
    python3-numpy \
    python3-opencv \
    python3-pil \
    python3-pip \
    python3-pytest \
    python3-pyqt5 \
    python3-scipy \
    python3-yaml \
    ros-humble-libcaer-vendor \
    ros-humble-realsense2-camera \
    ros-humble-realsense2-camera-msgs \
    usbutils \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends libjs-three \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir pyaer pyrealsense2

WORKDIR /app
ENV USE_YOLO_BOTTLE_TRIGGER=0 \
    SINGLE_BOTTLE_DEMO_MODE=1 \
    PRELOCK_FAST_SCALED_RGBD=1 \
    YOLO_MODEL_PATH=models/yolo_bottle.weights \
    YOLO_CONF_THRESHOLD=0.25
RUN mkdir -p models
COPY . /app

RUN mkdir -p /app/output /app/.cache/matplotlib /app/.cache/ros_log \
    && chmod +x /app/*.sh /app/scripts/*.sh

EXPOSE 8080

CMD ["bash", "-lc", "source /opt/ros/humble/setup.bash && exec ./run_final_grasp_demo.sh --with-davis"]
