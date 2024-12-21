#!/bin/bash

set -e

sudo apt-get update
sudo apt-get install -y \
        libgoogle-glog-dev \
        v4l-utils \
     		dialog \
     		libglew-dev \
     		glew-utils \
     		gstreamer1.0-libav \
     		gstreamer1.0-nice \
     		libgstreamer1.0-dev \
     		libgstrtspserver-1.0-dev \
     		libglib2.0-dev \
     		libsoup2.4-dev \
     		libjson-glib-dev \
     		avahi-utils \
     		libgstreamer-plugins-base1.0-dev \
        libgstreamer-plugins-good1.0-dev \
        libgstreamer-plugins-bad1.0-dev
