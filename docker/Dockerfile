FROM nvidia/vulkan:1.2.133-450

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get --allow-insecure-repositories --allow-unauthenticated update 
RUN apt install -y --allow-unauthenticated cmake libeigen3-dev libopencv-dev libzmqpp-dev 

RUN apt install -y git

