FROM ubuntu:18.04

# the great update
RUN apt-get update

# install python3, pip3
RUN apt-get install python3-pip -y --fix-missing
