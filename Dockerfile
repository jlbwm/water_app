FROM tensorflow/tensorflow:1.14.0-py3
MAINTAINER "Jason Li <jlbwm@mail.missouri.edu>"

WORKDIR /usr/src/water_app

COPY . .
RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo
RUN apt-get update || apt-get update
RUN ["apt-get", "install", "-y", "libsm6", "libxext6", "libxrender-dev"]
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install .
CMD gunicorn waterquality_Flask:app -c ./gunicorn.conf.py

