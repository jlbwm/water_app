FROM tensorflow/tensorflow:1.12.3-py3
MAINTAINER "Jason Li <jlbwm@mail.missouri.edu>"

WORKDIR /usr/src/water_app

COPY . .
RUN ["apt-get", "install", "-y", "libsm6", "libxext6", "libxrender-dev"]
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install .
CMD gunicorn waterquality_Flask:app -c ./gunicorn.conf.py

