FROM tensorflow/tensorflow:2.1.0-py3
MAINTAINER "Jason Li <jlbwm@mail.missouri.edu>"

WORKDIR /usr/src/water_app

COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install .
CMD gunicorn waterquality_Flask:app -c ./gunicorn.conf.py