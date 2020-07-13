workers = 1
worker_class = "gevent"
timeout = 300
# gunicorn default timeout is 30s, set to 5 min (300) for multi-upload
bind = "0.0.0.0:8001"