# how to start up the application

1. run `sudo systemctl restart nginx`
2. run `sudo supervisorctl reload`

# Fot testing purpose

1. cd to water_app folder
2. run `gunicorn waterquality_Flask:app -c ./gunicorn.conf.py`

# nginx site-enable config

- `sudo vim /etc/nginx/site-enable/water_quality_flask`

- basic config
```
server {
        listen 80;
        server_name 34.73.100.1;

        location /static {
                alias /home/ljx477/water_app/waterquality_Flask/static;
        }


        location / {
                proxy_pass http://localhost:8000;
                include /etc/nginx/proxy_params;
                proxy_redirect off;
        }
}
```

# supervisor config
- `sudo vim /etc/supervisor/conf.d/waterquality_app.conf`

- basic config
```
[program:waterquality_app]
directory=/home/ljx477/water_app
command=/home/ljx477/miniconda/bin/gunicorn waterquality_Flask:app -c ./gunicorn.conf.py
user=ljx477
autostart=true
autorestart=true
stopasgroup=true
killasgroup=true
stderr_logfile=/var/log/waterquality_app/waterquality_app.err.log
stdout_logfile=/var/log/waterquality_app/waterquality_app.out.log
```

# check gunicorn occupation

`ps -ax |grep gunicorn`

# docker

1. docker build `sudo docker build -t mywater .`
2. docker run  `docker run -i --name water-1 -p 8000:8000 mywater`
