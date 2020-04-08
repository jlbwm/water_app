#!/usr/bin/env python3
#coding:utf8
import redis,sys

def api_count_check(IP):
    key=IP+':api_name:api_count'
    # pool = redis.ConnectionPool(host='10.206.120.22', port=5022)
    # r = redis.Redis(connection_pool=pool)

    # r = redis.Redis(host='10.206.120.22', port=5022)
    # r = redis.Redis(host='localhost', port=8888)
    # pool = redis.ConnectionPool(host='192.168.64.18', port=6379,db=0,password=123)
    r = redis.Redis()
    limit=100
    expired_time=60
    check=r.exists(key)
    isExcced=0
    if check==True:
        r.incr(key)
        count = int(r.get(key))
        if count > limit:
            isExcced=1
            print("Excceded! Permission Denied!")
            sys.exit(0)
    else:
        r.set(key,1)
        r.expire(key,expired_time)
    return isExcced

if __name__ == '__main__':
    for i in range(1,101):
     	print(api_count_check('user1'))
  