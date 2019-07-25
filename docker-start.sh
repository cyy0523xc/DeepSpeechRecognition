#!/bin/bash
# 
# 容器启动
# Author: alex
# Created Time: 2019年07月25日 星期四 17时51分49秒

# 语料库地址
corpus=/home/ibbd/corpus/

sudo docker rm -f deepspeech
sudo docker run --rm -ti \
    --name=deepspeech \
    -v "$PWD":/speech/ \
    -v "$corpus":/corpus/ \
    -w /speech/ \
    --runtime=nvidia \
    registry.cn-hangzhou.aliyuncs.com/ibbd/speech:cu101-py36-u1804-tf \
    /bin/bash
