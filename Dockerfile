FROM ubuntu:latest

COPY ./requirements.txt /requirements.txt
RUN apt-get update \
	&& apt-get install gcc -y \
	&& apt-get clean
RUN apt install -y python3-pip
RUN pip3 install -r  /requirements.txt
RUN mkdir /app
WORKDIR /app
COPY ./app /app