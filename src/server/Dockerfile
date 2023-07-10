FROM python:3.11-bullseye
LABEL authors="mitchellwbooks"

COPY ./requirements.txt  /requirements.txt

RUN pip install -r /requirements.txt

RUN apt-get update
RUN apt install apt-transport-https
RUN apt install -y exempi

# SEE: https://onnxruntime.ai/docs/get-started/training-pytorch.html
# RUN python -m torch_ort.configure

COPY .  /code

RUN mkdir /pictures
RUN mkdir /data
RUN mkdir /thumbnails

ENTRYPOINT ["python", "-u", "/code/main.py"]