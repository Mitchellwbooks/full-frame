FROM python:3.11
LABEL authors="mitchellwbooks"

RUN pip install -r requirements.txt

RUN mkdir /pictures
RUN mkdir /data

ENTRYPOINT ["python", "main.py"]