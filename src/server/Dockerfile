FROM python:3.11
LABEL authors="mitchellwbooks"

RUN pip install -r requirements.txt

# SEE: https://onnxruntime.ai/docs/get-started/training-pytorch.html
RUN python -m torch_ort.configure

RUN mkdir /pictures
RUN mkdir /data

ENTRYPOINT ["python", "main.py"]