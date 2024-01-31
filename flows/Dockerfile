FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

ENV TF_DISABLE_GPU=1
ENV CUDA_VISIBLE_DEVICES -1


COPY . /app

CMD ["python", "deployment.py"]