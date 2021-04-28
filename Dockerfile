FROM python:3.9-slim

WORKDIR /app


RUN pip install -q kaggle
RUN mkdir -p ~/.kaggle
RUN cp kaggle.json ~/.kaggle/
RUN ls ~/.kaggle
RUN chmod 600 /root/.kaggle/kaggle.json
RUN mkdir data
RUN cd data/ && kaggle datasets download -d moltean/fruits && unzip -u fruits.zip


COPY transformers_pytorch ./transformers_pytorch/

RUN pip install upgrade pip && pip install --no-cache-dir -r requirements.txt



CMD ['bash']
