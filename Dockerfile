FROM python:3.9-slim

WORKDIR /app

COPY weights ./weights/
COPY dataloaders ./dataloaders/
COPY models ./models/
COPY sample_images ./sample_images/
COPY train_model.py ./
COPY test_model.py ./
COPY kaggle.json ./
COPY requirements.txt ./

RUN apt-get update && apt-get install -y --no-install-recommends unzip && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
RUN mkdir -p ~/.kaggle
RUN cp kaggle.json ~/.kaggle/
RUN ls ~/.kaggle
RUN chmod 600 /root/.kaggle/kaggle.json
RUN mkdir data
RUN cd data/ && kaggle datasets download -d moltean/fruits && unzip -u fruits.zip

CMD /usr/local/bin/shell.sh
