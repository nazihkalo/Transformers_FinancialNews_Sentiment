FROM python:3.6-slim
COPY ./app.py /deploy/
COPY ./serve.py /deploy/
COPY ./requirements2.txt /deploy/
COPY ./finbert-sentiment /deploy/finbert-sentiment/
COPY ./bert_sentiment_utils.py /deploy/
COPY ./templates /deploy/templates/
WORKDIR /deploy/
RUN pip install -r requirements2.txt
EXPOSE 80
ENTRYPOINT ["python", "app.py"]