FROM python:3.6-slim
COPY ./app.py /deploy/
COPY ./serve.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./finbert-sentiment /deploy/finbert-sentiment
COPY ./bert_sentiment_utils.py /deploy/
COPY ./templates /deploy/templates
WORKDIR /deploy/
RUN pip install -r requirements.txt
EXPOSE 80
ENTRYPOINT ["python", "app.py"]