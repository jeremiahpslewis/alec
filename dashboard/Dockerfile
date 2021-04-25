FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt src/ /app/

RUN apt-get update && apt-get install gcc && \
    pip install --upgrade pip  && \
    pip install -r requirements.txt

CMD ["streamlit", "hello"]
