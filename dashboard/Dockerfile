FROM python:3.9-slim-buster

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential gcc && apt-get clean

COPY .streamlit/ .streamlit/

COPY src/requirements.txt /app/

RUN pip install --upgrade pip  && \
    pip install -r requirements.txt

COPY src/ /app/

CMD ["streamlit", "run", "dashboard.py"]
