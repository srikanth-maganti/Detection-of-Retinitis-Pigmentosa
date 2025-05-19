FROM python:3.12
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8000
CMD gunicorn --bind 0.0.0.0:$PORT app:app