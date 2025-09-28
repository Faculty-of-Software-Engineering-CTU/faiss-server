FROM python:3.13

COPY . /app

RUN pip install

EXPOSE 8000

CMD [ "fastapi run api.py" ]