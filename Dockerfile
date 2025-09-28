FROM python:3.13

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8000

CMD [ "fastapi run api.py" ]