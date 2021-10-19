FROM python:3.8
WORKDIR /app
COPY ./app/requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
