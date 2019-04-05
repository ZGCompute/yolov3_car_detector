FROM jjanzic/docker-python3-opencv

WORKDIR /usr/src/app

COPY requirements.txt .
RUN apt-get update
RUN apt-get -y install libav-tools
RUN pip install -r requirements.txt

COPY cars.py .
COPY src ./src

ENTRYPOINT ["python3", "cars.py"]
